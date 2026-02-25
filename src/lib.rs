use anyhow::{Result, anyhow};
use indicatif::ProgressBar;
use rand::SeedableRng;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::Poisson;
use std::slice::ChunksExactMut;
// ── Genetic Architecture ─────────────────────────────────────────────────────

/// A bi-allelic locus with position and per-deme selection coefficients
#[derive(Debug, Clone)]
pub struct Locus {
    pub position: f64,
    /// Selection coefficient per deme when in derived state (1)
    /// Positive = beneficial, negative = deleterious
    pub selection_coeffs: Vec<f64>,
}

/// Pre-defined genetic architecture for finite sites model
#[derive(Debug, Clone)]
pub struct GeneticArchitecture {
    pub loci: Vec<Locus>,
    pub num_demes: usize,
    // Cached SoA layout (built once in new())
    positions: Vec<f64>,  // length = num_loci, contiguous
    sel_coeffs: Vec<f64>, // length = num_loci * num_demes, flat [l * num_demes + d]
}

impl GeneticArchitecture {
    pub fn new(loci: Vec<Locus>, num_demes: usize) -> Result<Self> {
        // Validate all loci have correct number of selection coefficients
        for (i, locus) in loci.iter().enumerate() {
            if locus.selection_coeffs.len() != num_demes {
                return Err(anyhow!(
                    "Locus {} has {} selection coeffs, expected {}",
                    i,
                    locus.selection_coeffs.len(),
                    num_demes
                ));
            }
        }

        // Validate positions are sorted and unique
        for w in loci.windows(2) {
            if w[0].position >= w[1].position {
                return Err(anyhow!("Locus positions not strictly increasing"));
            }
        }

        // Build cached flat arrays
        let positions: Vec<f64> = loci.iter().map(|l| l.position).collect();
        let mut sel_coeffs = Vec::with_capacity(loci.len() * num_demes);
        for locus in &loci {
            sel_coeffs.extend_from_slice(&locus.selection_coeffs);
        }

        Ok(Self {
            loci,
            num_demes,
            positions,
            sel_coeffs,
        })
    }

    pub fn num_loci(&self) -> usize {
        self.loci.len()
    }

    #[inline]
    pub fn position(&self, l: usize) -> f64 {
        self.positions[l]
    }

    #[inline]
    pub fn positions(&self) -> &[f64] {
        &self.positions
    }

    #[inline]
    pub fn selection_coeff(&self, l: usize, d: usize) -> f64 {
        debug_assert!(l * self.num_demes + d < self.sel_coeffs.len());
        // SAFETY: l < num_loci and d < num_demes, both enforced by callers
        // and validated in new(). sel_coeffs has length num_loci * num_demes.
        unsafe { *self.sel_coeffs.get_unchecked(l * self.num_demes + d) }
    }
}

// ── Parameters ────────────────────────────────────────────────────────────────
#[derive(Clone, Debug)]
pub struct Parameters {
    pub random_seed: u64,
    pub sequence_length: f64,
    pub runtime: usize,
    pub recombination_rate: f64,
    pub mutation_rate: f64, // per-locus per-generation
    /// Simplify every this many generations (0 = only at finalize).
    pub simplify_interval: usize,
}

impl Default for Parameters {
    fn default() -> Self {
        let mut rng = rand::rng();
        let random_seed = rng.random_range(1..u64::MAX);
        Self {
            random_seed,
            sequence_length: 1e7,
            runtime: 5_000,
            recombination_rate: 1e-8,
            mutation_rate: 1e-6,
            simplify_interval: 100,
        }
    }
}

// ── Deme configuration ────────────────────────────────────────────────────────
#[derive(Clone, Debug)]
pub struct DemeConfig {
    pub carrying_capacity: usize,
    /// Fraction of K_d that emigrate each generation.
    pub migration_rate: f64,
    /// Set during `initialize`; used by `commit_to_tables`.
    pub population_id: tskit::PopulationId,
}

// ── Double-Buffered Structures ────────────────────────────────────────────────

/// Double-buffered genotype storage for one deme (flat layout for cache locality)
struct GenotypeBuffer {
    current: Vec<u8>, // K * num_loci contiguous
    next: Vec<u8>,    // K * num_loci contiguous
    num_loci: usize,
}

impl GenotypeBuffer {
    fn new(capacity: usize, num_loci: usize) -> Self {
        Self {
            current: vec![0u8; capacity * num_loci],
            next: vec![0u8; capacity * num_loci],
            num_loci,
        }
    }

    fn current_genotype(&self, i: usize) -> &[u8] {
        let start = i * self.num_loci;
        &self.current[start..start + self.num_loci]
    }

    #[cfg(test)]
    fn next_genotype(&self, i: usize) -> &[u8] {
        let start = i * self.num_loci;
        &self.next[start..start + self.num_loci]
    }

    #[cfg(test)]
    fn set_current_genotype(&mut self, i: usize, values: &[u8]) {
        let start = i * self.num_loci;
        self.current[start..start + self.num_loci].copy_from_slice(values);
    }

    fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }
}

/// Double-buffered fitness storage
struct FitnessBuffer {
    current: Vec<f64>,
    next: Vec<f64>,
}

impl FitnessBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            current: vec![0.0; capacity],
            next: vec![0.0; capacity],
        }
    }

    fn current(&self) -> &[f64] {
        &self.current
    }

    fn next_mut(&mut self) -> &mut [f64] {
        &mut self.next
    }

    fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }
}

/// Double-buffered node ID storage
struct NodeBuffer {
    current: Vec<tskit::NodeId>,
    next: Vec<tskit::NodeId>,
}

impl NodeBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            current: vec![tskit::NodeId::NULL; capacity],
            next: vec![tskit::NodeId::NULL; capacity],
        }
    }

    fn current(&self) -> &[tskit::NodeId] {
        &self.current
    }

    fn next_mut(&mut self) -> &mut [tskit::NodeId] {
        &mut self.next
    }

    fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }
}

// ── Population metadata ────────────────────────────────────────────────────────

#[derive(
    serde::Serialize, serde::Deserialize, tskit::metadata::tskit_derive::PopulationMetadata,
)]
#[serializer("serde_json")]
pub struct PopulationMetadata {
    pub deme_id: usize,
    /// Per-generation allele frequencies: allele_freqs[gen][locus_idx]
    pub allele_freqs: Vec<Vec<f64>>,
    /// Per-generation mean fitness
    pub mean_fitness: Vec<f64>,
}

// ── Tracker trait ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DemeRecord {
    /// Allele frequencies per locus (derived allele frequency)
    pub allele_freqs: Vec<f64>,
    /// Mean fitness in this generation
    pub mean_fitness: f64,
}

pub struct GenerationRecord {
    pub demes: Vec<DemeRecord>,
}

/// Trait for recording per-generation statistics.
/// Implementations have access to WrightFisher to compute custom statistics.
pub trait TrackerTrait {
    /// Called after each generation completes.
    /// The tracker can call methods on WrightFisher or access its public fields.
    fn record_generation(
        &mut self,
        wf: &WrightFisher,
        generation: usize,
        record: GenerationRecord,
    ) -> Result<()>;
    /// Called once at the end of the simulation.
    fn finalize(&mut self, wf: &WrightFisher) -> Result<()>;
}

/// Default implementation that just collects all records.
pub struct SimpleTracker {
    pub records: Vec<GenerationRecord>,
}

impl Default for SimpleTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleTracker {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }
}

impl TrackerTrait for SimpleTracker {
    fn record_generation(
        &mut self,
        _wf: &WrightFisher,
        _generation: usize,
        record: GenerationRecord,
    ) -> Result<()> {
        self.records.push(record);
        Ok(())
    }

    fn finalize(&mut self, _wf: &WrightFisher) -> Result<()> {
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
// From https://github.com/tskit-dev/tskit-rust/blob/4200bd97890d309c4e24098321fdb9062fa130b5/examples/haploid_wright_fisher.rs#L33
fn rotate_edges(bookmark: &tskit::types::Bookmark, tables: &mut tskit::TableCollection) {
    let num_edges = tables.edges().num_rows().as_usize();
    let left =
        unsafe { std::slice::from_raw_parts_mut((*tables.as_mut_ptr()).edges.left, num_edges) };
    let right =
        unsafe { std::slice::from_raw_parts_mut((*tables.as_mut_ptr()).edges.right, num_edges) };
    let parent =
        unsafe { std::slice::from_raw_parts_mut((*tables.as_mut_ptr()).edges.parent, num_edges) };
    let child =
        unsafe { std::slice::from_raw_parts_mut((*tables.as_mut_ptr()).edges.child, num_edges) };
    let mid = bookmark.edges().as_usize();
    left.rotate_left(mid);
    right.rotate_left(mid);
    parent.rotate_left(mid);
    child.rotate_left(mid);
}

// ── Simulator ─────────────────────────────────────────────────────────────────
pub struct WrightFisher {
    pub params: Parameters,
    pub tables: tskit::TableCollection,
    pub rng: SmallRng,
    pub birth_time: i64,
    pub bookmark: tskit::types::Bookmark,

    // Genetic architecture
    pub architecture: GeneticArchitecture,

    pub deme_configs: Vec<DemeConfig>,

    // Double-buffered storage (one buffer per deme)
    genotype_buffers: Vec<GenotypeBuffer>,
    fitness_buffers: Vec<FitnessBuffer>,
    node_buffers: Vec<NodeBuffer>,

    // Reusable working memory
    breakpoints_buffer: Vec<f64>,  // Reused for recombination
    deme_records: Vec<DemeRecord>, // Updated in-place each generation
    parent_scratch: Vec<u8>,       // 2 * num_loci (pa then pb)
    offspring_scratch: Vec<u8>,    // num_loci

    // Cached Poisson distributions
    migration_poisson: Vec<Option<Poisson<f64>>>,
    rec_poisson: Option<Poisson<f64>>,
    mut_poisson: Option<Poisson<f64>>, // Poisson(mutation_rate × num_loci)
}

impl WrightFisher {
    // ── Construction ──────────────────────────────────────────────────────────

    pub fn initialize(
        params: Parameters,
        architecture: GeneticArchitecture,
        mut deme_configs: Vec<DemeConfig>,
    ) -> Result<Self> {
        // Validate deme count matches architecture
        if deme_configs.len() != architecture.num_demes {
            return Err(anyhow!(
                "Architecture expects {} demes, got {}",
                architecture.num_demes,
                deme_configs.len()
            ));
        }

        let rng = SmallRng::seed_from_u64(params.random_seed);
        let mut tables = tskit::TableCollection::new(params.sequence_length)?;
        let birth_time = params.runtime as i64 - 1;

        // Initialize populations
        for dc in deme_configs.iter_mut() {
            dc.population_id = tables.add_population()?;
        }

        // Initialize generation 0 with double buffers
        let num_loci = architecture.num_loci();
        let mut genotype_buffers = Vec::new();
        let mut fitness_buffers = Vec::new();
        let mut node_buffers = Vec::new();

        for dc in &deme_configs {
            let k = dc.carrying_capacity;

            // Create buffers
            genotype_buffers.push(GenotypeBuffer::new(k, num_loci));
            let mut fitness_buffer = FitnessBuffer::new(k);

            // Initialize nodes for generation 0
            let mut node_buffer = NodeBuffer::new(k);
            for i in 0..k {
                let node = tables.add_node(0, params.runtime as f64, dc.population_id, -1)?;
                node_buffer.current[i] = node;
                // Initialize fitness to 1.0 (neutral) for generation 0
                fitness_buffer.current[i] = 1.0;
            }
            node_buffers.push(node_buffer);
            fitness_buffers.push(fitness_buffer);
        }

        // Pre-allocate working memory
        let expected_breakpoints =
            (params.recombination_rate * params.sequence_length).ceil() as usize;
        let breakpoints_buffer = Vec::with_capacity((expected_breakpoints * 10).max(10));
        let deme_records = vec![
            DemeRecord {
                allele_freqs: vec![0.0; num_loci],
                mean_fitness: 0.0,
            };
            deme_configs.len()
        ];

        // Cache Poisson distributions
        let rec_poisson = if params.recombination_rate > 0.0 {
            Some(Poisson::new(
                params.recombination_rate * params.sequence_length,
            )?)
        } else {
            None
        };

        let mut_poisson = if params.mutation_rate > 0.0 && num_loci > 0 {
            Some(Poisson::new(params.mutation_rate * num_loci as f64)?)
        } else {
            None
        };

        let total_migration_rates = deme_configs
            .iter()
            .map(|d| d.carrying_capacity as f64 * d.migration_rate)
            .collect::<Vec<f64>>();
        let migration_poisson = total_migration_rates
            .iter()
            .map(|&rate| {
                if rate > 0.0 {
                    Some(Poisson::new(rate).expect("Invalid migration rate"))
                } else {
                    None
                }
            })
            .collect();

        let parent_scratch = vec![0u8; 2 * num_loci];
        let offspring_scratch = vec![0u8; num_loci];

        Ok(Self {
            params,
            tables,
            rng,
            birth_time,
            bookmark: tskit::types::Bookmark::default(),
            architecture,
            deme_configs,
            genotype_buffers,
            fitness_buffers,
            node_buffers,
            breakpoints_buffer,
            deme_records,
            parent_scratch,
            offspring_scratch,
            rec_poisson,
            mut_poisson,
            migration_poisson,
        })
    }

    // ── Fitness ───────────────────────────────────────────────────────────────

    /// Compute additive fitness for an individual in a deme
    /// Fitness = exo(\sum s_i) for all loci with derived allele (1)
    pub fn parental_fitness(&self, d: usize, i: usize) -> f64 {
        let genotype = self.genotype_buffers[d].current_genotype(i);
        let num_loci = genotype.len();
        let mut phenotype = 0.0;

        // SAFETY: l is in 0..num_loci, genotype slice has exactly num_loci elements
        unsafe {
            for l in 0..num_loci {
                if *genotype.get_unchecked(l) == 1 {
                    phenotype += self.architecture.selection_coeff(l, d);
                }
            }
        }

        phenotype.exp()
    }

    // ── Core reproductive step ────────────────────────────────────────────────

    /// Create offspring directly in pre-allocated buffers
    fn create_offspring_in_place(
        &mut self,
        source_deme: usize,
        pa_idx: usize,
        pb_idx: usize,
        birth_time: f64,
        dest_deme: usize,
        offspring_idx: usize,
    ) -> Result<()> {
        let num_loci = self.architecture.num_loci();
        let seq_len = self.params.sequence_length;

        // Get parent node IDs (Copy types, borrow released immediately)
        let pa_node = self.node_buffers[source_deme].current()[pa_idx];
        let pb_node = self.node_buffers[source_deme].current()[pb_idx];

        // Copy-in: copy parent genotypes into scratch buffers
        let pa_off = pa_idx * num_loci;
        let pb_off = pb_idx * num_loci;
        debug_assert!(pa_off + num_loci <= self.genotype_buffers[source_deme].current.len());
        debug_assert!(pb_off + num_loci <= self.genotype_buffers[source_deme].current.len());
        debug_assert!(2 * num_loci <= self.parent_scratch.len());
        // SAFETY: offsets validated by debug_assert; buffers allocated to K * num_loci
        // in initialize(), and pa_idx/pb_idx < K.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.genotype_buffers[source_deme]
                    .current
                    .as_ptr()
                    .add(pa_off),
                self.parent_scratch.as_mut_ptr(),
                num_loci,
            );
            std::ptr::copy_nonoverlapping(
                self.genotype_buffers[source_deme]
                    .current
                    .as_ptr()
                    .add(pb_off),
                self.parent_scratch.as_mut_ptr().add(num_loci),
                num_loci,
            );
        }

        // Sample recombination breakpoints (reuse buffer)
        self.breakpoints_buffer.clear();
        if let Some(dist) = &self.rec_poisson {
            let num_bp = dist.sample(&mut self.rng) as usize;
            for _ in 0..num_bp {
                self.breakpoints_buffer
                    .push(self.rng.random_range(0.0..seq_len));
            }
            self.breakpoints_buffer
                .sort_by(|a, b| a.partial_cmp(b).unwrap());
        }

        // Single-pass merge recombination into offspring_scratch
        // Both positions and breakpoints are sorted, so we advance through both in one pass.
        // SAFETY: all indices l are in 0..num_loci; parent_scratch has 2*num_loci elements;
        // offspring_scratch, positions have num_loci elements; bp_i < num_bp checked in while.
        let positions = self.architecture.positions();
        let num_bp = self.breakpoints_buffer.len();
        let mut bp_i = 0;
        let mut current_parent = 0usize; // 0 = pa (offset 0), 1 = pb (offset num_loci)

        unsafe {
            for l in 0..num_loci {
                let pos = *positions.get_unchecked(l);
                while bp_i < num_bp && *self.breakpoints_buffer.get_unchecked(bp_i) <= pos {
                    current_parent = 1 - current_parent;
                    bp_i += 1;
                }
                let parent_off = current_parent * num_loci;
                *self.offspring_scratch.get_unchecked_mut(l) =
                    *self.parent_scratch.get_unchecked(parent_off + l);
            }
        }

        // Apply mutations (in-place in offspring scratch)
        if let Some(dist) = &self.mut_poisson {
            let num_muts = dist.sample(&mut self.rng) as usize;
            for _ in 0..num_muts {
                let l = self.rng.random_range(0..num_loci);
                // SAFETY: l is sampled from 0..num_loci, offspring_scratch has num_loci elements
                unsafe {
                    *self.offspring_scratch.get_unchecked_mut(l) ^= 1;
                }
            }
        }

        // Compute fitness from offspring scratch (in destination deme context)
        // SAFETY: l in 0..num_loci, offspring_scratch has num_loci elements
        let mut phenotype = 0.0;
        unsafe {
            for l in 0..num_loci {
                if *self.offspring_scratch.get_unchecked(l) == 1 {
                    phenotype += self.architecture.selection_coeff(l, dest_deme);
                }
            }
        }
        let fitness = phenotype.exp();

        // Copy-out: write offspring genotype back to next buffer
        let out_off = offspring_idx * num_loci;
        debug_assert!(out_off + num_loci <= self.genotype_buffers[dest_deme].next.len());
        // SAFETY: offspring_idx < K, next buffer has K * num_loci elements
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.offspring_scratch.as_ptr(),
                self.genotype_buffers[dest_deme]
                    .next
                    .as_mut_ptr()
                    .add(out_off),
                num_loci,
            );
        }

        // Add child node to tskit
        let dest_pop_id = self.deme_configs[dest_deme].population_id;
        let child_node = self.tables.add_node(0, birth_time, dest_pop_id, -1)?;

        // Add edges
        let mut cur_node = pa_node;
        let mut other_node = pb_node;
        let mut start = 0.0;

        for &bp in &self.breakpoints_buffer {
            self.tables.add_edge(start, bp, cur_node, child_node)?;
            std::mem::swap(&mut cur_node, &mut other_node);
            start = bp;
        }
        self.tables.add_edge(start, seq_len, cur_node, child_node)?;

        // Write fitness and node to output buffers
        self.node_buffers[dest_deme].next_mut()[offspring_idx] = child_node;
        self.fitness_buffers[dest_deme].next_mut()[offspring_idx] = fitness;

        Ok(())
    }

    // ── Generation loop ───────────────────────────────────────────────────────

    pub fn step(&mut self, generation: usize, tracker: &mut dyn TrackerTrait) -> Result<()> {
        let birth_time = self.birth_time as f64;
        let num_demes = self.deme_configs.len();

        // ── Phase 1: Record statistics from current generation ───────────────
        for d in 0..num_demes {
            let k = self.deme_configs[d].carrying_capacity;
            let fitnesses = self.fitness_buffers[d].current();
            let num_loci = self.architecture.num_loci();

            // Mean fitness
            let mean_fitness = fitnesses.iter().sum::<f64>() / k as f64;

            // Allele frequencies (update in-place, flat buffer access)
            // SAFETY: i < k, l < num_loci, buffer has k * num_loci elements;
            // allele_freqs has num_loci elements.
            let geno_buf = &self.genotype_buffers[d].current;
            debug_assert!(k * num_loci <= geno_buf.len());
            for l in 0..num_loci {
                let mut count = 0usize;
                for i in 0..k {
                    if *geno_buf.get(i * num_loci + l).expect("invalid position") == 1 {
                        count += 1;
                    }
                }
                *self.deme_records[d]
                    .allele_freqs
                    .get_mut(l)
                    .expect("invalid position") = count as f64 / k as f64;
            }
            self.deme_records[d].mean_fitness = mean_fitness;
        }

        tracker.record_generation(
            self,
            generation,
            GenerationRecord {
                demes: self.deme_records.clone(), // Clone is cheap, just Vec<f64>
            },
        )?;

        // ── Phase 2: Generate next generation (one deme at a time) ───────────
        let fitnesses_weights: Vec<WeightedIndex<f64>> = self
            .fitness_buffers
            .iter()
            .map(|buffer| WeightedIndex::new(buffer.current()).expect("Invalid fitness"))
            .collect();
        for dest_deme in 0..num_demes {
            // First, we take inmigrants from all other populations
            let k = self.deme_configs[dest_deme].carrying_capacity;
            let mut n_local = k;
            let mut offspring_idx = 0;
            // Part A: Add migrants
            for source_deme in 0..num_demes {
                if source_deme == dest_deme {
                    continue;
                }
                let n_immigrants = match self.migration_poisson.get(source_deme).unwrap_or(&None) {
                    Some(mig_poisson) => mig_poisson.sample(&mut self.rng) as usize,
                    None => 0,
                };
                //eprintln!("Adding {n_immigrants} migrants from deme {source_deme}");
                n_local -= n_immigrants;
                let source_parent_dist = fitnesses_weights.get(source_deme).expect("Invalid index");
                for _ in 0..n_immigrants {
                    debug_assert!(n_immigrants > 0);
                    let pa = source_parent_dist.sample(&mut self.rng);
                    let pb = source_parent_dist.sample(&mut self.rng);

                    // Create offspring (born into dest_deme)
                    self.create_offspring_in_place(
                        source_deme,
                        pa,
                        pb,
                        birth_time,
                        dest_deme,
                        offspring_idx,
                    )?;

                    offspring_idx += 1;
                }
            }

            // Part B: Add N-M local offspring
            let local_parent_dist = fitnesses_weights.get(dest_deme).expect("Invalid index");
            for _ in 0..n_local {
                let pa = local_parent_dist.sample(&mut self.rng);
                let pb = local_parent_dist.sample(&mut self.rng);

                self.create_offspring_in_place(
                    dest_deme,
                    pa,
                    pb,
                    birth_time,
                    dest_deme,
                    offspring_idx,
                )?;

                offspring_idx += 1;
            }

            assert_eq!(offspring_idx, k, "Should fill exactly K offspring");
        }

        // ── Phase 3: Swap all buffers ─────────────────────────────────────────
        for d in 0..num_demes {
            self.genotype_buffers[d].swap();
            self.node_buffers[d].swap();
            self.fitness_buffers[d].swap();
        }

        // ── Phase 4: Periodic simplification ──────────────────────────────────
        if generation > 0 && generation.is_multiple_of(self.params.simplify_interval) {
            self.simplify()?;
        }

        self.birth_time -= 1;
        Ok(())
    }

    pub fn simplify(&mut self) -> Result<()> {
        // Collect all current nodes
        let mut samples: Vec<tskit::NodeId> = Vec::new();
        for node_buffer in &self.node_buffers {
            samples.extend_from_slice(node_buffer.current());
        }

        self.tables
            .sort(&self.bookmark, tskit::TableSortOptions::default())?;
        rotate_edges(&self.bookmark, &mut self.tables);

        if let Some(idmap) = self.tables.simplify(
            &samples,
            tskit::SimplificationOptions::KEEP_INPUT_ROOTS,
            true,
        )? {
            // Remap node IDs in current buffers
            for node_buffer in &mut self.node_buffers {
                for node in &mut node_buffer.current {
                    *node = idmap[usize::try_from(*node)?];
                }
            }
        }

        self.bookmark.set_edges(self.tables.edges().num_rows());
        Ok(())
    }

    pub fn run(&mut self, tracker: &mut dyn TrackerTrait) -> Result<()> {
        let bar = ProgressBar::new(self.params.runtime as u64);
        for g in 0..self.params.runtime {
            bar.inc(1);
            self.step(g, tracker)?;
        }
        bar.finish();
        Ok(())
    }

    // ── Finalization ──────────────────────────────────────────────────────────

    pub fn finalize(mut self, tracker: &mut dyn TrackerTrait) -> Result<tskit::TreeSequence> {
        tracker.finalize(&self)?;
        let mut all_nodes: Vec<tskit::NodeId> = Vec::new();
        for node_buffer in &self.node_buffers {
            all_nodes.extend_from_slice(node_buffer.current());
        }
        self.tables
            .sort(&self.bookmark, tskit::TableSortOptions::default())?;
        rotate_edges(&self.bookmark, &mut self.tables);
        self.tables.simplify(
            &all_nodes,
            tskit::SimplificationOptions::KEEP_INPUT_ROOTS,
            false,
        )?;
        self.tables.build_index()?;
        let tree_sequence = self
            .tables
            .tree_sequence(tskit::TreeSequenceFlags::default())?;
        Ok(tree_sequence)
    }

    /// Finalize with metadata from generation records.
    pub fn finalize_with_metadata(
        mut self,
        records: &[GenerationRecord],
    ) -> Result<tskit::TreeSequence> {
        let mut all_nodes: Vec<tskit::NodeId> = Vec::new();
        for node_buffer in &self.node_buffers {
            all_nodes.extend_from_slice(node_buffer.current());
        }
        self.tables
            .sort(&self.bookmark, tskit::TableSortOptions::default())?;
        rotate_edges(&self.bookmark, &mut self.tables);
        self.tables.simplify(
            &all_nodes,
            tskit::SimplificationOptions::KEEP_INPUT_ROOTS,
            false,
        )?;
        self.tables.build_index()?;

        // Create population metadata from records
        let mut populations = tskit::PopulationTable::default();
        let num_demes = self.deme_configs.len();
        for d in 0..num_demes {
            // Extract per-deme data from records
            let allele_freqs: Vec<Vec<f64>> = records
                .iter()
                .map(|gr| gr.demes[d].allele_freqs.clone())
                .collect();

            let mean_fitness: Vec<f64> =
                records.iter().map(|gr| gr.demes[d].mean_fitness).collect();

            let meta = PopulationMetadata {
                deme_id: d,
                allele_freqs,
                mean_fitness,
            };
            let _ = populations.add_row_with_metadata(&meta)?;
        }

        self.tables.set_populations(&populations)?;
        let tree_sequence = self
            .tables
            .tree_sequence(tskit::TreeSequenceFlags::default())?;
        Ok(tree_sequence)
    }
    pub fn get_mut_genotypes(&mut self, i_deme: usize) -> ChunksExactMut<'_, u8> {
        let long_genotypes = &mut self.genotype_buffers[i_deme].current;
        let chunk_size = self.deme_configs[i_deme].carrying_capacity;
        assert!(long_genotypes.len().is_multiple_of(chunk_size));
        long_genotypes.chunks_exact_mut(chunk_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genotype_recombination() {
        // Create simple architecture with 3 loci
        let loci = vec![
            Locus {
                position: 1e6,
                selection_coeffs: vec![0.0],
            },
            Locus {
                position: 5e6,
                selection_coeffs: vec![0.0],
            },
            Locus {
                position: 9e6,
                selection_coeffs: vec![0.0],
            },
        ];
        let arch = GeneticArchitecture::new(loci, 1).unwrap();

        let params = Parameters::default();
        let deme_configs = vec![DemeConfig {
            carrying_capacity: 2,
            migration_rate: 0.0,
            population_id: tskit::PopulationId::NULL,
        }];

        let mut sim = WrightFisher::initialize(params, arch, deme_configs).unwrap();

        // Set parent genotypes manually
        sim.genotype_buffers[0].set_current_genotype(0, &[1, 1, 1]); // parent A: all derived
        sim.genotype_buffers[0].set_current_genotype(1, &[0, 0, 0]); // parent B: all ancestral

        // Sample offspring (no mutations)
        sim.mut_poisson = None;
        sim.create_offspring_in_place(0, 0, 1, 0.0, 0, 0).unwrap();

        let offspring_genotype = sim.genotype_buffers[0].next_genotype(0);
        assert_eq!(offspring_genotype.len(), 3);
        // Each locus should be either 0 or 1 (no invalid values)
        assert!(offspring_genotype.iter().all(|&a| a == 0 || a == 1));
    }

    #[test]
    fn test_fitness_calculation() {
        let loci = vec![
            Locus {
                position: 1e6,
                selection_coeffs: vec![0.0, 0.1],
            },
            Locus {
                position: 5e6,
                selection_coeffs: vec![-0.1, -0.1],
            },
        ];
        let arch = GeneticArchitecture::new(loci, 2).unwrap();

        let params = Parameters::default();
        let deme_configs = vec![
            DemeConfig {
                carrying_capacity: 1,
                migration_rate: 0.0,
                population_id: tskit::PopulationId::NULL,
            },
            DemeConfig {
                carrying_capacity: 1,
                migration_rate: 0.0,
                population_id: tskit::PopulationId::NULL,
            },
        ];

        let mut sim = WrightFisher::initialize(params, arch, deme_configs).unwrap();

        // Test genotype: [[1, 0], [1, 0]]
        sim.genotype_buffers[0].set_current_genotype(0, &[1, 0]);
        sim.genotype_buffers[1].set_current_genotype(0, &[1, 0]);
        assert!((sim.parental_fitness(0, 0) - 1.0).abs() < 1e-6);
        assert!((sim.parental_fitness(1, 0).ln() - 0.1).abs() < 1e-6);
        // Test genotype: [[1, 1], [1, 1]] = both derived
        sim.genotype_buffers[0].set_current_genotype(0, &[1, 1]);
        sim.genotype_buffers[1].set_current_genotype(0, &[1, 1]);
        //
        assert!((sim.parental_fitness(0, 0).ln() - -0.1).abs() < 1e-6);
        assert!((sim.parental_fitness(1, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_architecture_validation() {
        // Should fail: mismatched selection coeff counts
        let loci = vec![
            Locus {
                position: 1e6,
                selection_coeffs: vec![0.1],
            },
            Locus {
                position: 5e6,
                selection_coeffs: vec![0.1, 0.2],
            }, // wrong!
        ];
        assert!(GeneticArchitecture::new(loci, 1).is_err());

        // Should fail: unsorted positions
        let loci = vec![
            Locus {
                position: 5e6,
                selection_coeffs: vec![0.1],
            },
            Locus {
                position: 1e6,
                selection_coeffs: vec![0.1],
            }, // out of order!
        ];
        assert!(GeneticArchitecture::new(loci, 1).is_err());
    }
}
