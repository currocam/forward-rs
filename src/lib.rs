pub mod inversion;

use anyhow::Result;
use indicatif::ProgressBar;
use rand::SeedableRng;
use rand::distr::weighted::WeightedIndex;
use rand::distr::{Distribution, Uniform};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand_distr::Poisson;
use std::collections::{HashMap, HashSet};

// ── Distribution of effect sizes ─────────────────────────────────────────────
#[derive(Clone, Debug)]
pub enum MutationEffectDist {
    /// Default. Each new mutation picks one trait at random and draws
    /// an effect from Uniform(-scale, scale); all other traits get 0.
    UniformSingle { scale: f64 },
    /// Original behavior. Each trait gets an independently drawn effect
    /// from Uniform(-scale, scale).
    UniformAll { scale: f64 },
}

impl MutationEffectDist {
    pub fn sample(&self, rng: &mut SmallRng, num_traits: usize) -> Vec<f64> {
        match self {
            Self::UniformSingle { scale } => {
                let t = rng.random_range(0..num_traits);
                let eff = Uniform::new(-scale, *scale).unwrap().sample(rng);
                let mut v = vec![0.0; num_traits];
                v[t] = eff;
                v
            }
            Self::UniformAll { scale } => {
                let dist = Uniform::new(-scale, *scale).unwrap();
                (0..num_traits).map(|_| dist.sample(rng)).collect()
            }
        }
    }
}

// ── Assortative mating ────────────────────────────────────────────────────────

/// Gaussian assortative mating: mate 2 is drawn proportional to
/// exp(-(z1 - z2)^2 / (2 σ^2)) where z is the phenotype for `trait_index`.
/// Only affects within-deme mate choice during offspring pool generation.
#[derive(Clone, Debug)]
pub struct AssortativeMating {
    /// Index into `WrightFisher::traits` used for mate choice.
    pub trait_index: usize,
    /// Standard deviation of the Gaussian mating kernel.
    /// Large σ → nearly random; small σ → highly assortative.
    pub sigma: f64,
}

// ── Parameters ────────────────────────────────────────────────────────────────
#[derive(Clone, Debug)]
pub struct Parameters {
    pub random_seed: u64,
    pub sequence_length: f64,
    pub runtime: usize,
    pub recombination_rate: f64,
    pub mutation_rate: f64,
    /// Simplify every this many generations (0 = only at finalize).
    pub simplify_interval: usize,
    /// Offspring pool = fecundity × K_d before density regulation.
    pub fecundity: f64,
    pub effect_dist: MutationEffectDist,
}

impl Default for Parameters {
    fn default() -> Self {
        let mut rng = rand::rng();
        let random_seed = rng.random_range(1..u64::MAX);
        Self {
            random_seed,
            sequence_length: 1e7,
            runtime: 5_000,
            recombination_rate: 1e-6,
            mutation_rate: 0.0,
            simplify_interval: 100,
            fecundity: 10.0,
            effect_dist: MutationEffectDist::UniformSingle { scale: 1.0 },
        }
    }
}

// ── Deme configuration ────────────────────────────────────────────────────────
#[derive(Clone, Debug)]
pub struct DemeConfig {
    pub carrying_capacity: usize,
    /// Fraction of K_d that emigrate each generation.
    pub migration_rate: f64,
    /// Length == num_traits.  Deme-specific phenotypic targets.
    pub trait_optima: Vec<f64>,
    /// Set during `initialize`; used by `commit_to_tables`.
    pub population_id: tskit::PopulationId,
}

// ── Polygenic components ──────────────────────────────────────────────────────

/// Per-individual mutation set kept **sorted by genomic position**.
/// Each entry is `(position, global_mutation_index)`.
pub type IndMuts = Vec<(f64, usize)>;

/// Global registry of every mutation ever created.
#[derive(Clone, Debug)]
pub struct MutationRegistry {
    pub positions: Vec<f64>,
    /// `effects[mutation_idx][trait_idx]`
    pub effects: Vec<Vec<f64>>,
    pub num_traits: usize,
}

impl MutationRegistry {
    pub fn new(num_traits: usize) -> Self {
        Self {
            positions: Vec::new(),
            effects: Vec::new(),
            num_traits,
        }
    }

    pub fn push(&mut self, pos: f64, effects: Vec<f64>) -> usize {
        debug_assert_eq!(effects.len(), self.num_traits);
        let idx = self.positions.len();
        self.positions.push(pos);
        self.effects.push(effects);
        idx
    }
}

/// A single quantitative trait under Gaussian stabilizing selection.
///
#[derive(Clone, Debug)]
pub struct GaussianSelectionTrait {
    /// Accumulated effect of all fixed mutations.
    pub baseline: f64,
    /// Strength of selection 1 / (2 * sigma^2)
    pub selection_coeff: f64,
}

impl GaussianSelectionTrait {
    pub fn new(selection_coeff: f64) -> Self {
        Self {
            baseline: 0.0,
            selection_coeff,
        }
    }

    pub fn fitness_contribution(&self, phenotype: f64, optimum: f64) -> f64 {
        (-self.selection_coeff * (phenotype - optimum).powi(2)).exp()
    }
}

// ── Population metadata ────────────────────────────────────────────────────────

#[derive(
    serde::Serialize, serde::Deserialize, tskit::metadata::tskit_derive::PopulationMetadata,
)]
#[serializer("serde_json")]
pub struct PopulationMetadata {
    pub deme_id: usize,
    /// Per-generation mean phenotypes (offspring pool) for each trait
    pub mean_phenotypes_offspring: Vec<Vec<f64>>,
    /// Per-generation mean fitness of offspring pool
    pub mean_fitness_offspring: Vec<f64>,
    /// Per-generation mean phenotypes (survivors) for each trait
    pub mean_phenotypes_survivors: Vec<Vec<f64>>,
    /// Per-generation number of segregating mutations
    pub num_segregating_muts: Vec<usize>,
    /// Per-generation flat row-major covariance of offspring pool: `[generation][T²]`
    pub cov_phenotypes_offspring: Vec<Vec<f64>>,
    /// Per-generation flat row-major covariance of survivors: `[generation][T²]`
    pub cov_phenotypes_survivors: Vec<Vec<f64>>,
}

// ── Tracker trait ─────────────────────────────────────────────────────────────

pub struct DemeRecord {
    /// Mean phenotype per trait across the pre-selection offspring pool.
    pub mean_phenotypes_offspring: Vec<f64>,
    /// Mean fitness across the pre-selection offspring pool.
    pub mean_fitness_offspring: f64,
    /// Mean phenotype per trait after density regulation (before migration).
    pub mean_phenotypes_survivors: Vec<f64>,
    /// Number of segregating mutations in this deme.
    pub num_segregating_muts: usize,
    /// Flat row-major covariance matrix of the offspring pool (T² elements).
    pub cov_phenotypes_offspring: Vec<f64>,
    /// Flat row-major covariance matrix of the K survivors (T² elements).
    pub cov_phenotypes_survivors: Vec<f64>,
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

// ── Offspring candidate ───────────────────────────────────────────────────────

#[derive(Clone)]
pub struct OffspringCandidate {
    /// Which deme this offspring was born into.
    pub birth_deme: usize,
    /// Local index into `deme_parents[birth_deme]` / `deme_muts[birth_deme]`.
    pub pa: usize,
    pub pb: usize,
    pub breakpoints: Vec<f64>,
    pub inherited_muts: IndMuts,
    /// New mutations NOT yet in the registry: `(position, per-trait effects)`.
    pub new_mut_data: Vec<(f64, Vec<f64>)>,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn rotate_edges(bookmark: &tskit::types::Bookmark, tables: &mut tskit::TableCollection) {
    let num_edges = tables.edges().num_rows().as_usize();
    let mid = bookmark.edges().as_usize();
    if mid == 0 || mid == num_edges {
        return;
    }
    unsafe {
        let p = (*tables.as_mut_ptr()).edges;
        std::slice::from_raw_parts_mut(p.left, num_edges).rotate_left(mid);
        std::slice::from_raw_parts_mut(p.right, num_edges).rotate_left(mid);
        std::slice::from_raw_parts_mut(p.parent, num_edges).rotate_left(mid);
        std::slice::from_raw_parts_mut(p.child, num_edges).rotate_left(mid);
    }
}

/// MLE estimates for a multivariate normal from a slice of data points.
/// Returns `(mean, cov_flat)` where `cov_flat` is the T×T covariance
/// stored row-major (population covariance, divide by n).
/// Returns empty vecs if data is empty or T == 0.
fn mle_mvn(data: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    let t = data[0].len();
    if t == 0 {
        return (vec![], vec![]);
    }

    // Mean
    let mut mean = vec![0.0_f64; t];
    for x in data {
        for (i, &v) in x.iter().enumerate() {
            mean[i] += v;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }

    // Covariance (population, 1/n)
    let mut cov = vec![0.0_f64; t * t];
    for x in data {
        for i in 0..t {
            for j in 0..t {
                cov[i * t + j] += (x[i] - mean[i]) * (x[j] - mean[j]);
            }
        }
    }
    for c in &mut cov {
        *c /= n as f64;
    }

    (mean, cov)
}

// ── Simulator ─────────────────────────────────────────────────────────────────
//#[derive(Clone)]
pub struct WrightFisher {
    pub params: Parameters,
    pub tables: tskit::TableCollection,
    pub rng: SmallRng,
    pub birth_time: i64,
    pub bookmark: tskit::types::Bookmark,

    pub registry: MutationRegistry,
    /// Shared selection trait definitions (no optimum; optima live in DemeConfig).
    pub traits: Vec<GaussianSelectionTrait>,
    pub deme_configs: Vec<DemeConfig>,
    /// `deme_parents[d][i]` = NodeId of individual i in deme d.
    pub deme_parents: Vec<Vec<tskit::NodeId>>,
    /// `deme_muts[d][i]` = segregating mutations of individual i in deme d.
    pub deme_muts: Vec<Vec<IndMuts>>,

    /// Optional assortative mating rule applied during offspring pool generation.
    pub assortative_mating: Option<AssortativeMating>,

    // Cached Poisson distributions (λ = rate × seq_len).
    rec_poisson: Option<Poisson<f64>>,
    mut_poisson: Option<Poisson<f64>>,
}

impl WrightFisher {
    // ── Construction ──────────────────────────────────────────────────────────

    pub fn initialize(
        params: Parameters,
        traits: Vec<GaussianSelectionTrait>,
        mut deme_configs: Vec<DemeConfig>,
    ) -> Result<Self> {
        let mut tables = tskit::TableCollection::new(params.sequence_length)?;

        // Register one tskit population per deme.
        for dc in deme_configs.iter_mut() {
            dc.population_id = tables.add_population()?;
        }

        let parental_time = params.runtime as f64;
        let num_traits = traits.len();

        let mut deme_parents: Vec<Vec<tskit::NodeId>> = Vec::with_capacity(deme_configs.len());
        let mut deme_muts: Vec<Vec<IndMuts>> = Vec::with_capacity(deme_configs.len());

        for dc in &deme_configs {
            let k = dc.carrying_capacity;
            let pop_id = dc.population_id;
            let parents = (0..k)
                .map(|_| tables.add_node(0, parental_time, pop_id, -1))
                .collect::<Result<Vec<_>, _>>()?;
            deme_parents.push(parents);
            deme_muts.push(vec![IndMuts::new(); k]);
        }

        let rng = SmallRng::seed_from_u64(params.random_seed);
        let birth_time = params.runtime as i64 - 1;

        let rec_poisson = if params.recombination_rate * params.sequence_length > 0.0 {
            Some(Poisson::new(
                params.recombination_rate * params.sequence_length,
            )?)
        } else {
            None
        };
        let mut_poisson =
            if !traits.is_empty() && params.mutation_rate * params.sequence_length > 0.0 {
                Some(Poisson::new(params.mutation_rate * params.sequence_length)?)
            } else {
                None
            };

        Ok(Self {
            registry: MutationRegistry::new(num_traits),
            traits,
            deme_configs,
            deme_parents,
            deme_muts,
            params,
            tables,
            rng,
            birth_time,
            bookmark: tskit::types::Bookmark::default(),
            assortative_mating: None,
            rec_poisson,
            mut_poisson,
        })
    }

    // ── Phenotype ─────────────────────────────────────────────────────────────

    pub fn candidate_phenotype(&self, c: &OffspringCandidate, t: usize) -> f64 {
        let baseline = self.traits[t].baseline;
        let inherited: f64 = c
            .inherited_muts
            .iter()
            .map(|&(_, idx)| self.registry.effects[idx][t])
            .sum();
        let new_muts: f64 = c.new_mut_data.iter().map(|(_, eff)| eff[t]).sum();
        baseline + inherited + new_muts
    }

    /// Phenotype of the current (parental) individual `i` in deme `d` for trait `t`.
    pub fn parent_phenotype(&self, d: usize, i: usize, t: usize) -> f64 {
        let baseline = self.traits[t].baseline;
        let inherited: f64 = self.deme_muts[d][i]
            .iter()
            .map(|&(_, idx)| self.registry.effects[idx][t])
            .sum();
        baseline + inherited
    }

    // ── Fitness ───────────────────────────────────────────────────────────────

    pub fn candidate_fitness_in_deme(&self, c: &OffspringCandidate, d: usize) -> f64 {
        (0..self.traits.len())
            .map(|t| {
                let z = self.candidate_phenotype(c, t);
                let opt = self.deme_configs[d].trait_optima[t];
                self.traits[t].fitness_contribution(z, opt)
            })
            .product()
    }

    // ── Fixed-mutation removal ────────────────────────────────────────────────

    pub fn remove_fixed_mutations(&mut self) {
        if self.traits.is_empty() {
            return;
        }
        let total_n: usize = self
            .deme_configs
            .iter()
            .map(|dc| dc.carrying_capacity)
            .sum();
        if total_n == 0 {
            return;
        }

        let mut counts: HashMap<usize, usize> = HashMap::new();
        for deme in &self.deme_muts {
            for ind in deme {
                for &(_, idx) in ind {
                    *counts.entry(idx).or_insert(0) += 1;
                }
            }
        }

        let fixed: HashSet<usize> = counts
            .into_iter()
            .filter(|&(_, count)| count == total_n)
            .map(|(idx, _)| idx)
            .collect();

        if fixed.is_empty() {
            return;
        }

        for &idx in &fixed {
            for (t, tr) in self.traits.iter_mut().enumerate() {
                tr.baseline += self.registry.effects[idx][t];
            }
        }

        for deme in &mut self.deme_muts {
            for ind in deme.iter_mut() {
                ind.retain(|&(_, idx)| !fixed.contains(&idx));
            }
        }
    }

    // ── Core reproductive step ────────────────────────────────────────────────

    pub fn sample_candidate(
        &mut self,
        d: usize,
        pa: usize,
        pb: usize,
    ) -> Result<OffspringCandidate> {
        let seq_len = self.params.sequence_length;

        // ── Recombination breakpoints ─────────────────────────────────────────
        let num_bp = match &self.rec_poisson {
            Some(dist) => dist.sample(&mut self.rng) as usize,
            None => 0,
        };
        let breakpoints = if num_bp > 0 && seq_len >= 2.0 {
            let bp_dist = Uniform::new(1u64, seq_len as u64)?;
            let mut bps_u64: Vec<u64> =
                (0..num_bp).map(|_| bp_dist.sample(&mut self.rng)).collect();
            bps_u64.sort_unstable();
            bps_u64.dedup();
            bps_u64.iter().map(|&x| x as f64).collect()
        } else {
            vec![]
        };

        // ── Mutation inheritance ──────────────────────────────────────────────
        let mut inherited_muts: IndMuts = Vec::new();
        let mut cur_idx = pa;
        let mut other_idx = pb;
        let mut start = 0.0f64;

        for &x in &breakpoints {
            let muts = &self.deme_muts[d][cur_idx];
            let lo = muts.partition_point(|&(pos, _)| pos < start);
            let hi = muts.partition_point(|&(pos, _)| pos < x);
            inherited_muts.extend_from_slice(&muts[lo..hi]);
            std::mem::swap(&mut cur_idx, &mut other_idx);
            start = x;
        }
        let muts = &self.deme_muts[d][cur_idx];
        let lo = muts.partition_point(|&(pos, _)| pos < start);
        inherited_muts.extend_from_slice(&muts[lo..]);

        // ── New mutations ─────────────────────────────────────────────────────
        let mut new_mut_data: Vec<(f64, Vec<f64>)> = Vec::new();
        if let Some(dist) = &self.mut_poisson {
            let num_new = dist.sample(&mut self.rng) as usize;
            let pos_dist = Uniform::new(0.0f64, seq_len)?;
            let num_traits = self.registry.num_traits;
            for _ in 0..num_new {
                let pos = pos_dist.sample(&mut self.rng);
                let effects = self.params.effect_dist.sample(&mut self.rng, num_traits);
                new_mut_data.push((pos, effects));
            }
        }

        Ok(OffspringCandidate {
            birth_deme: d,
            pa,
            pb,
            breakpoints,
            inherited_muts,
            new_mut_data,
        })
    }

    pub fn commit_to_tables(
        &mut self,
        birth_time: f64,
        candidate: OffspringCandidate,
    ) -> Result<(tskit::NodeId, IndMuts)> {
        let seq_len = self.params.sequence_length;
        let d = candidate.birth_deme;
        let parent_a = self.deme_parents[d][candidate.pa];
        let parent_b = self.deme_parents[d][candidate.pb];
        let pop_id = self.deme_configs[d].population_id;

        // Register new mutations now that this offspring has survived.
        let mut child_muts = candidate.inherited_muts;
        for (pos, effects) in candidate.new_mut_data {
            let idx = self.registry.push(pos, effects);
            let at = child_muts.partition_point(|&(p, _)| p < pos);
            child_muts.insert(at, (pos, idx));
        }

        // Add child node tagged with birth deme.
        let child = self.tables.add_node(0, birth_time, pop_id, -1)?;

        // Replay edges.
        let mut cur_node = parent_a;
        let mut other_node = parent_b;
        let mut start = 0.0f64;

        for &x in &candidate.breakpoints {
            self.tables.add_edge(start, x, cur_node, child)?;
            std::mem::swap(&mut cur_node, &mut other_node);
            start = x;
        }
        self.tables.add_edge(start, seq_len, cur_node, child)?;

        Ok((child, child_muts))
    }

    // ── Generation loop ───────────────────────────────────────────────────────

    pub fn step(&mut self, generation: usize, tracker: &mut dyn TrackerTrait) -> Result<()> {
        let birth_time = self.birth_time as f64;
        let num_demes = self.deme_configs.len();
        let num_traits = self.traits.len();

        // ── Phase 1: per-deme reproduction + density regulation ───────────────
        // deme_survivors[d] = Vec of survivors (post-selection, pre-migration)
        let mut deme_survivors: Vec<Vec<OffspringCandidate>> = Vec::with_capacity(num_demes);
        let mut gen_deme_records: Vec<DemeRecord> = Vec::with_capacity(num_demes);

        for d in 0..num_demes {
            let k = self.deme_configs[d].carrying_capacity;
            let pool_size = ((self.params.fecundity * k as f64) as usize).max(k);
            let uniform_parent = Uniform::new(0usize, k)?;

            // Precompute parental phenotypes for assortative mating (if active).
            // Store (phenotypes_vec, sigma) so the closure below owns all it needs
            // without re-borrowing `self`.
            let am_data: Option<(Vec<f64>, f64)> =
                self.assortative_mating.as_ref().map(|am| {
                    let zs = (0..k)
                        .map(|i| self.parent_phenotype(d, i, am.trait_index))
                        .collect();
                    (zs, am.sigma)
                });

            // Produce offspring pool.
            let candidates: Vec<OffspringCandidate> = (0..pool_size)
                .map(|_| {
                    let pa = uniform_parent.sample(&mut self.rng);
                    let pb = if let Some((ref zs, sigma)) = am_data {
                        let z_pa = zs[pa];
                        let weights: Vec<f64> = zs
                            .iter()
                            .map(|&z| (-(z_pa - z).powi(2) / (2.0 * sigma * sigma)).exp())
                            .collect();
                        WeightedIndex::new(&weights)?.sample(&mut self.rng)
                    } else {
                        uniform_parent.sample(&mut self.rng)
                    };
                    self.sample_candidate(d, pa, pb)
                })
                .collect::<Result<_>>()?;

            // Compute fitnesses.
            let fitnesses: Vec<f64> = if num_traits == 0 {
                vec![1.0f64; pool_size]
            } else {
                candidates
                    .iter()
                    .map(|c| self.candidate_fitness_in_deme(c, d))
                    .collect()
            };

            // Record pool statistics.
            let mean_fitness_offspring = fitnesses.iter().sum::<f64>() / pool_size as f64;
            let mean_phenotypes_offspring: Vec<f64> = (0..num_traits)
                .map(|t| {
                    candidates
                        .iter()
                        .map(|c| self.candidate_phenotype(c, t))
                        .sum::<f64>()
                        / pool_size as f64
                })
                .collect();

            // Collect full-pool phenotype vectors for offspring MVN
            let offspring_phenos: Vec<Vec<f64>> = (0..pool_size)
                .map(|i| {
                    (0..num_traits)
                        .map(|t| self.candidate_phenotype(&candidates[i], t))
                        .collect()
                })
                .collect();
            let (_, cov_phenotypes_offspring) = mle_mvn(&offspring_phenos);

            // Density regulation: draw K survivors ∝ fitness.
            let survivor_indices: Vec<usize> = if num_traits == 0 {
                let ud = Uniform::new(0usize, pool_size)?;
                (0..k).map(|_| ud.sample(&mut self.rng)).collect()
            } else {
                let dist = WeightedIndex::new(&fitnesses)?;
                (0..k).map(|_| dist.sample(&mut self.rng)).collect()
            };

            let mean_phenotypes_survivors: Vec<f64> = (0..num_traits)
                .map(|t| {
                    survivor_indices
                        .iter()
                        .map(|&i| self.candidate_phenotype(&candidates[i], t))
                        .sum::<f64>()
                        / k as f64
                })
                .collect();

            // Survivor phenotypes (all K survivors)
            let survivor_phenos: Vec<Vec<f64>> = survivor_indices
                .iter()
                .map(|&i| {
                    (0..num_traits)
                        .map(|t| self.candidate_phenotype(&candidates[i], t))
                        .collect()
                })
                .collect();
            let (_, cov_phenotypes_survivors) = mle_mvn(&survivor_phenos);

            // Count segregating mutations in this deme
            let mut counts: HashMap<usize, usize> = HashMap::new();
            for ind in &self.deme_muts[d] {
                for &(_, idx) in ind {
                    *counts.entry(idx).or_insert(0) += 1;
                }
            }
            let total_n = self.deme_configs[d].carrying_capacity;
            let num_segregating_muts = counts
                .iter()
                .filter(|&(_, count)| *count > 0 && *count < total_n)
                .count();

            gen_deme_records.push(DemeRecord {
                mean_phenotypes_offspring,
                mean_fitness_offspring,
                mean_phenotypes_survivors,
                num_segregating_muts,
                cov_phenotypes_offspring,
                cov_phenotypes_survivors,
            });

            let survivors: Vec<OffspringCandidate> = survivor_indices
                .into_iter()
                .map(|i| candidates[i].clone())
                .collect();
            deme_survivors.push(survivors);
        }

        tracker.record_generation(
            self,
            generation,
            GenerationRecord {
                demes: gen_deme_records,
            },
        )?;

        // ── Phase 2: migration ────────────────────────────────────────────────
        // Build migrant pool: collect n_mig random individuals from each deme.
        let mut migrant_pool: Vec<OffspringCandidate> = Vec::new();
        let mut stayers: Vec<Vec<OffspringCandidate>> = Vec::with_capacity(num_demes);

        for d in 0..num_demes {
            let k = self.deme_configs[d].carrying_capacity;
            let n_mig = (self.deme_configs[d].migration_rate * k as f64).round() as usize;
            let mut survivors = std::mem::take(&mut deme_survivors[d]);
            for _ in 0..n_mig {
                if survivors.is_empty() {
                    break;
                }
                let idx = Uniform::new(0usize, survivors.len())?.sample(&mut self.rng);
                migrant_pool.push(survivors.swap_remove(idx));
            }
            stayers.push(survivors);
        }

        // Shuffle the migrant pool, then distribute.
        migrant_pool.shuffle(&mut self.rng);
        let mut migrant_iter = migrant_pool.into_iter();
        let mut immigrants: Vec<Vec<OffspringCandidate>> = Vec::with_capacity(num_demes);
        for d in 0..num_demes {
            let k = self.deme_configs[d].carrying_capacity;
            let n_mig = (self.deme_configs[d].migration_rate * k as f64).round() as usize;
            immigrants.push(migrant_iter.by_ref().take(n_mig).collect());
        }

        // ── Phase 3: commit all survivors to tables ───────────────────────────
        let mut new_deme_parents: Vec<Vec<tskit::NodeId>> = Vec::with_capacity(num_demes);
        let mut new_deme_muts: Vec<Vec<IndMuts>> = Vec::with_capacity(num_demes);

        for d in 0..num_demes {
            let mut children_d: Vec<tskit::NodeId> = Vec::new();
            let mut child_muts_d: Vec<IndMuts> = Vec::new();

            for c in stayers[d].drain(..) {
                let (node, muts) = self.commit_to_tables(birth_time, c)?;
                children_d.push(node);
                child_muts_d.push(muts);
            }
            for c in immigrants[d].drain(..) {
                let (node, muts) = self.commit_to_tables(birth_time, c)?;
                children_d.push(node);
                child_muts_d.push(muts);
            }

            new_deme_parents.push(children_d);
            new_deme_muts.push(child_muts_d);
        }

        // ── Phase 4: periodic simplification ─────────────────────────────────
        let si = self.params.simplify_interval;
        let do_simplify = si > 0 && self.birth_time % si as i64 == 0;

        if do_simplify {
            let all_children: Vec<tskit::NodeId> =
                new_deme_parents.iter().flatten().copied().collect();

            self.tables
                .sort(&self.bookmark, tskit::TableSortOptions::default())?;
            rotate_edges(&self.bookmark, &mut self.tables);

            if let Some(idmap) = self.tables.simplify(
                &all_children,
                tskit::SimplificationOptions::default(),
                true,
            )? {
                for deme_nodes in new_deme_parents.iter_mut() {
                    for node in deme_nodes.iter_mut() {
                        *node = idmap[usize::try_from(*node)?];
                    }
                }
            }

            self.bookmark.set_edges(self.tables.edges().num_rows());
        }

        self.deme_parents = new_deme_parents;
        self.deme_muts = new_deme_muts;

        if do_simplify {
            self.remove_fixed_mutations();
        }

        self.birth_time -= 1;
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
        self.remove_fixed_mutations();
        tracker.finalize(&self)?;
        let all_parents: Vec<tskit::NodeId> = self.deme_parents.iter().flatten().copied().collect();
        self.tables
            .sort(&self.bookmark, tskit::TableSortOptions::default())?;
        rotate_edges(&self.bookmark, &mut self.tables);
        self.tables.simplify(
            &all_parents,
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
    /// Automatically handles different numbers of traits.
    pub fn finalize_with_metadata(
        mut self,
        records: &[GenerationRecord],
    ) -> Result<tskit::TreeSequence> {
        self.remove_fixed_mutations();
        let all_parents: Vec<tskit::NodeId> = self.deme_parents.iter().flatten().copied().collect();
        self.tables
            .sort(&self.bookmark, tskit::TableSortOptions::default())?;
        rotate_edges(&self.bookmark, &mut self.tables);
        self.tables.simplify(
            &all_parents,
            tskit::SimplificationOptions::KEEP_INPUT_ROOTS,
            false,
        )?;
        self.tables.build_index()?;

        // Create population metadata from records
        let mut populations = tskit::PopulationTable::default();
        let num_demes = self.deme_configs.len();
        for d in 0..num_demes {
            // Extract per-deme data from records
            let mut mean_phenotypes_offspring: Vec<Vec<f64>> = Vec::new();
            let mut mean_fitness_offspring: Vec<f64> = Vec::new();
            let mut mean_phenotypes_survivors: Vec<Vec<f64>> = Vec::new();
            let mut num_segregating_muts: Vec<usize> = Vec::new();
            let mut cov_phenotypes_offspring: Vec<Vec<f64>> = Vec::new();
            let mut cov_phenotypes_survivors: Vec<Vec<f64>> = Vec::new();

            for record in records {
                if d < record.demes.len() {
                    mean_phenotypes_offspring
                        .push(record.demes[d].mean_phenotypes_offspring.clone());
                    mean_fitness_offspring.push(record.demes[d].mean_fitness_offspring);
                    mean_phenotypes_survivors
                        .push(record.demes[d].mean_phenotypes_survivors.clone());
                    num_segregating_muts.push(record.demes[d].num_segregating_muts);
                    cov_phenotypes_offspring
                        .push(record.demes[d].cov_phenotypes_offspring.clone());
                    cov_phenotypes_survivors
                        .push(record.demes[d].cov_phenotypes_survivors.clone());
                }
            }

            let meta = PopulationMetadata {
                deme_id: d,
                mean_phenotypes_offspring,
                mean_fitness_offspring,
                mean_phenotypes_survivors,
                num_segregating_muts,
                cov_phenotypes_offspring,
                cov_phenotypes_survivors,
            };
            let _ = populations.add_row_with_metadata(&meta)?;
        }

        self.tables.set_populations(&populations)?;
        let tree_sequence = self
            .tables
            .tree_sequence(tskit::TreeSequenceFlags::default())?;
        Ok(tree_sequence)
    }
}
