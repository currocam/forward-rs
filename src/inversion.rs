use anyhow::{Result, anyhow};
use rand::SeedableRng;
use rand::distr::weighted::WeightedIndex;
use rand::distr::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand_distr::Poisson;
use std::collections::{HashMap, HashSet};

use crate::{
    AssortativeMating, DemeConfig, DemeRecord, GaussianSelectionTrait, GenerationRecord, IndMuts,
    MutationRegistry, Parameters, rotate_edges,
};

// ── Inversion definition ─────────────────────────────────────────────────────
#[derive(Clone, Debug)]
pub struct Inversion {
    /// Start position (inclusive).
    pub left: f64,
    /// End position (exclusive, continuous genome convention).
    pub right: f64,
}

// ── Offspring candidate ──────────────────────────────────────────────────────

#[derive(Clone)]
pub struct InvOffspringCandidate {
    pub birth_deme: usize,
    pub pa: usize,
    pub pb: usize,
    pub breakpoints: Vec<f64>,
    pub inherited_muts: IndMuts,
    pub new_mut_data: Vec<(f64, Vec<f64>)>,
    /// One entry per inversion: whether this offspring carries it.
    pub inv_status: Vec<bool>,
}

// ── Tracker trait ────────────────────────────────────────────────────────────
//#[derive(Clone)]
pub trait InvTrackerTrait {
    fn record_generation(
        &mut self,
        wf: &InvWrightFisher,
        generation: usize,
        record: GenerationRecord,
    ) -> Result<()>;
    fn finalize(&mut self, wf: &InvWrightFisher) -> Result<()>;
}

#[derive(
    serde::Serialize, serde::Deserialize, tskit::metadata::tskit_derive::PopulationMetadata,
)]
#[serializer("serde_json")]
pub struct PopulationMetadata {
    pub deme_id: usize,
    /// Per-generation inversion frequencies for this deme
    pub inv_freqs: Vec<Vec<f64>>,
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

// ── Simulator ────────────────────────────────────────────────────────────────
//#[derive(Clone)]
pub struct InvWrightFisher {
    pub params: Parameters,
    pub tables: tskit::TableCollection,
    pub rng: SmallRng,
    pub birth_time: i64,
    pub bookmark: tskit::types::Bookmark,

    pub registry: MutationRegistry,
    pub traits: Vec<GaussianSelectionTrait>,
    pub deme_configs: Vec<DemeConfig>,
    pub deme_parents: Vec<Vec<tskit::NodeId>>,
    pub deme_muts: Vec<Vec<IndMuts>>,

    pub inversions: Vec<Inversion>,
    /// `deme_inv_status[deme][individual][inv_idx]`
    pub deme_inv_status: Vec<Vec<Vec<bool>>>,

    pub assortative_mating: Option<AssortativeMating>,

    // Cached Poisson distributions (λ = rate × seq_len).
    rec_poisson: Option<Poisson<f64>>,
    mut_poisson: Option<Poisson<f64>>,
}

impl InvWrightFisher {
    pub fn initialize(
        params: Parameters,
        traits: Vec<GaussianSelectionTrait>,
        mut deme_configs: Vec<DemeConfig>,
        inversions: Vec<Inversion>,
    ) -> Result<Self> {
        let mut tables = tskit::TableCollection::new(params.sequence_length)?;

        for dc in deme_configs.iter_mut() {
            dc.population_id = tables.add_population()?;
        }

        let parental_time = params.runtime as f64;
        let num_traits = traits.len();
        let num_inv = inversions.len();

        let mut deme_parents: Vec<Vec<tskit::NodeId>> = Vec::with_capacity(deme_configs.len());
        let mut deme_muts: Vec<Vec<IndMuts>> = Vec::with_capacity(deme_configs.len());
        let mut deme_inv_status: Vec<Vec<Vec<bool>>> = Vec::with_capacity(deme_configs.len());

        for dc in &deme_configs {
            let k = dc.carrying_capacity;
            let pop_id = dc.population_id;
            let parents = (0..k)
                .map(|_| tables.add_node(0, parental_time, pop_id, -1))
                .collect::<Result<Vec<_>, _>>()?;
            deme_parents.push(parents);
            deme_muts.push(vec![IndMuts::new(); k]);
            deme_inv_status.push(vec![vec![false; num_inv]; k]);
        }

        // Validate inversion boundaries.
        for inv in &inversions {
            if inv.left >= inv.right {
                return Err(anyhow!(
                    "Inversion left ({}) must be < right ({})",
                    inv.left,
                    inv.right
                ));
            }
            if inv.left < 0.0 || inv.right > params.sequence_length {
                return Err(anyhow!(
                    "Inversion [{}, {}) out of [0, {})",
                    inv.left,
                    inv.right,
                    params.sequence_length
                ));
            }
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
            inversions,
            deme_inv_status,
            assortative_mating: None,
            params,
            tables,
            rng,
            birth_time,
            bookmark: tskit::types::Bookmark::default(),
            rec_poisson,
            mut_poisson,
        })
    }

    // ── Phenotype ────────────────────────────────────────────────────────────

    pub fn parent_phenotype(&self, d: usize, i: usize, t: usize) -> f64 {
        let baseline = self.traits[t].baseline;
        let inherited: f64 = self.deme_muts[d][i]
            .iter()
            .map(|&(_, idx)| self.registry.effects[idx][t])
            .sum();
        baseline + inherited
    }

    pub fn candidate_phenotype(&self, c: &InvOffspringCandidate, t: usize) -> f64 {
        let baseline = self.traits[t].baseline;
        let inherited: f64 = c
            .inherited_muts
            .iter()
            .map(|&(_, idx)| self.registry.effects[idx][t])
            .sum();
        let new_muts: f64 = c.new_mut_data.iter().map(|(_, eff)| eff[t]).sum();
        baseline + inherited + new_muts
    }

    pub fn candidate_fitness_in_deme(&self, c: &InvOffspringCandidate, d: usize) -> f64 {
        (0..self.traits.len())
            .map(|t| {
                let z = self.candidate_phenotype(c, t);
                let opt = self.deme_configs[d].trait_optima[t];
                self.traits[t].fitness_contribution(z, opt)
            })
            .product()
    }

    // ── Fixed-mutation removal ───────────────────────────────────────────────

    pub fn remove_fixed_mutations(&mut self) {
        if self.traits.is_empty() {
            return;
        }
        let total_n: usize = self
            .deme_configs
            .iter()
            .map(|dc| dc.carrying_capacity)
            .sum();
        assert!(
            total_n > 0,
            "Total carrying capacity must be greater than zero"
        );
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

    // ── Checkpoint (clone) ────────────────────────────────────────────────────

    /// Create a deep copy of the simulator state.
    ///
    /// Takes `&mut self` because it simplifies the tables in place first,
    /// which also benefits the original.
    pub fn try_clone(&mut self) -> Result<Self> {
        // Simplify tables in place with current parents as samples
        let all_parents: Vec<tskit::NodeId> = self.deme_parents.iter().flatten().copied().collect();
        self.tables
            .sort(&self.bookmark, tskit::TableSortOptions::default())?;
        rotate_edges(&self.bookmark, &mut self.tables);
        let mut new_deme_parents = self.deme_parents.clone();
        if let Some(idmap) = self.tables.simplify(
            &all_parents,
            tskit::SimplificationOptions::KEEP_INPUT_ROOTS,
            true,
        )? {
            for deme_nodes in new_deme_parents.iter_mut() {
                for node in deme_nodes.iter_mut() {
                    *node = idmap[usize::try_from(*node)?];
                }
            }
        }
        self.deme_parents = new_deme_parents;
        self.remove_fixed_mutations();

        // Fresh bookmark for the copy
        let mut bookmark = tskit::types::Bookmark::default();
        bookmark.set_edges(self.tables.edges().num_rows());

        // Also reset our own bookmark post-simplification
        self.bookmark = tskit::types::Bookmark::default();
        self.bookmark.set_edges(self.tables.edges().num_rows());

        Ok(Self {
            params: self.params.clone(),
            tables: self.tables.deepcopy()?,
            rng: self.rng.clone(),
            birth_time: self.birth_time,
            bookmark,
            registry: self.registry.clone(),
            traits: self.traits.clone(),
            deme_configs: self.deme_configs.clone(),
            deme_parents: self.deme_parents.clone(),
            deme_muts: self.deme_muts.clone(),
            inversions: self.inversions.clone(),
            deme_inv_status: self.deme_inv_status.clone(),
            assortative_mating: self.assortative_mating.clone(),
            rec_poisson: self.rec_poisson,
            mut_poisson: self.mut_poisson,
        })
    }

    /// Returns true if inversion `inv_idx` is segregating (not fixed, not lost).
    pub fn inversion_segregating(&self, inv_idx: usize) -> bool {
        let total_n: usize = self
            .deme_configs
            .iter()
            .map(|dc| dc.carrying_capacity)
            .sum();
        if total_n == 0 {
            return false;
        }
        let count: usize = self
            .deme_inv_status
            .iter()
            .flatten()
            .filter(|ind| ind[inv_idx])
            .count();
        count > 0 && count < total_n
    }

    // ── Core reproductive step ───────────────────────────────────────────────

    pub fn sample_candidate(
        &mut self,
        d: usize,
        pa: usize,
        pb: usize,
    ) -> Result<InvOffspringCandidate> {
        let seq_len = self.params.sequence_length;

        // ── Recombination breakpoints ────────────────────────────────────────
        let num_bp = match &self.rec_poisson {
            Some(dist) => dist.sample(&mut self.rng) as usize,
            None => 0,
        };
        let mut breakpoints = if num_bp > 0 && seq_len >= 2.0 {
            let bp_dist = Uniform::new(1u64, seq_len as u64)?;
            let mut bps_u64: Vec<u64> =
                (0..num_bp).map(|_| bp_dist.sample(&mut self.rng)).collect();
            bps_u64.sort_unstable();
            bps_u64.dedup();
            bps_u64.iter().map(|&x| x as f64).collect()
        } else {
            vec![]
        };

        // ── Inversion breakpoint adjustment ──────────────────────────────────
        let num_inv = self.inversions.len();
        let mut child_inv_status = vec![false; num_inv];

        for (inv_idx, inv) in self.inversions.iter_mut().enumerate() {
            let inv_left = inv.left;
            let inv_right = inv.right;
            let pa_inv = self.deme_inv_status[d][pa][inv_idx];
            let pb_inv = self.deme_inv_status[d][pb][inv_idx];

            match (pa_inv, pb_inv) {
                (false, false) => {
                    // HOM non-inverted: standard recombination
                    child_inv_status[inv_idx] = false;
                }
                (true, true) => {
                    // HOM inverted: both carry inversion
                    child_inv_status[inv_idx] = true;
                    // Count breakpoints in [left, right)
                    let count = breakpoints
                        .iter()
                        .filter(|&&bp| bp >= inv_left && bp < inv_right)
                        .count();
                    if count % 2 != 0 {
                        // Odd → toggle boundary breakpoints to make even
                        let has_left = breakpoints.contains(&inv_left);
                        let has_right = breakpoints.contains(&inv_right);
                        breakpoints.retain(|&bp| bp != inv_left && bp != inv_right);
                        if !has_left {
                            breakpoints.push(inv_left);
                        }
                        if !has_right {
                            breakpoints.push(inv_right);
                        }
                        let mut bps_u64: Vec<u64> = breakpoints.iter().map(|&x| x as u64).collect();
                        bps_u64.sort_unstable();
                        bps_u64.dedup();
                        breakpoints = bps_u64.iter().map(|&x| x as f64).collect();
                    }
                }
                _ => {
                    // HET: suppress recombination within inversion [left, right)
                    breakpoints.retain(|&bp| bp < inv_left || bp >= inv_right);
                    // Child inherits inversion status from whichever parent
                    // is "current" at position inv.left
                    let n_before = breakpoints.iter().filter(|&&bp| bp < inv_left).count();
                    child_inv_status[inv_idx] = if n_before % 2 == 0 {
                        pa_inv // pa is current
                    } else {
                        pb_inv // pb is current (swapped)
                    };
                }
            }
        }

        // assertions on breakpoints.
        debug_assert!(
            breakpoints.windows(2).all(|w| w[0] < w[1]),
            "Breakpoints not sorted/unique after inversion adjustment: {:?}",
            breakpoints
        );
        debug_assert!(
            breakpoints.iter().all(|&bp| bp > 0.0 && bp < seq_len),
            "Breakpoint out of (0, seq_len): {:?}",
            breakpoints
        );
        assert_eq!(child_inv_status.len(), num_inv);

        // ── Mutation inheritance ─────────────────────────────────────────────
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

        // ── New mutations (effects only; positions deferred to commit time) ──
        let mut new_mut_data: Vec<(f64, Vec<f64>)> = Vec::new();
        if let Some(dist) = &self.mut_poisson {
            let num_new = dist.sample(&mut self.rng) as usize;
            let num_traits = self.registry.num_traits;
            for _ in 0..num_new {
                let effects = self.params.effect_dist.sample(&mut self.rng, num_traits);
                new_mut_data.push((f64::NAN, effects));
            }
        }

        Ok(InvOffspringCandidate {
            birth_deme: d,
            pa,
            pb,
            breakpoints,
            inherited_muts,
            new_mut_data,
            inv_status: child_inv_status,
        })
    }

    pub fn commit_to_tables(
        &mut self,
        birth_time: f64,
        candidate: InvOffspringCandidate,
    ) -> Result<(tskit::NodeId, IndMuts, Vec<bool>)> {
        let seq_len = self.params.sequence_length;
        let d = candidate.birth_deme;
        let parent_a = self.deme_parents[d][candidate.pa];
        let parent_b = self.deme_parents[d][candidate.pb];
        let pop_id = self.deme_configs[d].population_id;

        let mut child_muts = candidate.inherited_muts;
        if !candidate.new_mut_data.is_empty() {
            let pos_dist = Uniform::new(0.0f64, seq_len)?;
            for (_, effects) in candidate.new_mut_data {
                let pos = pos_dist.sample(&mut self.rng);
                let idx = self.registry.push(pos, effects);
                let at = child_muts.partition_point(|&(p, _)| p < pos);
                child_muts.insert(at, (pos, idx));
            }
        }

        let child = self.tables.add_node(0, birth_time, pop_id, -1)?;

        let mut cur_node = parent_a;
        let mut other_node = parent_b;
        let mut start = 0.0f64;

        for &x in &candidate.breakpoints {
            self.tables.add_edge(start, x, cur_node, child)?;
            std::mem::swap(&mut cur_node, &mut other_node);
            start = x;
        }
        self.tables.add_edge(start, seq_len, cur_node, child)?;

        Ok((child, child_muts, candidate.inv_status))
    }

    // ── Generation loop ──────────────────────────────────────────────────────

    pub fn step(&mut self, generation: usize, tracker: &mut dyn InvTrackerTrait) -> Result<()> {
        let birth_time = self.birth_time as f64;
        let num_demes = self.deme_configs.len();
        let num_traits = self.traits.len();
        let num_inv = self.inversions.len();

        // ── Phase 1: per-deme reproduction + density regulation ──────────────
        let mut deme_survivors: Vec<Vec<InvOffspringCandidate>> = Vec::with_capacity(num_demes);
        let mut gen_deme_records: Vec<DemeRecord> = Vec::with_capacity(num_demes);

        for d in 0..num_demes {
            let k = self.deme_configs[d].carrying_capacity;
            let pool_size = ((self.params.fecundity * k as f64) as usize).max(k);
            let uniform_parent = Uniform::new(0usize, k)?;

            // Precompute parental phenotypes for assortative mating (if active).
            let am_data: Option<(Vec<f64>, f64)> =
                self.assortative_mating.as_ref().map(|am| {
                    let zs = (0..k)
                        .map(|i| self.parent_phenotype(d, i, am.trait_index))
                        .collect();
                    (zs, am.sigma)
                });

            let candidates: Vec<InvOffspringCandidate> = (0..pool_size)
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

            let fitnesses: Vec<f64> = if num_traits == 0 {
                vec![1.0f64; pool_size]
            } else {
                candidates
                    .iter()
                    .map(|c| self.candidate_fitness_in_deme(c, d))
                    .collect()
            };

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
            let (_, cov_phenotypes_offspring) = crate::mle_mvn(&offspring_phenos);

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
            let (_, cov_phenotypes_survivors) = crate::mle_mvn(&survivor_phenos);

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

            let survivors: Vec<InvOffspringCandidate> = survivor_indices
                .into_iter()
                .map(|i| candidates[i].clone())
                .collect();

            debug_assert!(
                survivors.iter().all(|s| s.inv_status.len() == num_inv),
                "inv_status length mismatch in survivors"
            );

            deme_survivors.push(survivors);
        }

        tracker.record_generation(
            self,
            generation,
            GenerationRecord {
                demes: gen_deme_records,
            },
        )?;

        // ── Phase 2: migration ───────────────────────────────────────────────
        let mut migrant_pool: Vec<InvOffspringCandidate> = Vec::new();
        let mut stayers: Vec<Vec<InvOffspringCandidate>> = Vec::with_capacity(num_demes);

        for (d, config) in self.deme_configs.iter().enumerate() {
            let k = config.carrying_capacity;
            let n_mig = (config.migration_rate * k as f64).round() as usize;
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

        migrant_pool.shuffle(&mut self.rng);
        let mut migrant_iter = migrant_pool.into_iter();
        let mut immigrants: Vec<Vec<InvOffspringCandidate>> = Vec::with_capacity(num_demes);
        for config in self.deme_configs.iter() {
            let k = config.carrying_capacity;
            let n_mig = (config.migration_rate * k as f64).round() as usize;
            immigrants.push(migrant_iter.by_ref().take(n_mig).collect());
        }

        // ── Phase 3: commit all survivors to tables ──────────────────────────
        let mut new_deme_parents: Vec<Vec<tskit::NodeId>> = Vec::with_capacity(num_demes);
        let mut new_deme_muts: Vec<Vec<IndMuts>> = Vec::with_capacity(num_demes);
        let mut new_deme_inv_status: Vec<Vec<Vec<bool>>> = Vec::with_capacity(num_demes);

        for d in 0..num_demes {
            let mut children_d: Vec<tskit::NodeId> = Vec::new();
            let mut child_muts_d: Vec<IndMuts> = Vec::new();
            let mut child_inv_d: Vec<Vec<bool>> = Vec::new();

            for c in stayers[d].drain(..) {
                let (node, muts, inv) = self.commit_to_tables(birth_time, c)?;
                children_d.push(node);
                child_muts_d.push(muts);
                child_inv_d.push(inv);
            }
            for c in immigrants[d].drain(..) {
                let (node, muts, inv) = self.commit_to_tables(birth_time, c)?;
                children_d.push(node);
                child_muts_d.push(muts);
                child_inv_d.push(inv);
            }

            assert_eq!(
                children_d.len(),
                self.deme_configs[d].carrying_capacity,
                "Deme {} size mismatch after migration",
                d
            );

            new_deme_parents.push(children_d);
            new_deme_muts.push(child_muts_d);
            new_deme_inv_status.push(child_inv_d);
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
        self.deme_inv_status = new_deme_inv_status;

        if do_simplify {
            self.remove_fixed_mutations();
        }

        self.birth_time -= 1;
        Ok(())
    }

    pub fn finalize<T: InvTrackerTrait>(mut self, tracker: &mut T) -> Result<tskit::TreeSequence> {
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
        self.tables.build_index()?;
        let tree_sequence = self
            .tables
            .tree_sequence(tskit::TreeSequenceFlags::default())?;
        Ok(tree_sequence)
    }

    /// Finalize with metadata from tracker (requires concrete tracker type).
    pub fn finalize_with_metadata(
        mut self,
        records: &[GenerationRecord],
        inv_freqs: &[Vec<Vec<f64>>],
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

        // Create population metadata from tracker data
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

            // Extract per-deme inversion frequencies
            let inv_freqs_d: Vec<Vec<f64>> = inv_freqs
                .iter()
                .map(|gen_freqs| {
                    if d < gen_freqs.len() {
                        gen_freqs[d].clone()
                    } else {
                        Vec::new()
                    }
                })
                .collect();

            let meta = PopulationMetadata {
                deme_id: d,
                inv_freqs: inv_freqs_d,
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
