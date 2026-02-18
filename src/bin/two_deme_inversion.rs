use anyhow::Result;
use clap::Parser;
use forward_rs::inversion::*;
use forward_rs::*;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(about = "Two-deme WF simulation with inversions")]
struct Args {
    #[arg(long, default_value_t = 15000)]
    runtime: usize,
    #[arg(long, default_value_t = 8000)]
    burnin: usize,
    #[arg(long, default_value_t = 100)]
    intro_window: usize,
    #[arg(long, default_value_t = 100)]
    carrying_capacity: usize,
    #[arg(long, default_value_t = 0.1)]
    migration_rate: f64,
    #[arg(long, default_value_t = 0.01)]
    selection_coeff: f64,
    #[arg(long, default_value_t = 1e-9)]
    mutation_rate: f64,
    #[arg(long, default_value_t = 100)]
    simplify_interval: usize,
    #[arg(long, default_value_t = 4e6)]
    inv_left: f64,
    #[arg(long, default_value_t = 6e6)]
    inv_right: f64,
    /// Random seed (random if omitted)
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value = "two_deme_inversion.trees")]
    output: String,
}

// ── Tracker ──────────────────────────────────────────────────────────────────

struct InvTracker {
    pub records: Vec<GenerationRecord>,
    /// `inv_freqs[generation][deme][inv_idx]`
    pub inv_freqs: Vec<Vec<Vec<f64>>>,
}

impl InvTracker {
    fn new() -> Self {
        Self {
            records: Vec::new(),
            inv_freqs: Vec::new(),
        }
    }
}

impl InvTrackerTrait for InvTracker {
    fn record_generation(
        &mut self,
        wf: &InvWrightFisher,
        _generation: usize,
        record: GenerationRecord,
    ) -> Result<()> {
        // Compute inversion frequencies per deme.
        let num_inv = wf.inversions.len();
        let mut gen_freqs: Vec<Vec<f64>> = Vec::with_capacity(wf.deme_configs.len());
        for d in 0..wf.deme_configs.len() {
            let k = wf.deme_inv_status[d].len();
            let mut freqs = vec![0.0; num_inv];
            if k > 0 {
                for ind in &wf.deme_inv_status[d] {
                    for (inv_idx, &has_inv) in ind.iter().enumerate() {
                        if has_inv {
                            freqs[inv_idx] += 1.0;
                        }
                    }
                }
                for f in &mut freqs {
                    *f /= k as f64;
                }
            }
            gen_freqs.push(freqs);
        }
        self.inv_freqs.push(gen_freqs);
        self.records.push(record);
        Ok(())
    }

    fn finalize(&mut self, _wf: &InvWrightFisher) -> Result<()> {
        Ok(())
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    let random_seed = args
        .seed
        .unwrap_or_else(|| rand::rng().random_range(1..u64::MAX));

    eprintln!("=== Two demes with divergent selection + inversions ===");
    eprintln!("seed={random_seed}");

    let params = Parameters {
        random_seed,
        mutation_rate: args.mutation_rate,
        simplify_interval: args.simplify_interval,
        runtime: args.runtime,
        ..Parameters::default()
    };

    let deme_configs = vec![
        DemeConfig {
            carrying_capacity: args.carrying_capacity,
            migration_rate: args.migration_rate,
            trait_optima: vec![-3.0],
            population_id: tskit::PopulationId::NULL,
        },
        DemeConfig {
            carrying_capacity: args.carrying_capacity,
            migration_rate: args.migration_rate,
            trait_optima: vec![3.0],
            population_id: tskit::PopulationId::NULL,
        },
    ];

    let traits = vec![GaussianSelectionTrait::new(args.selection_coeff)];

    let inversions = vec![Inversion {
        left: args.inv_left,
        right: args.inv_right,
    }];

    let mut sim = InvWrightFisher::initialize(params, traits, deme_configs, inversions)?;
    let mut tracker = InvTracker::new();

    // Phase 1: burn-in (no inversions).
    let burnin = args.burnin;
    for g in 0..burnin {
        if g % 100 == 0 {
            eprintln!("Burn-in generation {}/{}", g, burnin);
        }
        sim.step(g, &mut tracker)?;
    }
    // Checkpoint (simplifies tables in place once).
    let mut checkpoint = sim.try_clone()?;
    let records_after_checkpoint = tracker.records.len();

    // On loss, restore from checkpoint and retry.
    let runtime = sim.params.runtime;
    let intro_window = args.intro_window;
    let mut attempts = 0u32;
    'condition: loop {
        attempts += 1;
        for g in burnin..burnin + intro_window {
            if g % 100 == 0 {
                eprintln!(
                    "Establishment window generation {}/{}",
                    g - burnin,
                    intro_window
                );
            }
            sim.step(g, &mut tracker)?;
        }
        // Introduce inversion at 50% in deme 1.
        let inverted_i = sim
            .rng
            .random_range(0..sim.deme_configs[1].carrying_capacity);
        sim.deme_inv_status[1][inverted_i][0] = true;
        for g in burnin + intro_window..runtime {
            if g % 100 == 0 {
                eprintln!("Attempt {} — generation {}/{}", attempts, g, runtime);
            }
            sim.step(g, &mut tracker)?;
            if !sim.inversion_segregating(0) {
                eprintln!("  → fixed or lost at generation {}, retrying", g);
                tracker.records.truncate(records_after_checkpoint);
                tracker.inv_freqs.truncate(records_after_checkpoint);
                sim = checkpoint.try_clone()?;
                let rng = SmallRng::seed_from_u64(sim.params.random_seed + attempts as u64);
                sim.rng = rng;
                continue 'condition;
            }
        }
        eprintln!(
            "Inversion survived to runtime after {} attempt(s)",
            attempts
        );
        break;
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    let num_records = tracker.records.len();
    eprintln!("Tracker records: {}", num_records);

    // Print inversion frequency trajectory (every 100 generations).
    eprintln!("\nInversion frequency trajectory:");
    eprintln!("{:>6}  {:>10}  {:>10}", "gen", "deme0", "deme1");
    for (g, gen_freqs) in tracker.inv_freqs.iter().enumerate() {
        if g % 100 == 0 || g == num_records - 1 {
            eprintln!(
                "{:>6}  {:>10.4}  {:>10.4}",
                g, gen_freqs[0][0], gen_freqs[1][0]
            );
        }
    }

    // Print phenotype summary.
    if let Some(first) = tracker.records.first() {
        for (d, dr) in first.demes.iter().enumerate() {
            eprintln!(
                "Gen 0 deme {} — mean pheno: {:.4}",
                d,
                dr.mean_phenotypes_offspring.first().copied().unwrap_or(0.0),
            );
        }
    }
    if let Some(last) = tracker.records.last() {
        for (d, dr) in last.demes.iter().enumerate() {
            eprintln!(
                "Gen {} deme {} — mean pheno: {:.4}",
                num_records - 1,
                d,
                dr.mean_phenotypes_offspring.first().copied().unwrap_or(0.0),
            );
        }
    }

    let ts = sim.finalize_with_metadata(&tracker.records, &tracker.inv_freqs)?;
    eprintln!("num_trees: {}", ts.num_trees());
    eprintln!("num_populations: {}", ts.populations().num_rows());
    ts.dump(&args.output, tskit::TableOutputOptions::default())?;
    Ok(())
}
