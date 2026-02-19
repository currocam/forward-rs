use anyhow::Result;
use clap::Parser;
use forward_rs::inversion::*;
use forward_rs::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(about = "Two-deme WF simulation with inversions")]
struct Args {
    #[arg(long, default_value_t = 10000)]
    runtime: usize,
    #[arg(long, default_value_t = 5000)]
    burnin: usize,
    #[arg(long, default_value_t = 200)]
    intro_window: usize,
    #[arg(long, default_value_t = 100)]
    carrying_capacity: usize,
    #[arg(long, default_value_t = 0.1)]
    migration_rate: f64,
    #[arg(long, default_value_t = 0.01)]
    selection_coeff: f64,
    #[arg(long, default_value_t = 1e-9)]
    mutation_rate: f64,
    #[arg(long, default_value_t = 1e-7)]
    recombination_rate: f64,
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

    let params = Parameters {
        random_seed,
        mutation_rate: args.mutation_rate,
        recombination_rate: args.recombination_rate,
        simplify_interval: args.simplify_interval,
        runtime: args.runtime,
        ..Parameters::default()
    };
    eprintln!("Params: {:?}", params);

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
    eprintln!("{:?}", deme_configs);

    let traits = vec![GaussianSelectionTrait::new(args.selection_coeff)];
    eprintln!("Traits: {:?}", traits);

    let inversions = vec![Inversion {
        left: args.inv_left,
        right: args.inv_right,
    }];
    eprintln!("Inversions: {:?}", inversions);

    let mut sim = InvWrightFisher::initialize(params, traits, deme_configs, inversions)?;
    let mut tracker = InvTracker::new();

    // Phase 1: burn-in (no inversions).
    let bar = ProgressBar::new(args.runtime as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .expect("Failed to set progress bar style")
        .progress_chars("##-"),
    );
    bar.set_message("Burn-in");

    let burnin = args.burnin;
    for g in 0..burnin {
        sim.step(g, &mut tracker)?;
        bar.inc(1);
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
        bar.set_message(format!("Inversion establishment (attempt {})", attempts));
        for g in burnin..burnin + intro_window {
            sim.step(g, &mut tracker)?;
            bar.inc(1);
        }
        // Introduce inversion in deme 1.
        let inverted_i = sim
            .rng
            .random_range(0..sim.deme_configs[1].carrying_capacity);
        sim.deme_inv_status[1][inverted_i][0] = true;
        for g in burnin + intro_window..runtime {
            sim.step(g, &mut tracker)?;
            bar.inc(1);
            if !sim.inversion_segregating(0) {
                tracker.records.truncate(records_after_checkpoint);
                tracker.inv_freqs.truncate(records_after_checkpoint);
                sim = checkpoint.try_clone()?;
                let rng = SmallRng::seed_from_u64(sim.params.random_seed + attempts as u64);
                sim.rng = rng;
                // Reset progress bar
                bar.set_position(burnin as u64);
                bar.reset_eta();
                continue 'condition;
            }
        }
        break;
    }
    let ts = sim.finalize_with_metadata(&tracker.records, &tracker.inv_freqs)?;
    ts.dump(&args.output, tskit::TableOutputOptions::default())?;
    Ok(())
}
