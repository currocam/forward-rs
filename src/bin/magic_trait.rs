use anyhow::Result;
use clap::Parser;
use forward_rs::*;
use rand::Rng;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    about = "Two-deme Wright-Fisher simulation with divergent selection and assortative mating"
)]
struct Args {
    /// Random seed (random if omitted)
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value_t = 10000)]
    runtime: usize,
    #[arg(long, default_value_t = 100)]
    carrying_capacity_deme0: usize,
    #[arg(long, default_value_t = 100)]
    carrying_capacity_deme1: usize,
    #[arg(long, default_value_t = 0.1)]
    migration_rate: f64,
    #[arg(long, default_value_t = 1e-9)]
    mutation_rate: f64,
    #[arg(long, default_value_t = 1e-8)]
    recombination_rate: f64,
    #[arg(long, default_value_t = 0.02)]
    selection_coeff: f64,
    /// Standard deviation of the Gaussian mating kernel.
    /// Smaller values = stronger assortative mating.
    /// Set very large (e.g. 1e9) for effectively random mating.
    #[arg(long, default_value_t = 10.0)]
    assortative_sigma: f64,
    #[arg(long, default_value = "two_deme_divergent_assortative.trees")]
    output: String,
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!(
        "=== Two demes with divergent selection + assortative mating (σ={}) ===",
        args.assortative_sigma
    );

    let random_seed = args
        .seed
        .unwrap_or_else(|| rand::rng().random_range(1..u64::MAX));

    let params = Parameters {
        mutation_rate: args.mutation_rate,
        recombination_rate: args.recombination_rate,
        runtime: args.runtime,
        random_seed,
        ..Parameters::default()
    };

    let deme_configs = vec![
        DemeConfig {
            carrying_capacity: args.carrying_capacity_deme0,
            migration_rate: args.migration_rate,
            trait_optima: vec![-3.0],
            population_id: tskit::PopulationId::NULL,
        },
        DemeConfig {
            carrying_capacity: args.carrying_capacity_deme1,
            migration_rate: args.migration_rate,
            trait_optima: vec![3.0],
            population_id: tskit::PopulationId::NULL,
        },
    ];

    let traits = vec![GaussianSelectionTrait::new(args.selection_coeff)];

    let mut sim = WrightFisher::initialize(params, traits, deme_configs)?;

    // Enable assortative mating on the single selection trait (index 0).
    sim.assortative_mating = Some(AssortativeMating {
        trait_index: 0,
        sigma: args.assortative_sigma,
    });

    let mut tracker = SimpleTracker::new();
    sim.run(&mut tracker)?;
    let ts = sim.finalize_with_metadata(&tracker.records)?;
    ts.dump(&args.output, tskit::TableOutputOptions::default())?;
    Ok(())
}
