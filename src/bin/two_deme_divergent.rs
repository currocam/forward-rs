use anyhow::Result;
use clap::Parser;
use forward_rs::*;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(about = "Two-deme Wright-Fisher simulation with divergent selection")]
struct Args {
    #[arg(long, default_value_t = 15000)]
    runtime: usize,
    #[arg(long, default_value_t = 100)]
    carrying_capacity_deme0: usize,
    #[arg(long, default_value_t = 100)]
    carrying_capacity_deme1: usize,
    #[arg(long, default_value_t = 0.01)]
    migration_rate: f64,
    #[arg(long, default_value_t = 1e-9)]
    mutation_rate: f64,
    #[arg(long, default_value_t = 0.02)]
    selection_coeff: f64,
    #[arg(long, default_value = "two_deme_divergent.trees")]
    output: String,
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("=== Two demes with divergent selection (one trait) ===");

    let params = Parameters {
        mutation_rate: args.mutation_rate,
        runtime: args.runtime,
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
    let mut tracker = SimpleTracker::new();
    sim.run(&mut tracker)?;
    let ts = sim.finalize_with_metadata(&tracker.records)?;
    ts.dump(&args.output, tskit::TableOutputOptions::default())?;
    Ok(())
}
