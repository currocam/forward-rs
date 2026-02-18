use anyhow::Result;
use clap::Parser;
use forward_rs::*;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(about = "Single-deme Wright-Fisher simulation with two traits")]
struct Args {
    #[arg(long, default_value_t = 15000)]
    runtime: usize,
    #[arg(long, default_value_t = 500)]
    carrying_capacity: usize,
    #[arg(long, default_value_t = 1e-9)]
    mutation_rate: f64,
    #[arg(long, default_value_t = 0.05)]
    selection_coeff_trait0: f64,
    #[arg(long, default_value_t = 0.03)]
    selection_coeff_trait1: f64,
    #[arg(long, default_value = "single_deme_two_traits.trees")]
    output: String,
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("=== Single deme with two independent traits under stabilizing selection ===");

    let params = Parameters {
        mutation_rate: args.mutation_rate,
        runtime: args.runtime,
        ..Parameters::default()
    };

    let deme_configs = vec![DemeConfig {
        carrying_capacity: args.carrying_capacity,
        migration_rate: 0.0,
        trait_optima: vec![1.0, -1.0],
        population_id: tskit::PopulationId::NULL,
    }];

    let traits = vec![
        GaussianSelectionTrait::new(args.selection_coeff_trait0),
        GaussianSelectionTrait::new(args.selection_coeff_trait1),
    ];

    let mut sim = WrightFisher::initialize(params, traits, deme_configs)?;
    let mut tracker = SimpleTracker::new();
    sim.run(&mut tracker)?;
    let ts = sim.finalize_with_metadata(&tracker.records)?;
    ts.dump(&args.output, tskit::TableOutputOptions::default())?;
    Ok(())
}
