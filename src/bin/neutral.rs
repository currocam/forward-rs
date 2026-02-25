use anyhow::Result;
use clap::Parser;
use forward_rs::*;
use rand::Rng;
// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(about = "Two-deme neutral Wright-Fisher simulation")]
struct Args {
    /// Random seed (random if omitted)
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value_t = 15000)]
    runtime: usize,
    #[arg(long, default_value_t = 1000)]
    carrying_capacity_deme0: usize,
    #[arg(long, default_value_t = 1000)]
    carrying_capacity_deme1: usize,
    #[arg(long, default_value_t = 0.01)]
    migration_rate: f64,
    #[arg(long, default_value = "neutral.trees")]
    output: String,
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("=== Neutral: Two demes, random drift only ===");

    let random_seed = args
        .seed
        .unwrap_or_else(|| rand::rng().random_range(1..u64::MAX));

    let params = Parameters {
        mutation_rate: 0.0,
        runtime: args.runtime,
        random_seed,
        recombination_rate: 1e-8,
        sequence_length: 1e8,
        simplify_interval: 100,
    };
    eprintln!("{:?}", params);

    // Empty architecture (no loci, neutral model)
    let architecture = GeneticArchitecture::new(vec![], 2)?;

    let deme_configs = vec![
        DemeConfig {
            carrying_capacity: args.carrying_capacity_deme0,
            migration_rate: args.migration_rate,
            population_id: tskit::PopulationId::NULL,
        },
        DemeConfig {
            carrying_capacity: args.carrying_capacity_deme1,
            migration_rate: args.migration_rate,
            population_id: tskit::PopulationId::NULL,
        },
    ];
    eprintln!("{:?}", deme_configs);

    let mut sim = WrightFisher::initialize(params, architecture, deme_configs)?;
    let mut tracker = SimpleTracker::new();
    sim.run(&mut tracker)?;
    let ts = sim.finalize_with_metadata(&tracker.records)?;
    // ── Summary statistics ────────────────────────────────────────────────────
    eprintln!("num_trees: {}", ts.num_trees());
    eprintln!("num_populations: {}", ts.populations().num_rows());
    ts.dump(&args.output, tskit::TableOutputOptions::default())?;
    Ok(())
}
