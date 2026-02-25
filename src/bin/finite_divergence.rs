use anyhow::Result;
use clap::Parser;
use forward_rs::*;
use rand::Rng;

#[derive(Parser)]
#[command(about = "Two-deme finite sites model with divergent selection")]
struct Args {
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value_t = 10000)]
    runtime: usize,
    #[arg(long, default_value_t = 500)]
    carrying_capacity: usize,
    #[arg(long, default_value_t = 0.01)]
    migration_rate: f64,
    #[arg(long, default_value_t = 1e-5)]
    mutation_rate: f64,
    #[arg(long, default_value_t = 20)]
    num_loci: usize,
    #[arg(long, default_value_t = 0.05)]
    selection_coeff: f64,
    #[arg(long, default_value = "finite_divergence.trees")]
    output: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let random_seed = args
        .seed
        .unwrap_or_else(|| rand::rng().random_range(1..u64::MAX));

    eprintln!(
        "=== Finite sites: Two demes with {} loci under divergent selection ===",
        args.num_loci
    );
    eprintln!("  Selection coefficient: ±{}", args.selection_coeff);
    eprintln!("  Migration rate: {}", args.migration_rate);
    eprintln!("  Mutation rate (per locus): {}", args.mutation_rate);

    // Create loci with divergent selection
    let seq_len = 1e7;
    let spacing = seq_len / (args.num_loci as f64 + 1.0);
    let loci: Vec<Locus> = (0..args.num_loci)
        .map(|i| {
            let pos = spacing * (i as f64 + 1.0);
            Locus {
                position: pos,
                selection_coeffs: vec![
                    args.selection_coeff,  // deme 0: beneficial
                    -args.selection_coeff, // deme 1: deleterious
                ],
            }
        })
        .collect();

    let architecture = GeneticArchitecture::new(loci, 2)?;

    let params = Parameters {
        random_seed,
        runtime: args.runtime,
        recombination_rate: 1e-8,
        mutation_rate: args.mutation_rate,
        sequence_length: seq_len,
        simplify_interval: args.runtime / 100,
    };

    let deme_configs = vec![
        DemeConfig {
            carrying_capacity: args.carrying_capacity,
            migration_rate: args.migration_rate,
            population_id: tskit::PopulationId::NULL,
        },
        DemeConfig {
            carrying_capacity: args.carrying_capacity,
            migration_rate: args.migration_rate,
            population_id: tskit::PopulationId::NULL,
        },
    ];

    let mut sim = WrightFisher::initialize(params, architecture, deme_configs)?;
    let mut tracker = SimpleTracker::new();

    eprintln!("Running simulation for {} generations...", args.runtime);
    sim.run(&mut tracker)?;

    eprintln!("Finalizing tree sequence...");
    let ts = sim.finalize_with_metadata(&tracker.records)?;

    eprintln!("Simulation complete. Writing to {}", args.output);
    eprintln!("  Total trees: {}", ts.num_trees());
    eprintln!("  Total nodes: {}", ts.nodes().num_rows());

    ts.dump(&args.output, tskit::TableOutputOptions::default())?;
    Ok(())
}
