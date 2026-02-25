use anyhow::Result;
use clap::Parser;
use forward_rs::*;
use rand::Rng;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(about = "Single barrier model")]
struct Args {
    /// Random seed (random if omitted)
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value_t = 100000)]
    runtime: usize,
    #[arg(long, default_value_t = 500)]
    mainland_size: usize,
    #[arg(long, default_value_t = 500)]
    island_size: usize,
    #[arg(long, default_value_t = 0.005)]
    migration_rate: f64,
    #[arg(long, default_value_t = 1e-3)]
    mutation_rate: f64,
    #[arg(long, default_value_t = 0.05)]
    selection_coeff: f64,
    #[arg(long, default_value = "single_barrier.trees")]
    output: String,
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    let random_seed = args
        .seed
        .unwrap_or_else(|| rand::rng().random_range(1..u64::MAX));

    // Create loci with divergent selection
    let seq_len = 1e8;
    let recombination_rate = 1e-8;
    let barrier = Locus {
        position: seq_len / 2.0,
        selection_coeffs: vec![
            1.0,                   // strongly beneficial
            -args.selection_coeff, // island deleterious
        ],
    };
    let loci: Vec<Locus> = vec![barrier];
    let architecture = GeneticArchitecture::new(loci, 2)?;

    let params = Parameters {
        random_seed,
        runtime: args.runtime,
        recombination_rate: recombination_rate,
        mutation_rate: args.mutation_rate,
        sequence_length: seq_len,
        simplify_interval: 100,
    };

    let mainland = DemeConfig {
        carrying_capacity: args.mainland_size,
        migration_rate: args.migration_rate,
        population_id: tskit::PopulationId::NULL,
    };
    let island = DemeConfig {
        carrying_capacity: args.island_size,
        // Migration from island to mainland is neglible
        migration_rate: 0.0,
        population_id: tskit::PopulationId::NULL,
    };
    let deme_configs = vec![mainland, island];
    let mut sim = WrightFisher::initialize(params, architecture, deme_configs)?;
    let mut tracker = SimpleTracker::new();
    // We initialize mainland with fixed differences
    for genotypes in sim.get_mut_genotypes(0) {
        assert_eq!(genotypes.len(), args.mainland_size);
        genotypes.fill(1);
    }
    for genotypes in sim.get_mut_genotypes(1) {
        assert_eq!(genotypes.len(), args.island_size);
        genotypes.fill(0);
    }
    sim.run(&mut tracker)?;
    let ts = sim.finalize_with_metadata(&tracker.records)?;
    ts.dump(&args.output, tskit::TableOutputOptions::default())?;
    Ok(())
}
