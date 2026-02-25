# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`forward-rs` is a Rust-based population genetics simulator implementing the Wright-Fisher model with a **finite sites genetic architecture**:
- Pre-defined bi-allelic loci with direct selection coefficients
- Multiple demes with configurable migration rates
- Efficient tree sequence recording via `tskit`
- Deme-specific selection at each locus

The simulator tracks genotypes and fitness across generations, recording complete genealogies for downstream analysis. The finite sites model provides faster simulations and clearer interpretation compared to infinite sites models.

## Building and Running

### Core Commands

```bash
# Build release binary
cargo build --release

# Run all tests
cargo test

# Run a single test
cargo test test_name -- --exact

# Run example simulations (produces .trees files)
make examples

# Clean build artifacts
make clean
```

### Example Binaries

Located in `src/bin/`:
- `neutral` - Neutral drift in two populations with migration (no loci)
- `divergence` - Divergent selection across two demes (10 loci by default)
- `finite_divergence` - Divergent selection with customizable number of loci (20 by default)

Run individual examples:
```bash
# Neutral model
cargo run --release --bin neutral -- --runtime 10000 --carrying-capacity-deme0 500

# Divergent selection with 20 loci
cargo run --release --bin finite_divergence -- --runtime 5000 --num-loci 20 --selection-coeff 0.05
```

### Code Quality

```bash
# Format code (uses rustfmt via cargo, not the Makefile's uvx ruff)
cargo fmt

# Lint with clippy
cargo clippy --all-targets

# Run with Miri for UB detection (for specific tests)
cargo +nightly miri test
```

## Architecture and Key Concepts

### Core Simulator: `WrightFisher`

The main simulator in `src/lib.rs` manages a finite sites genetic architecture:

**State Components:**
- `tables: tskit::TableCollection` - Stores nodes, edges in tree sequence format
- `architecture: GeneticArchitecture` - Pre-defined loci with selection coefficients
- `genotype_buffers[d]` - Double-buffered genotype storage (current/next generation)
- `fitness_buffers[d]` - Double-buffered fitness values
- `node_buffers[d]` - Double-buffered tskit node IDs
- `breakpoints_buffer` - Reusable buffer for recombination breakpoints
- `deme_records` - Reusable buffer for generation statistics
- `rng: SmallRng` - Reproducible random number generator

**Simulation Pipeline (per generation in `step()`):**

1. **Phase 1: Record Statistics from Current Generation**
   - Compute mean fitness and allele frequencies from current buffers
   - Update `deme_records` in-place (no allocation)
   - Call tracker to record generation data

2. **Phase 2: Generate Next Generation (One Deme at a Time)**
   - For each destination deme:
     - Compute N_immigrants = round(K × migration_rate) and N_local = K - N_immigrants
     - **Part A: Migrants** - Sample N_immigrants from random source demes (on-the-fly)
     - **Part B: Locals** - Sample N_local from destination deme parents
     - All offspring written directly to `next` buffers via `create_offspring_in_place()`
   - **No intermediate allocations** - uses pre-allocated buffers throughout

3. **Phase 3: Swap All Buffers**
   - Swap genotype_buffers, node_buffers, and fitness_buffers for all demes
   - Next generation becomes current generation (zero-cost swap)

4. **Phase 4: Periodic Simplification** (every `simplify_interval` generations)
   - Simplify tree sequences to reduce memory
   - Remap node IDs in current buffers

### Genetic Architecture

- **GeneticArchitecture**: Pre-defined loci with positions and selection coefficients
  - `loci: Vec<Locus>` - Each locus has position and per-deme selection coeffs
  - Validates positions are sorted and selection coeffs match number of demes

- **Locus**: Bi-allelic site with deme-specific selection
  - `position: f64` - Genomic position
  - `selection_coeffs: Vec<f64>` - One per deme (positive = beneficial, negative = deleterious)

- **Genotype Storage**: `Vec<u8>` with 0/1 alleles
  - Pre-allocated to `architecture.num_loci()` per individual
  - O(1) random access (no binary search needed)
  - Mutations toggle bits: `genotype[locus_idx] ^= 1`

### Fitness Calculation

**Multiplicative fitness** (`parental_fitness()`):
```rust
fitness = ∏(1 + s_i) for all loci where genotype[i] == 1
```

For example, with two loci at s = [0.1, -0.05] and genotype = [1, 1]:
```
fitness = (1 + 0.1) × (1 - 0.05) = 1.1 × 0.95 = 1.045
```

### Tracking Statistics

Implement the `TrackerTrait` to record per-generation statistics:
- Inheritance: `record_generation()` called after each generation
- Finalization: `finalize()` called at end to flush results
- Access: Tracker receives `&WrightFisher` to compute custom metrics

See `SimpleTracker` for minimal example; binaries often create custom trackers to write `.trees` files with metadata.

## Key Implementation Details

### Genotype Recombination and Double-Buffering

**Performance Optimization**: The simulator uses double-buffering to eliminate per-generation allocations:
- Each deme has `current` and `next` buffers for genotypes, fitnesses, and node IDs
- `create_offspring_in_place()` writes directly to pre-allocated `next` buffers
- After all offspring are created, buffers are swapped (zero-cost operation)
- **Result**: ~0 allocations per generation (vs ~K allocations in naive approach)

In `create_offspring_in_place()`:
1. Sample recombination breakpoints ~ Poisson(recombination_rate × sequence_length)
   - Reuses `breakpoints_buffer` across all offspring (no allocation)
2. Copy alleles from alternating parents based on locus positions relative to breakpoints
   - Creates temporary genotype, then writes to `next` buffer
3. Apply mutations by randomly toggling bits: `genotype[locus_idx] ^= 1`
   - Number of mutations ~ Poisson(mutation_rate × num_loci)
4. Compute fitness and write to `next` buffer
5. Add tskit edges and node, write node ID to `next` buffer

This is O(num_loci) regardless of how many alleles are derived, versus O(num_muts × log(num_muts)) for sorted mutation lists.

### Random Numbers

Uses `SmallRng` (seeded with `random_seed` parameter) for:
- Reproducible results
- Faster generation than `StdRng`
- Cached Poisson distributions for breakpoints and mutations

### Tree Sequence Integration

- `tskit::NodeId` represents individuals
- Birth time decrements each generation (generation T is recorded at time T)
- `tables.simplify()` maintains ancestry of current population
- `rotate_edges()` utility handles special sorting required by tskit
- **Note**: Genotypes are stored separately, not in tskit mutation table

### Mutation Model

- **Bi-directional**: Mutations can flip 0→1 or 1→0 with equal probability
- **No back-mutation tracking**: Each mutation event randomly selects a locus to toggle
- **Fixed alleles can revert**: Unlike infinite sites, derived alleles at 100% frequency can mutate back to ancestral

## Testing

Tests should verify:
- Genotype inheritance under recombination (test_genotype_recombination)
- Fitness calculations for known genotypes (test_fitness_calculation)
- GeneticArchitecture validation (test_architecture_validation)
- Population size maintenance after migration
- Allele frequency tracking

Note: Use `cfg(test)` for test-only utilities; avoid `#[ignore]` without justification.

## Dependencies and Versions

- **tskit 0.15.0-alpha** (git): Tree sequence library; alpha version due to Rust bindings
- **rand 0.9 + rand_distr 0.5**: Sampling and distributions
- **serde + serde_json**: Metadata serialization
- **clap 4.5**: CLI argument parsing with derive macros
- **indicatif 0.18**: Progress bars
- **anyhow 1.0**: Error context and propagation

## Common Patterns

**Creating a simulation with divergent selection:**
```rust
// Define loci with divergent selection
let loci = vec![
    Locus {
        position: 1e6,
        selection_coeffs: vec![0.05, -0.05], // beneficial in deme 0, deleterious in deme 1
    },
    Locus {
        position: 5e6,
        selection_coeffs: vec![0.1, -0.1],
    },
];
let architecture = GeneticArchitecture::new(loci, 2)?;

let params = Parameters {
    runtime: 1000,
    mutation_rate: 1e-5,
    ..Default::default()
};

let deme_configs = vec![
    DemeConfig {
        carrying_capacity: 500,
        migration_rate: 0.01,
        population_id: tskit::PopulationId::NULL,
    },
    DemeConfig {
        carrying_capacity: 500,
        migration_rate: 0.01,
        population_id: tskit::PopulationId::NULL,
    },
];

let mut sim = WrightFisher::initialize(params, architecture, deme_configs)?;
let mut tracker = SimpleTracker::new();
sim.run(&mut tracker)?;
let ts = sim.finalize_with_metadata(&tracker.records)?;
```

**Accessing simulation state:**
- Population size: sum of `deme_configs[d].carrying_capacity`
- Allele frequencies: `DemeRecord.allele_freqs` from tracker
- Tree quality: `tables.edges().num_rows()` reflects genealogy complexity
- Genotypes: Access via `parental_fitness(d, i)` or directly through buffer fields (private)
  - Note: Direct genotype access requires modifying struct visibility (buffers are private)

## Edition Note

Project uses Rust edition 2024 (very recent). If you encounter compiler errors about features/syntax, check that your toolchain is up to date.
