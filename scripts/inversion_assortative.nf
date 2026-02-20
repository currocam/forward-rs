#!/usr/bin/env nextflow

/*
 * Pipeline parameters
 */
params.n_replicates       = 10
params.base_seed          = 3971298

// Simulator params (defaults mirror the binary)
params.runtime            = 40000
params.burnin             = 20000
params.intro_window       = 10
params.carrying_capacity  = 100
params.migration_rate     = 0.15
params.selection_coeff    = 0.01
params.mutation_rate      = 1e-9
params.recombination_rate = 1e-8
params.simplify_interval  = 1000
params.inv_left           = 3000000
params.inv_right          = 7000000
params.assortative_sigma  = 10.0

/*
 * Build the inversion_assortative binary once
 */
process build_binary {
    cpus 4
    cache 'lenient'

    output:
    path "inversion_assortative", emit: binary

    script:
    """
    cargo build --release --bin inversion_assortative -j $task.cpus --target-dir target
    cp target/release/inversion_assortative .
    """
}

/*
 * Run one replicate
 */
process run_simulation {
    publishDir "${launchDir}/results/inversion_assortative", mode: 'copy'

    input:
    tuple val(rep), path(binary)

    output:
    path "rep_${rep}.trees", emit: trees

    script:
    def seed = params.base_seed + rep
    """
    ./${binary} \
        --seed ${seed} \
        --runtime ${params.runtime} \
        --burnin ${params.burnin} \
        --intro-window ${params.intro_window} \
        --carrying-capacity ${params.carrying_capacity} \
        --migration-rate ${params.migration_rate} \
        --selection-coeff ${params.selection_coeff} \
        --mutation-rate ${params.mutation_rate} \
        --recombination-rate ${params.recombination_rate} \
        --simplify-interval ${params.simplify_interval} \
        --inv-left ${params.inv_left} \
        --inv-right ${params.inv_right} \
        --assortative-sigma ${params.assortative_sigma} \
        --output rep_${rep}.trees
    """
}

workflow {
    binary = build_binary()

    replicates = channel.of(1..params.n_replicates)
        .combine(binary.binary)

    run_simulation(replicates)
}
