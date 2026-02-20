#!/usr/bin/env nextflow

/*
 * Pipeline parameters
 */
params.examples = ["neutral", "divergence", "assortative", "two_traits"]
params.seed = 371298

/*
 * Build simulator binary
 */
process build_binary {
    cpus 4
    cache 'lenient'

    input:
    tuple val(name), val(seed), path(notebook)

    output:
    tuple val(name), val(seed), path("${name}"), path(notebook), emit: result

    script:
    """
    cargo build --release --bin ${name} -j $task.cpus --target-dir target
    cp target/release/${name} .
    """
}

/*
 * Run simulator to generate tree sequence
 */
process run_simulator {
    input:
    tuple val(name), val(seed), path(binary), path(notebook)

    output:
    tuple val(name), val(seed), path("${name}.trees"), path(notebook), emit: result

    script:
    """
    ./${binary} --seed ${seed} --output ${name}.trees
    """
}

/*
 * Export notebook to HTML
 */
process export_notebook {
    publishDir "notebooks/html", mode: 'copy'

    input:
    tuple val(name), val(seed), path(trees), path(notebook)

    output:
    tuple val(name), path("${name}.html"), emit: result

    script:
    """
    uvx marimo export html --sandbox ${notebook} -o ${name}.html -- -file ${name}.trees -seed ${seed}
    """
}

workflow {
    main:
    // Create tuples of (example_name, seed) for each example
    examples_ch = channel.fromList(params.examples)
        .map { name -> tuple(name, params.seed, file("notebooks/${name}.py")) }

    // Build binaries
    built = build_binary(examples_ch)

    // Run simulators
    simulated = run_simulator(built.result)

    // Export notebooks
    exported = export_notebook(simulated.result)

    emit:
    html = exported.result
}
