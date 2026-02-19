# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "msprime==1.4.0",
#     "numpy==2.4.2",
#     "pyslim==1.1.0",
#     "scienceplots==2.2.0",
#     "seaborn==0.13.2",
#     "tskit==1.0.1",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import msprime
    import numpy as np
    import tskit

    return msprime, np, tskit


@app.cell
def _():
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    import seaborn as sns

    plt.style.use("science")
    return plt, sns


@app.cell
def _(fwd_ts, msprime, seed):
    demography = msprime.Demography.island_model(
        initial_size=[100, 100], migration_rate=0.1
    )
    msprime_ts = msprime.sim_ancestry(
        samples={"pop_0": 100, "pop_1": 100},
        demography=demography,
        sequence_length=fwd_ts.sequence_length,
        recombination_rate=1e-8,
        random_seed=seed,
        ploidy=1,
    )
    msprime_ts = msprime.sim_mutations(msprime_ts, rate=1e-8, random_seed=seed)
    return (msprime_ts,)


@app.cell
def _(fwd_ts, np):
    assert max(np.array([tree.num_roots for tree in fwd_ts.trees()])) == 1
    return


@app.cell
def _(fwd_ts, msprime_ts):
    fwd_ts.num_trees / msprime_ts.num_trees
    return


@app.cell
def _():
    import marimo as mo

    infile = mo.cli_args().get("file", "notebooks/trees/assortative.trees")
    seed = int(mo.cli_args().get("seed", 42))
    return infile, seed


@app.cell
def _(infile, msprime, seed, tskit):
    fwd_ts = tskit.load(infile)
    fwd_ts = msprime.sim_mutations(fwd_ts, rate=1e-8, random_seed=seed)
    fwd_ts
    return (fwd_ts,)


@app.cell
def _():
    title = r"$N_1=100, N_2=100, m=0.01, s=0.02, \sigma=1.0$"
    return (title,)


@app.cell
def _(fwd_ts):
    sample_sets = [fwd_ts.samples(population_id=0), fwd_ts.samples(population_id=1)]
    return (sample_sets,)


@app.cell
def _(fwd_ts, np):
    windows = np.linspace(0, fwd_ts.sequence_length, 100)
    return (windows,)


@app.cell
def _(fwd_ts, msprime_ts, plt, sample_sets, title, windows):
    plt.figure(figsize=(5, 3))
    plt.stairs(
        fwd_ts.diversity(sample_sets=sample_sets[0], windows=windows, mode="branch")
        / 2,
        windows,
        label="Selection",
    )
    plt.stairs(
        msprime_ts.diversity(sample_sets=sample_sets[0], windows=windows, mode="branch")
        / 2,
        windows,
        label="Neutral",
    )
    plt.xlabel("Position")
    plt.ylabel("TMRCA")
    plt.title(f"Pop0: {title}")
    plt.legend()
    plt.show()
    return


@app.cell
def _(fwd_ts, msprime_ts, plt, sample_sets, title, windows):
    plt.figure(figsize=(5, 3))
    plt.stairs(
        fwd_ts.diversity(sample_sets=sample_sets[1], windows=windows, mode="branch")
        / 2,
        windows,
        label="Selection",
    )
    plt.stairs(
        msprime_ts.diversity(sample_sets=sample_sets[1], windows=windows, mode="branch")
        / 2,
        windows,
        label="Neutral",
    )
    plt.xlabel("Position")
    plt.ylabel("TMRCA")
    plt.title(f"Pop1: {title}")
    plt.legend()
    plt.show()
    return


@app.cell
def _(fwd_ts, msprime_ts, plt, sample_sets, title, windows):
    plt.figure(figsize=(5, 3))
    plt.stairs(
        fwd_ts.divergence(sample_sets=sample_sets, windows=windows, mode="branch") / 2,
        windows,
        label="Selection",
    )
    plt.stairs(
        msprime_ts.divergence(sample_sets=sample_sets, windows=windows, mode="branch")
        / 2,
        windows,
        label="Neutral",
    )
    plt.xlabel("Position")
    plt.ylabel("Cross population TMRCA")
    plt.legend()
    plt.title(f"{title}")
    return


@app.cell
def _(fwd_ts):
    import json

    metadatas = [json.loads(pop.metadata) for pop in fwd_ts.populations()]
    return (metadatas,)


@app.cell
def _(metadatas, np):
    np.array(metadatas[0]["mean_phenotypes_offspring"]).shape
    return


@app.cell
def _(metadatas, np, plt, title):
    plt.plot(np.array(metadatas[0]["mean_phenotypes_offspring"])[:, 0], label="Pop0")
    plt.plot(np.array(metadatas[1]["mean_phenotypes_offspring"])[:, 0], label="Pop1")
    plt.axhline(3, color="k", ls="--")
    plt.axhline(-3, color="k", ls="--")

    plt.xlabel("Generation")
    plt.ylabel("Mean ecological phenotype")
    plt.legend()
    plt.title(f"{title}")
    return


@app.cell
def _(metadatas, np, plt, title):
    plt.plot(np.array(metadatas[0]["mean_phenotypes_offspring"])[:, 1], label="Pop0")
    plt.plot(np.array(metadatas[1]["mean_phenotypes_offspring"])[:, 1], label="Pop1")

    plt.xlabel("Generation")
    plt.ylabel("Mean assortative phenotype")
    plt.legend()
    plt.title(f"{title}")
    return


@app.cell
def _(metadatas, np):
    offspring_0 = np.array(metadatas[0]["all_phenotypes_offspring"])[:, :, 0]
    offspring_1 = np.array(metadatas[1]["all_phenotypes_offspring"])[:, :, 0]
    return offspring_0, offspring_1


@app.cell
def _(np, offspring_0, offspring_1):
    dist_opt_0 = np.sqrt((offspring_0 - (-3)) ** 2).mean(axis=1)
    dist_opt_1 = np.sqrt((offspring_1 - (3)) ** 2).mean(axis=1)
    return dist_opt_0, dist_opt_1


@app.cell
def _(metadatas):
    metadatas[0].keys()
    return


@app.cell
def _(dist_opt_0, dist_opt_1, plt, title):
    plt.plot(dist_opt_0, label="Pop0")
    plt.plot(dist_opt_1, label="Pop0")

    # plt.plot(np.array(metadatas[1]["num_segregating_muts"]).flatten(), label="Pop1")
    plt.xlabel("Generation")
    plt.ylabel("Distance to the optimum")
    plt.legend()
    plt.title(f"{title}")
    return


@app.cell
def _(metadatas, np, plt, title):
    plt.plot(np.array(metadatas[0]["num_segregating_muts"]).flatten(), label="Pop0")
    plt.plot(np.array(metadatas[1]["num_segregating_muts"]).flatten(), label="Pop1")
    plt.xlabel("Generation")
    plt.ylabel("\\# segregating mutations")
    plt.legend()
    plt.title(f"{title}")
    return


@app.cell
def _(metadatas, np, plt, sns, title):
    runtime = len(np.array(metadatas[0]["num_segregating_muts"]).flatten())
    sns.lineplot(
        x=np.arange(runtime),
        y=np.array(metadatas[0]["num_segregating_muts"]).flatten(),
        label="Pop0",
    )
    sns.lineplot(np.array(metadatas[1]["num_segregating_muts"]).flatten(), label="Pop1")
    plt.xlabel("Generation")
    plt.ylabel("\\# segregating mutations")
    plt.legend()
    plt.title(f"{title}")
    return


if __name__ == "__main__":
    app.run()
