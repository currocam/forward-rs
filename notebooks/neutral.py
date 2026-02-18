# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "msprime==1.4.0",
#     "numpy==2.4.2",
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
    import numpy as np
    import tskit
    import msprime

    return msprime, np, tskit


@app.cell
def _():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scienceplots

    plt.style.use("science")
    return (plt,)


@app.cell
def _(fwd_ts, msprime):
    demography = msprime.Demography.island_model(
        initial_size=[100, 100], migration_rate=0.01
    )
    msprime_ts = msprime.sim_ancestry(
        samples={"pop_0": 100, "pop_1": 100},
        demography=demography,
        sequence_length=fwd_ts.sequence_length,
        recombination_rate=1e-6,
        random_seed=42,
        ploidy=1,
    )
    msprime_ts = msprime.sim_mutations(msprime_ts, rate=1e-8, random_seed=42)
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
def _(msprime, tskit):
    fwd_ts = tskit.load("notebooks/trees/neutral.trees")
    fwd_ts = msprime.sim_mutations(fwd_ts, rate=1e-8, random_seed=42)
    fwd_ts
    return (fwd_ts,)


@app.cell
def _():
    title = r"$N_1=100, N_2=100, m=0.01$"
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
        label="Fwd",
    )
    plt.stairs(
        msprime_ts.diversity(sample_sets=sample_sets[0], windows=windows, mode="branch")
        / 2,
        windows,
        label="Msprime",
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
        label="Fwd",
    )
    plt.stairs(
        msprime_ts.diversity(sample_sets=sample_sets[1], windows=windows, mode="branch")
        / 2,
        windows,
        label="Msprime",
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
    )
    plt.stairs(
        msprime_ts.divergence(sample_sets=sample_sets, windows=windows, mode="branch")
        / 2,
        windows,
    )
    plt.xlabel("Position")
    plt.ylabel("Cross population TMRCA")
    plt.title(f"{title}")
    return


if __name__ == "__main__":
    app.run()
