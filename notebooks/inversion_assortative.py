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
    import msprime
    import numpy as np
    import tskit

    return msprime, np, tskit


@app.cell
def _():
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use(["science"])
    return (plt,)


@app.cell
def _(fwd_ts, msprime, seed):
    demography = msprime.Demography.island_model(
        initial_size=[200, 200], migration_rate=0.1
    )
    msprime_ts = msprime.sim_ancestry(
        samples={"pop_0": 100, "pop_1": 100},
        demography=demography,
        sequence_length=fwd_ts.sequence_length,
        recombination_rate=1e-8,
        random_seed=seed,
        ploidy=1,
    )
    msprime_ts = msprime.sim_mutations(msprime_ts, rate=1e-7, random_seed=seed)
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
    seed = mo.cli_args().get("seed", 6870)
    file = mo.cli_args().get("file", "notebooks/trees/inversion_assortative2.trees")
    return file, seed


@app.cell
def _(file, msprime, tskit):
    fwd_ts = tskit.load(file)
    fwd_ts = msprime.sim_mutations(fwd_ts, rate=1e-7, random_seed=42)
    fwd_ts
    return (fwd_ts,)


@app.cell
def _():
    title = r"$N_1=100, N_2=100, m=0.1, s=0.01, \sigma=10$"
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
    plt.axvspan(xmin=3e6, xmax=7e6, color="C3", alpha=0.1, label="Inversion")
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
    plt.axvspan(xmin=3e6, xmax=7e6, color="C3", alpha=0.1, label="Inversion")
    plt.stairs(
        fwd_ts.Fst(sample_sets=sample_sets, windows=windows) / 2,
        windows,
        label="Selection",
    )
    plt.stairs(
        msprime_ts.Fst(sample_sets=sample_sets, windows=windows) / 2,
        windows,
        label="Neutral",
    )
    plt.xlabel("Position")
    plt.ylabel("Fst")
    plt.title(f"Pop0: {title}")
    plt.legend()
    plt.show()
    return


@app.cell
def _(fwd_ts, msprime_ts, plt, sample_sets, title, windows):
    _pop0 = (
        fwd_ts.diversity(sample_sets=sample_sets[0], windows=windows, mode="branch") / 2
    )
    _pop1 = (
        fwd_ts.diversity(sample_sets=sample_sets[1], windows=windows, mode="branch") / 2
    )
    within = (_pop0 + _pop1) / 2
    cross = (
        fwd_ts.divergence(sample_sets=sample_sets, windows=windows, mode="branch") / 2
    )

    plt.axvspan(
        xmin=3e6 * 1e-7, xmax=7e6 * 1e-7, color="C3", alpha=0.1, label="Inversion"
    )
    plt.stairs(
        cross / within,
        windows * 1e-7,
        label="Forward simulation",
    )

    _pop0 = (
        msprime_ts.diversity(sample_sets=sample_sets[0], windows=windows, mode="branch")
        / 2
    )
    _pop1 = (
        msprime_ts.diversity(sample_sets=sample_sets[1], windows=windows, mode="branch")
        / 2
    )
    within = (_pop0 + _pop1) / 2
    cross = (
        msprime_ts.divergence(sample_sets=sample_sets, windows=windows, mode="branch")
        / 2
    )

    plt.xlabel("Position (Morgan)")
    plt.ylabel(r"$\frac{T_\text{cross}}{T_\text{within}}$")
    plt.title(f"{title}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(fwd_ts):
    import json

    metadatas = [json.loads(pop.metadata) for pop in fwd_ts.populations()]
    return (metadatas,)


@app.cell
def _():
    runtime = 60_000
    introduction = 20000 + 10
    return introduction, runtime


@app.cell
def _(metadatas):
    metadatas[0].keys()
    return


@app.cell
def _(np):
    def moving_average(data_set, periods=3):
        weights = np.ones(periods) / periods
        return np.convolve(data_set, weights, mode='valid')

    return (moving_average,)


@app.cell
def _(metadatas, moving_average, np):
    moving_average(np.array(metadatas[0]["mean_phenotypes_survivors"])[:, 0], periods=10).shape, np.array(metadatas[0]["mean_phenotypes_survivors"])[:, 0].shape
    return


@app.cell
def _(introduction, metadatas, moving_average, np, plt, runtime, title):
    _periods = 100
    _half = _periods // 2
    _x_ma = np.arange(_half, runtime - (_periods - 1 - _half))

    _data0 = np.array(metadatas[0]["mean_phenotypes_offspring"])
    _data1 = np.array(metadatas[1]["mean_phenotypes_offspring"])

    _fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    _ax1.plot(_x_ma, moving_average(_data0[:, 0], _periods), label="Pop0")
    _ax1.plot(_x_ma, moving_average(_data1[:, 0], _periods), label="Pop1")
    _ax1.axvspan(introduction, runtime, color="C3", alpha=0.1, label="Inversion")
    _ax1.set_ylabel("Ecological phenotype")
    _ax1.legend()

    _ax2.plot(_x_ma, moving_average(_data0[:, 1], _periods), label="Pop0")
    _ax2.plot(_x_ma, moving_average(_data1[:, 1], _periods), label="Pop1")
    _ax2.axvspan(introduction, runtime, color="C3", alpha=0.1, label="Inversion")
    _ax2.set_ylabel("Assortative phenotype")
    _ax2.set_xlabel("Generation")
    _ax2.set_xlim(0, runtime)
    _ax2.legend()

    _fig.suptitle(title)
    plt.tight_layout()
    plt.show()
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


if __name__ == "__main__":
    app.run()
