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
    import json

    import msprime
    import numpy as np
    import tskit

    return json, msprime, np, tskit


@app.cell
def _():
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    plt.style.use("science")
    return (plt,)


@app.cell
def _(fwd_ts, msprime, seed):
    msprime_ts = msprime.sim_ancestry(
        samples=500,
        population_size=500,
        sequence_length=fwd_ts.sequence_length,
        recombination_rate=1e-6,
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

    infile = mo.cli_args().get("file", "notebooks/trees/one_deme.trees")
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
    title = r"$N=500, s_1=0.05, s_2=0.03$"
    return (title,)


@app.cell
def _(fwd_ts, np):
    windows = np.linspace(0, fwd_ts.sequence_length, 100)
    return (windows,)


@app.cell
def _(fwd_ts, msprime_ts, plt, title, windows):
    plt.figure(figsize=(5, 3))
    plt.stairs(
        fwd_ts.diversity(windows=windows, mode="branch") / 2,
        windows,
        label="Selection",
    )
    plt.stairs(
        msprime_ts.diversity(windows=windows, mode="branch") / 2,
        windows,
        label="Neutral",
    )
    plt.xlabel("Position")
    plt.ylabel("TMRCA")
    plt.title(f"{title}")
    plt.legend()
    plt.show()
    return


@app.cell
def _(fwd_ts, json):
    metadatas = [json.loads(pop.metadata) for pop in fwd_ts.populations()]
    return (metadatas,)


@app.cell
def _(metadatas, np, plt, title):
    traj = np.array(metadatas[0]["mean_phenotypes_offspring"])
    thin = 40
    traj = traj[::thin, :]
    plt.figure(figsize=(5, 3))

    colors = np.arange(len(traj)) * thin
    scatter = plt.scatter(
        traj[:, 0], traj[:, 1], c=colors, cmap="viridis", marker="o", s=20
    )
    plt.colorbar(scatter, label="Generation")

    # Add arrows to show direction of time
    for i in range(len(traj) - 1):
        plt.annotate(
            "",
            xy=traj[i + 1],
            xytext=traj[i],
            arrowprops=dict(arrowstyle="->", lw=0.5, color="gray", alpha=0.5),
        )

    plt.xlabel("Trait 1")
    plt.ylabel("Trait 2")
    plt.axhline(-1.0, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    plt.title(f"{title}")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(metadatas, np, plt, title):
    plt.plot(np.array(metadatas[0]["num_segregating_muts"]))
    plt.ylabel("\\# segregating mutations")
    plt.xlabel("Generation")
    plt.tight_layout()
    plt.title(f"{title}")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
