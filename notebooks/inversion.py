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

    # plt.style.use("science")
    return (plt,)


@app.cell
def _(fwd_ts, msprime):
    demography = msprime.Demography.island_model(
        initial_size=[100, 100], migration_rate=0.1
    )
    msprime_ts = msprime.sim_ancestry(
        samples={"pop_0": 100, "pop_1": 100},
        demography=demography,
        sequence_length=fwd_ts.sequence_length,
        recombination_rate=1e-7,
        random_seed=42,
        ploidy=1,
    )
    msprime_ts = msprime.sim_mutations(msprime_ts, rate=1e-7, random_seed=42)
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
    fwd_ts = tskit.load("notebooks/trees/inversion.trees")
    fwd_ts = msprime.sim_mutations(fwd_ts, rate=1e-7, random_seed=42)
    fwd_ts
    return (fwd_ts,)


@app.cell
def _():
    title = r"$N_1=100, N_2=100, m=0.1, s=0.01$"
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
    plt.axvspan(xmin=4e6, xmax=6e6, color="C3", alpha=0.1, label="Inversion")
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
    plt.axvspan(xmin=4e6, xmax=6e6, color="C3", alpha=0.1, label="Inversion")
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
        xmin=4e6 * 1e-7, xmax=6e6 * 1e-7, color="C3", alpha=0.1, label="Inversion"
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
    introduction = 5000 + 200
    runtime = 10000
    return introduction, runtime


@app.cell
def _(introduction, metadatas, np, plt, runtime, title):
    plt.axhline(3, color="k", ls="--")
    plt.axhline(-3, color="k", ls="--")
    plt.axvspan(
        xmin=introduction,
        xmax=runtime,
        color="C3",
        alpha=0.1,
        label="Inversion introgression",
    )
    plt.plot(
        np.array(metadatas[0]["mean_phenotypes_offspring"]).flatten(), label="Pop0"
    )
    plt.plot(
        np.array(metadatas[1]["mean_phenotypes_offspring"]).flatten(), label="Pop1"
    )

    plt.xlabel("Generation")
    plt.ylabel("Mean phenotype")
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
def _(metadatas, np, plt):
    plt.plot(np.array(metadatas[0]["inv_freqs"]))
    plt.plot(np.array(metadatas[1]["inv_freqs"]))
    return


@app.cell
def _(introduction, metadatas, np, plt, runtime, title):
    # Generate an animation of how mean phenotype changes

    plt.axhline(3, color="k", ls="--")
    plt.axhline(-3, color="k", ls="--")
    plt.axvspan(
        xmin=introduction,
        xmax=runtime,
        color="C3",
        alpha=0.1,
        label="Inversion introgression",
    )
    plt.plot(
        np.array(metadatas[0]["mean_phenotypes_offspring"]).flatten(), label="Pop0"
    )
    plt.plot(
        np.array(metadatas[1]["mean_phenotypes_offspring"]).flatten(), label="Pop1"
    )

    plt.xlabel("Generation")
    plt.ylabel("Mean phenotype")
    plt.legend()
    plt.title(f"{title}")
    return


@app.cell
def _(metadatas, np, plt):
    import matplotlib.animation as animation
    import marimo as mo

    fig, ax = plt.subplots(figsize=(8, 5))

    inv_freqs0 = np.array(metadatas[0]["inv_freqs"]).flatten()
    inv_freqs1 = np.array(metadatas[1]["inv_freqs"]).flatten()
    generations = np.arange(len(inv_freqs0))

    # Thin frames: sample every 500 generations
    thin_step = 10
    thin_indices = np.arange(0, len(inv_freqs0), thin_step)
    inv_freqs0_thin = inv_freqs0[thin_indices]
    inv_freqs1_thin = inv_freqs1[thin_indices]
    generations_thin = generations[thin_indices]

    # Set axis limits upfront (use full generation range for context)
    ax.set_xlim(0, len(inv_freqs0))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Inversion Frequency")
    ax.set_title("Inversion Frequency Over Time")
    ax.legend(loc="upper left")

    num_frames = len(inv_freqs0_thin)

    def update(frame):
        ax.clear()
        ax.set_xlim(0, len(inv_freqs0))
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Inversion Frequency")
        ax.set_title("Inversion Frequency Over Time")

        # Plot data up to current frame
        if frame >= 0:
            ax.plot(
                generations_thin[: frame + 1],
                inv_freqs0_thin[: frame + 1],
                lw=2,
                marker="o",
                markersize=4,
                label="Deme 0",
                color="C0",
            )
            ax.plot(
                generations_thin[: frame + 1],
                inv_freqs1_thin[: frame + 1],
                lw=2,
                marker="s",
                markersize=4,
                label="Deme 1",
                color="C1",
            )
        ax.legend(loc="upper left")
        return ax

    ani = animation.FuncAnimation(fig=fig, func=update, frames=num_frames, interval=100)
    plt.close()  # avoid rendering chart twice
    plt.rcParams["animation.html"] = "jshtml"
    mo.Html(ani.to_html5_video())
    return


if __name__ == "__main__":
    app.run()
