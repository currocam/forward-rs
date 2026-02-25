# /// script
# dependencies = [
#     "jax==0.9.0.1",
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

__generated_with = "0.20.2"
app = marimo.App(width="columns")


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

    plt.style.use("science")
    return (plt,)


@app.cell
def _():
    mainland_size = 500
    island_size = 500
    selection_coeff = 0.05
    migration_rate = 0.005
    mutation_rate = recombination_rate=1e-8
    return (
        island_size,
        mainland_size,
        migration_rate,
        mutation_rate,
        recombination_rate,
        selection_coeff,
    )


@app.cell
def _(
    fwd_ts,
    island_size,
    mainland_size,
    migration_rate,
    msprime,
    mutation_rate,
    recombination_rate,
    seed,
):
    demography = msprime.Demography.isolated_model(
        initial_size=[mainland_size, island_size]
    )
    # Backwards: migration from island to mainland

    demography.add_migration_rate_change(time=0, rate=migration_rate, source="pop_1", dest="pop_0")
    msprime_ts = msprime.sim_ancestry(
        samples={"pop_0": mainland_size, "pop_1": island_size},
        demography=demography,
        sequence_length=fwd_ts.sequence_length,
        recombination_rate=recombination_rate,
        random_seed=seed,
        ploidy=1,
    )
    msprime_ts = msprime.sim_mutations(msprime_ts, rate=mutation_rate, random_seed=seed)
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
    return


@app.cell
def _(infile, msprime, seed, tskit):
    fwd_ts = tskit.load(infile)
    fwd_ts = msprime.sim_mutations(fwd_ts, rate=1e-8, random_seed=seed)
    return (fwd_ts,)


@app.cell
def _():
    title = r"$N_1=500, N_2=100, m=0.005$"
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
def _(fwd_ts, mainland_size, msprime_ts, sample_sets):
    fwd_ts.diversity(sample_sets=sample_sets[0], mode="branch") / 2, mainland_size, msprime_ts.diversity(sample_sets=sample_sets[0], mode="branch") / 2
    return


@app.cell
def _(fwd_ts, msprime_ts, sample_sets):
    fwd_ts.diversity(sample_sets=sample_sets[1], mode="branch") / 2, msprime_ts.diversity(sample_sets=sample_sets[1], mode="branch") / 2
    return


@app.cell
def _(fwd_ts, msprime_ts, sample_sets):
    fwd_ts.divergence(sample_sets=sample_sets, mode="branch") / 2, msprime_ts.divergence(sample_sets=sample_sets, mode="branch") / 2
    return


@app.cell
def _():
    return


@app.cell
def _(mainland_size, migration_rate, msprime_ts, sample_sets):
    msprime_ts.divergence(sample_sets=sample_sets, mode="branch") / 2, mainland_size + 1 / migration_rate
    return


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
    plt.legend()
    plt.gca()
    return


@app.cell
def _():
    return


@app.cell
def _(fwd_ts):
    import json
    metadata =  json.loads(fwd_ts.population(1).metadata)
    return (metadata,)


@app.cell
def _(metadata):
    metadata["allele_freqs"][20]
    return


@app.cell
def _(metadata, np):
    np.array(metadata["allele_freqs"])[:, 0].mean()
    return


@app.cell
def _(metadata, np, plt):
    plt.plot(np.array(metadata["allele_freqs"])[:, 0])
    return


@app.cell
def _(metadata, np, plt):
    plt.plot(np.array(metadata["allele_freqs"])[:, 0])
    return


@app.cell
def _():
    import marimo as mo

    infile = mo.cli_args().get("file", "single_barrier.trees")
    seed = mo.cli_args().get("seed", 12387912)
    return infile, seed


@app.cell
def _():
    import jax
    import jax.numpy as jnp

    def gets(model, i):
        """Extract selection coefficient for locus i.

        Handles both scalar and vector selection coefficients.
        If s is a scalar, returns it; otherwise indexes into the array.
        """
        s = model["s"]
        # If s is scalar (shape () or (1,)), return it; otherwise index
        return jnp.where(s.ndim == 0, s, s[i])

    return jax, jnp


@app.cell
def _(jnp):
    def aeschbacher_model(migration_rate, selection_coeff, positions):
        """Create an Aeschbacher model for genomic flux calculations.

        Args:
            migration_rate: Migration rate between demes
            selection_coeff: Selection coefficient(s) - scalar or array matching positions
            positions: Genomic positions of selected loci

        Returns:
            Dictionary with model parameters
        """
        # Ensure selection coefficient is an array
        s_arr = jnp.atleast_1d(jnp.array(selection_coeff, dtype=jnp.float32))
        pos_arr = jnp.array(positions, dtype=jnp.float32)

        # If s is scalar, broadcast to length of positions
        if s_arr.shape[0] == 1 and pos_arr.shape[0] > 1:
            s_arr = jnp.repeat(s_arr, pos_arr.shape[0])

        return {
            "m": jnp.float32(migration_rate),
            "s": s_arr,
            "xs": jnp.sort(pos_arr),
            # number of MSPs (migration-selection-permutation sites) on either side
            "n": 100
        }

    def recrate(d):
        """Haldane map function: recombination rate as function of distance."""
        return 0.5 * (1 - jnp.exp(-2 * d))

    from jax import lax

    def gff(model, x):
        """Compute genomic flux factor at position x.

        The gff combines selection and recombination effects from nearby loci.
        It finds up to n loci on each side of position x, weights them by
        recombination distance, and accumulates a log-likelihood.

        This implementation is vmap-compatible - use jax.vmap to vectorize over
        multiple positions.

        Args:
            model: Dictionary with 'xs' (locus positions), 's' (selection coeffs),
                   and 'n' (number of loci to consider on each side)
            x: Genomic position to evaluate

        Returns:
            Genomic flux factor (typically between 0 and 1)

        Julia reference implementation:
        struct AeschbacherModel{T,V}
            m :: T          # migration rate
            s :: V          # selection coefficients
            xs:: Vector{T}  # map positions of the selected loci
            n :: Int64      # consider n MSPs on either side
        end

        gets(M::AeschbacherModel{T,V}, i) where {T,V<:AbstractVector} = M.s[i]
        gets(M::AeschbacherModel{T,V}, _) where {T,V<:Real} = M.s

        me(model::AeschbacherModel, x) = model.m*gff(model, x)

        function gff(model::AeschbacherModel, x)
            @unpack xs, n = model
            left = findlast(z->z < x, xs)
            left = isnothing(left) ? 0 : left  # no locus on the left -> 0
            loggff = 0.
            # left
            S = 0.0
            for i=0:n-1
                left - i <= 0 && break
                s = gets(model, left-i)
                r = recrate(abs(xs[left-i] - x))
                loggff += log(r + S) - log(r + S + s)
                S += s
            end
            S = 0.0
            for i=1:n
                left + i > length(xs) && break
                s = gets(model, left+i)
                r = recrate(abs(xs[left+i] - x))
                loggff += log(r + S) - log(r + S + s)
                S += s
            end
            return exp(loggff)
        end
        """
        xs = model["xs"]
        s = model["s"]
        n = model["n"]

        # Find the last locus strictly less than x
        # searchsorted with side='left' gives insertion point; subtract 1 for "last < x"
        left = jnp.searchsorted(xs, x, side='left') - 1

        # Left loop: iterate from left position backwards
        def left_body(i, state):
            loggff, S = state
            idx = left - i
            # Check if index is valid (>= 0)
            in_bounds = idx >= 0

            # Get values using clamped index (prevents out-of-bounds)
            s_val = s[jnp.maximum(idx, 0)]
            r_val = recrate(jnp.abs(xs[jnp.maximum(idx, 0)] - x))

            # Only accumulate if in bounds
            delta = jnp.where(
                in_bounds,
                jnp.log(r_val + S) - jnp.log(r_val + S + s_val),
                0.0
            )
            loggff_new = loggff + delta
            S_new = jnp.where(in_bounds, S + s_val, S)
            return (loggff_new, S_new)

        loggff, _ = lax.fori_loop(0, n, left_body, (0.0, 0.0))

        # Right loop: iterate from left+1 position forwards
        def right_body(i, state):
            loggff, S = state
            idx = left + i
            # Check if index is valid (< len(xs))
            in_bounds = idx < len(xs)

            # Get values using clamped index
            idx_clamped = jnp.minimum(idx, len(xs) - 1)
            s_val = s[idx_clamped]
            r_val = recrate(jnp.abs(xs[idx_clamped] - x))

            # Only accumulate if in bounds
            delta = jnp.where(
                in_bounds,
                jnp.log(r_val + S) - jnp.log(r_val + S + s_val),
                0.0
            )
            loggff_new = loggff + delta
            S_new = jnp.where(in_bounds, S + s_val, S)
            return (loggff_new, S_new)

        loggff, _ = lax.fori_loop(1, n + 1, right_body, (loggff, 0.0))

        # Return exp(loggff)
        return jnp.exp(loggff)

    def me(model, x):
        """Effective migration rate at position x."""
        return model["m"] * gff(model, x)

    return aeschbacher_model, me, recrate


@app.cell
def _(jax, jnp, np, recrate):
    jax.config.update("jax_enable_x64", True)

    # Gauss-Legendre quadrature nodes and weights on [0, 1]
    _nodes, _weights = np.polynomial.legendre.leggauss(200)
    GL_NODES = jnp.array((_nodes + 1) / 2)
    GL_WEIGHTS = jnp.array(_weights / 2)

    def wright_log_density(p, alpha, beta1, beta2, h):
        """Log of unnormalized Wright distribution density.

        Wright(alpha, beta1, beta2, h) where:
            alpha = -N*s   (selection parameter)
            beta1 = N*(me*pbar + u)  (forward mutation/migration)
            beta2 = N*(me*qbar + u)  (backward mutation/migration)
            h = dominance coefficient
        """
        return (
            (2 * beta1 - 1) * jnp.log(p)
            + (2 * beta2 - 1) * jnp.log(1 - p)
            - 2 * alpha * (h * p + (1 - 2 * h) * p**2 / 2)
        )

    def wright_moments(alpha, beta1, beta2, h):
        """Compute (E[p], E[p(1-p)]) under the Wright distribution."""
        log_f = wright_log_density(GL_NODES, alpha, beta1, beta2, h)
        log_f = log_f - jnp.max(log_f)  # numerical stability
        f = jnp.exp(log_f)
        Z = jnp.dot(GL_WEIGHTS, f)
        Ep = jnp.dot(GL_WEIGHTS, GL_NODES * f) / Z
        Epq = jnp.dot(GL_WEIGHTS, GL_NODES * (1 - GL_NODES) * f) / Z
        return Ep, Epq

    def zwaenepoel_model(migration_rate, loci, positions, N, pbar=None, mode=1):
        """Create a Zwaenepoel mainland-island model.

        Args:
            migration_rate: migration rate m
            loci: list of dicts, each with 's' (selection), 'h' (dominance), 'u' (mutation)
            positions: genomic positions of selected loci
            N: effective population size
            pbar: mainland allele frequencies (default: zeros, i.e. beneficial allele absent)
            mode: 1 (haploid migrant) or 2 (diploid migrant)
        """
        L = len(loci)
        pos = jnp.array(positions)
        pbar = jnp.zeros(L) if pbar is None else jnp.array(pbar)
        R = 0.5 * (1 - jnp.exp(-2 * jnp.abs(pos[:, None] - pos[None, :])))
        return {
            "m": float(migration_rate), "loci": loci, "positions": pos,
            "N": float(N), "pbar": pbar, "R": R, "mode": mode,
        }

    def _locuseffect(s, h, p, pq, m, pbar, r):
        q = 1 - p
        qbar = 1 - pbar
        num = -s * h * (qbar - q) + s * (1 - 2 * h) * (pbar * q - pq)
        den = m + r + s * (h - q + 2 * (1 - 2 * h) * pq)
        return num / den

    def _diploidfactor(s, h, p, pq, pbar):
        q = 1 - p
        qbar = 1 - pbar
        return -s * (qbar - q) + s * (1 - 2 * h) * (pbar * qbar - pq)

    def zwaenepoel_equilibrium(model, tol=1e-9, max_iter=1000):
        """Find equilibrium allele frequencies via fixed-point iteration."""
        loci, m, N, pbar, R, mode = (
            model["loci"], model["m"], model["N"],
            model["pbar"], model["R"], model["mode"],
        )
        L = len(loci)
        Ep = jnp.ones(L)
        Epq = Ep * (1 - Ep)

        for _ in range(max_iter):
            # Compute genomic flux factors (excluding focal locus)
            gs = []
            for i in range(L):
                lg = 0.0
                for j in range(L):
                    if i == j:
                        continue
                    s, h = loci[j]["s"], loci[j]["h"]
                    lg += _locuseffect(s, h, Ep[j], Epq[j], m, pbar[j], R[i, j])
                    if mode == 2:
                        lg += _diploidfactor(s, h, Ep[j], Epq[j], pbar[j])
                gs.append(float(jnp.exp(lg)))
            gs = jnp.array(gs)

            # Predict new allele frequencies from Wright distribution
            new_Ep_list, new_Epq_list = [], []
            for i in range(L):
                s, h, u = loci[i]["s"], loci[i]["h"], loci[i]["u"]
                me_i = m * gs[i]
                alpha = -N * s
                beta1 = N * (me_i * float(pbar[i]) + u)
                beta2 = N * (me_i * (1 - float(pbar[i])) + u)
                ep, epq = wright_moments(alpha, beta1, beta2, h)
                new_Ep_list.append(ep)
                new_Epq_list.append(epq)

            new_Ep = jnp.array(new_Ep_list)
            new_Epq = jnp.array(new_Epq_list)

            if jnp.linalg.norm(new_Ep - Ep) < tol:
                Ep, Epq = new_Ep, new_Epq
                break
            Ep, Epq = new_Ep, new_Epq

        return {"model": model, "Ep": Ep, "Epq": Epq}

    def zwaenepoel_me(eq, x):
        """Effective migration rate at arbitrary genomic position x."""
        model = eq["model"]
        m, loci, pbar = model["m"], model["loci"], model["pbar"]
        positions, mode = model["positions"], model["mode"]
        Ep, Epq = eq["Ep"], eq["Epq"]
        lg = 0.0
        for j in range(len(loci)):
            s, h = loci[j]["s"], loci[j]["h"]
            rij = recrate(jnp.abs(x - positions[j]))
            lg += _locuseffect(s, h, Ep[j], Epq[j], m, pbar[j], rij)
            if mode == 2:
                lg += _diploidfactor(s, h, Ep[j], Epq[j], pbar[j])
        return m * jnp.exp(lg)

    return zwaenepoel_equilibrium, zwaenepoel_me, zwaenepoel_model


@app.cell
def _(fwd_ts, jnp, recombination_rate):
    barrier_pos = jnp.array([0.5 * fwd_ts.sequence_length * recombination_rate])
    barrier_pos
    return (barrier_pos,)


@app.cell
def _(aeschbacher_model, barrier_pos, migration_rate, selection_coeff):
    amodel = aeschbacher_model(migration_rate=migration_rate, selection_coeff=selection_coeff, positions=barrier_pos)
    return (amodel,)


@app.cell
def _(
    barrier_pos,
    migration_rate,
    selection_coeff,
    zwaenepoel_equilibrium,
    zwaenepoel_model,
):
    loci = [{"s": selection_coeff, "h": 1.0, "u": 1e-3} for _ in range(len(barrier_pos))]

    zmodel = zwaenepoel_model(
        migration_rate=migration_rate, loci=loci, positions=barrier_pos, N=100,
    )
    eq = zwaenepoel_equilibrium(zmodel)
    return (eq,)


@app.cell
def _(eq, metadata, np, plt):
    import seaborn as sns

    sns.histplot(np.array(metadata["allele_freqs"])[:, 0], bins=20)
    plt.axvline(eq["Epq"])
    return


@app.cell
def _():
    return


@app.cell
def _(
    amodel,
    eq,
    fwd_ts,
    jax,
    jnp,
    mainland_size,
    me,
    plt,
    recombination_rate,
    sample_sets,
    title,
    windows,
    zwaenepoel_me,
):
    xs = jnp.linspace(0, fwd_ts.sequence_length * recombination_rate, 200)
    me_aeschbacher = jax.vmap(lambda x: me(amodel, x))(xs)
    cross_aeschbacher = mainland_size + 1 / me_aeschbacher
    plt.stairs(
        fwd_ts.divergence(sample_sets=sample_sets, windows=windows, mode="branch") / 2,
        windows * recombination_rate,
    )
    plt.plot(
        xs, cross_aeschbacher, label="Aeschbacher model", color="C2", linestyle="--"
    )
    me_zwaenepoel = jax.vmap(lambda x: zwaenepoel_me(eq, x))(xs)

    cross_zwaenepoel = mainland_size + 1 / me_zwaenepoel
    plt.plot(
        xs, cross_zwaenepoel, label="Zwaenepoel model", color="C3", linestyle="--"
    )

    plt.xlabel("Position (Morgan)")
    plt.ylabel("Cross population TMRCA")
    plt.title(f"{title}")
    plt.legend()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
