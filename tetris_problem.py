import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def intro_title(mo):
    mo.md(r"""
    # Tetris algorithm for atomic arrays active sorting
    """)
    return


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## Introduction

    I recently discovered the [Wang et al. 2023 [1]](https://doi.org/10.1103/PhysRevApplied.19.054032) paper, discussing a possible algorithm achieving a fast atomic array reconfiguration, given a detected occupation matrix of atoms. Indeed the random loading of individual atoms in an array of tzeezers leads to a probability half of occupation of each site [[2](https://doi.org/10.1103/PhysRevLett.89.023005),[3](https://doi.org/10.1038/35082512)]. Therefore, the preparation of a dense array – maximazing the number of neighbouring atoms – requires a global reorganisation of the initially loaded array of atoms.

    The proposal made in reference [[1]](https://doi.org/10.1103/PhysRevApplied.19.054032) suggests a sequential (row by row) treatment, compatible with a streaming – row after row – of data from an EMCCD camera, triggering correction operations even before the full image is received by the host server orchestrating the experimental runs.

    In this notebook, I will try to implement a Python version of that algorithm, and reproduce the benchmarkings that are claimed in the publication. This work is a proof of principle, a real production ready version of the program running in a lab should trim many several steps of what follows (only there for testing and pedagogical reasons), and probably also be implemented in a statically typed and compiled language such as C or Rust.
    """)
    return


@app.cell(hide_code=True)
def tetris_algorithm_docs(mo):
    mo.md(r"""
    ## Tetris algorithm implementation

    In what follows, we refer as *loading array* the large array of sites in which can be loaded, and *target array* the subset of sites that must be densely populated after selective sorting. In this notebook we address only the square shape for the loading and target arrays, but the problem can be easily generalized to other shapes (staggered, Kagome, hexagonal...).

    ```

    +---------- loading array ------------+                  +---------- loading array ------------+
    |  .   .   .   o   .   .   .   .   .  |                  |  .   .   .   .   .   .   .   .   .  |
    |  .   o   .   .   .   .   .   .   .  |                  |  .   .   .   .   .   .   o   .   .  |
    |  .   . +-- target array ---+ o   .  |                  |  .   . +-- target array ---+ .   .  |
    |  o   . | o   .   o   o   o | .   .  |                  |  .   . | o   o   o   o   o | .   .  |
    |  .   . | .   o   .   .   . | .   o  | AFTER SELECTIVE  |  .   . | o   o   o   o   o | .   .  |
    |  o   . | .   .   o   o   . | .   .  | ---------------> |  .   . | o   o   o   o   o | .   .  |
    |  .   o +-------------------+ .   .  |     SORTING      |  .   . +-------------------+ .   .  |
    |  .   o   .   .   .   o   .   .   .  |                  |  .   .   .   .   .   .   .   .   .  |
    +-------------------------------------+                  +-------------------------------------+

    .  empty site   o  atom occupying a site                 .  empty site   o  atom occupying a site

    ```
    """)
    return


@app.cell
def imports():
    # loading dependencies

    # built-in
    import os
    from math import sqrt
    from pathlib import Path

    # third-party
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # this notebook's own package
    from tetris import AtomsConfiguration
    from tetris.benchmark import (
        run_displacement_benchmark,
        run_success_rate_benchmark,
    )
    from tetris.cache import (
        BenchmarkParameters,
        algorithm_digest,
        load_results,
        save_results,
    )

    return (
        AtomsConfiguration,
        BenchmarkParameters,
        Path,
        algorithm_digest,
        load_results,
        mo,
        np,
        os,
        plt,
        run_displacement_benchmark,
        run_success_rate_benchmark,
        save_results,
        sqrt,
    )


@app.cell
def notebook_constants(Path, os):
    # general static parameters for the notebook (usually default values)
    DEFAULT_LOADING_ARRAY_SIZE: int = 18

    # benchmark sweep
    LOADING_ARRAY_SIZES: tuple[int, ...] = tuple(range(4, 101))
    MARGIN_VARIANTS: tuple[bool, ...] = (False, True)
    N_WORKERS: int = max(1, (os.cpu_count() or 1) - 1)

    # success rate: a fixed wall clock budget per task. The cost of a
    # configuration is taken from each worker's own CPU time, never from
    # this number, so a task that got a core to itself does not read as
    # a faster algorithm.
    SECONDS_PER_TASK: float = 60.0

    # displacements: a fixed sample count instead, so that every point
    # carries the same weight. Wang et al. average 10000 runs per point.
    DISPLACEMENT_SIZES: tuple[int, ...] = tuple(range(8, 101, 4))
    DISPLACEMENT_SAMPLES: int = 10_000

    # curve styles shared by the plotting cells
    VARIANT_STYLES: tuple[tuple[str, str, str], ...] = (
        ("no_margin", "No margin", "o"),
        ("with_margin", "With margin", "s"),
    )

    _root = Path(__file__).parent
    # caches, so a plain re-run doesn't retrigger the computation
    BENCHMARK_STATS_PATH: Path = _root / "benchmark_stats.json"
    DISPLACEMENT_STATS_PATH: Path = _root / "displacement_stats.json"
    # sources whose content decides what the benchmark measures: editing
    # them invalidates both caches
    ALGORITHM_SOURCES: tuple[Path, ...] = (
        _root / "tetris" / "config.py",
        _root / "tetris" / "fast.py",
    )
    return (
        ALGORITHM_SOURCES,
        BENCHMARK_STATS_PATH,
        DEFAULT_LOADING_ARRAY_SIZE,
        DISPLACEMENT_SAMPLES,
        DISPLACEMENT_SIZES,
        DISPLACEMENT_STATS_PATH,
        LOADING_ARRAY_SIZES,
        MARGIN_VARIANTS,
        N_WORKERS,
        SECONDS_PER_TASK,
        VARIANT_STYLES,
    )


@app.cell(hide_code=True)
def atoms_configuration_docs(mo):
    mo.md(r"""
    The algorithm lives in the `tetris` package next to this notebook, rather than in a cell, so that it can be imported by the benchmark workers and covered by tests. `AtomsConfiguration` is in charge of:

    1. Generating a random occupation matrix (probability half for each loading site) – or alternatively loading a manually chosen one.
    2. Simulating the tetris algorithm execution flow on the generated occupation matrix. This is achieved by the `construct_tetriminoes` method which creates several attributes: `tetriminoes_matrix` (the matrix with atoms horizontally compactified), `tetriminoes_motions` (the list of motions to realize so as to prepare the tetrominoes matrix, starting from the loaded matrix).
    3. A `configuration_kept` attribute is also created after the execution of `construct_tetriminoes`, to know if a loaded configuration should be discarded before starting the vertical stacking of the atoms. This happens if – after `construct_tetriminoes` execution – at least one column of the loading array does not contain enough atoms to fill the corresponding column of the target array.
    4. Counting the parallel displacements the rearrangement costs, with `count_parallel_displacements`.
    5. Plotting the loaded and tetriminoes configurations, with the `plot_configuration` method.

    A row is packed by dealing its atoms round the target window from a running shift. For a uniform square target that is the same assignment as the rule stated in [[1]](https://doi.org/10.1103/PhysRevApplied.19.054032), which always serves the columns that are furthest behind, and the two agree on every configuration the test suite throws at them.
    """)
    return
@app.cell(hide_code=True)
def testing_class_intro(mo):
    mo.md(r"""
    ### Testing the class

    Let's generate a randomly loaded matrix, visualize it and execute the tetris algorithm on it.
    """)
    return


@app.cell
def demo_random_configuration(
    AtomsConfiguration,
    DEFAULT_LOADING_ARRAY_SIZE: int,
):
    atoms_config = AtomsConfiguration(DEFAULT_LOADING_ARRAY_SIZE)
    atoms_config.plot_configuration()
    return (atoms_config,)


@app.cell
def demo_verbose_tetriminoes(atoms_config):
    atoms_config.construct_tetriminoes(verbose=True)
    return


@app.cell
def demo_plot_all(atoms_config):
    atoms_config.plot_configuration(which="all", size=4)
    return


@app.cell(hide_code=True)
def large_array_intro(mo):
    mo.md(r"""
    Let's try with a much larger number of atoms.
    """)
    return


@app.cell
def large_array_demo(AtomsConfiguration):
    array = AtomsConfiguration(loading_array_size=200)
    array.construct_tetriminoes()
    array.plot_configuration(which="all")
    return


@app.cell(hide_code=True)
def benchmarking_intro(mo):
    mo.md(r"""
    ## Benchmarking

    We compute the rejection rate for different loading array sizes. A priori the target array size $\ell$ is chosen to have roughly half of the number of sites in the loading area (since the loading probability is 50% for each site), hence, with $L$ the loading array size and $\lfloor \bullet \rfloor$ the floor function:
    \[
        \ell = \lfloor L / \sqrt{2} \rfloor
    \]
    In practice, to improve the success probability we can also reduce that quantity by one unit, to have a security margin:
    \[
        \ell_{\text{margin}} = \ell_{\text{no margin}} - 1 = \lfloor L / \sqrt{2} \rfloor - 1
    \]
    The margin variant is the one matching [[1]](https://doi.org/10.1103/PhysRevApplied.19.054032), whose simulations start from a $\lceil \sqrt{2}\ell + 1 \rceil$ square reservoir for an $\ell \times \ell$ target.

    ### Computation

    We take $L \in [4, 100]$, and $\ell$ with and without security margin. Each $(L, \ell)$ configuration gets its own worker, its own random stream and a 60 s wall clock budget. Three details matter for what comes out of it.

    Each worker seeds itself from a `SeedSequence` child of one recorded entropy value. NumPy's global generator survives a `fork` unchanged, so workers sharing it would draw identical configurations, and the two margin variants at a given $L$ would be paired rather than independent.

    Tasks are drawn from a single queue in shuffled order. Running them in size order lets any drift over the half hour the sweep takes – a laptop heating up, for instance – arrive as a trend against $L$.

    The cost of a configuration is measured from each worker's own CPU time, not from the 60 s budget. Workers do not all get the same share of the machine: a task that happens to run when few others do gets a physical core to itself and completes far more runs, which under the old accounting read as a faster algorithm.
    """)
    return


@app.cell
def benchmark_rejection_rate(
    ALGORITHM_SOURCES,
    BENCHMARK_STATS_PATH,
    BenchmarkParameters,
    LOADING_ARRAY_SIZES,
    MARGIN_VARIANTS,
    N_WORKERS,
    SECONDS_PER_TASK,
    algorithm_digest,
    load_results,
    np,
    run_success_rate_benchmark,
    save_results,
):
    success_rate_parameters = BenchmarkParameters(
        sizes=LOADING_ARRAY_SIZES,
        margin_variants=MARGIN_VARIANTS,
        seconds_per_task=SECONDS_PER_TASK,
        workers=N_WORKERS,
        algorithm_digest=algorithm_digest(*ALGORITHM_SOURCES),
    )
    stats, _reason = load_results(
        BENCHMARK_STATS_PATH, success_rate_parameters
    )
    if stats is None:
        print(f"running the sweep, {_reason}")
        _entropy = np.random.SeedSequence().entropy
        stats = run_success_rate_benchmark(
            sizes=LOADING_ARRAY_SIZES,
            margin_variants=MARGIN_VARIANTS,
            seconds_per_task=SECONDS_PER_TASK,
            workers=N_WORKERS,
            entropy=_entropy,
        )
        save_results(
            BENCHMARK_STATS_PATH,
            success_rate_parameters,
            _entropy,
            stats,
        )
    return (stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting the results
    """)
    return


@app.cell
def _(np, stats):
    # arrays reused by several of the plotting cells below
    sizes_arr = np.array(sorted(stats.keys()))
    return (sizes_arr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Success rate

    We can start by brutally plotting the raw results, separating the cases with and without the security margin. The bars are Wilson score intervals at one standard deviation. They are small, since each point rests on thousands to hundreds of thousands of runs, and they are asymmetric near a rate of one, where a symmetric bar would reach past one.
    """)
    return


@app.cell
def success_rate_plot(VARIANT_STYLES, np, plt, sizes_arr, stats):
    fig_success_rate, (ax_success_full, ax_success_zoom) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True
    )
    for _variant, _label, _marker in VARIANT_STYLES:
        _rates = np.array(
            [stats[_size][_variant]["success_rate"] for _size in sizes_arr]
        )
        _errors = np.vstack(
            [
                _rates
                - [stats[_s][_variant]["wilson_low"] for _s in sizes_arr],
                [stats[_s][_variant]["wilson_high"] for _s in sizes_arr]
                - _rates,
            ]
        )
        for _ax in (ax_success_full, ax_success_zoom):
            _ax.errorbar(
                sizes_arr,
                _rates,
                yerr=_errors,
                fmt=_marker,
                markersize=4,
                capsize=2,
                label=_label,
            )

    ax_success_full.set_ylabel("Success rate")
    ax_success_full.set_title("Success rate vs. loading array size")
    ax_success_full.grid(alpha=0.3)
    ax_success_full.legend()

    ax_success_zoom.set_ylim(0.99, 1.0005)
    ax_success_zoom.set_xlabel("Loading array size $L$")
    ax_success_zoom.set_ylabel("Success rate")
    ax_success_zoom.set_title("Same data, zoomed on the margin variant")
    ax_success_zoom.grid(alpha=0.3)

    fig_success_rate.tight_layout()
    fig_success_rate
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Without security margin, the success rate of the procedure can be really bad, and it exhibits structure that comes from the truncation happening when the loading size is converted into the target size: $\ell$ is a floor, so the contraction ratio $(\ell/L)^2$ jumps around as $L$ grows rather than settling on $1/2$. Plotting against that ratio instead of against $L$ collapses most of it.
    """)
    return


@app.cell
def success_rate_vs_contraction_ratio_plot(VARIANT_STYLES, np, plt, stats):
    fig_success_vs_ratio, ax_success_vs_ratio = plt.subplots(figsize=(10, 4))
    for _variant, _label, _marker in VARIANT_STYLES:
        _entries = sorted(
            (
                stats[_size][_variant]["contraction_ratio"],
                stats[_size][_variant]["success_rate"],
                stats[_size][_variant]["wilson_low"],
                stats[_size][_variant]["wilson_high"],
            )
            for _size in stats
        )
        _ratios, _rates, _low, _high = (
            np.array(_column) for _column in zip(*_entries)
        )
        ax_success_vs_ratio.errorbar(
            _ratios,
            _rates,
            yerr=np.vstack([_rates - _low, _high - _rates]),
            fmt=_marker,
            markersize=4,
            capsize=2,
            label=_label,
        )
    ax_success_vs_ratio.set_xlabel("Contraction ratio $(\\ell/L)^2$")
    ax_success_vs_ratio.set_ylabel("Success rate")
    ax_success_vs_ratio.set_title("Success rate vs. contraction ratio")
    ax_success_vs_ratio.grid(alpha=0.3)
    ax_success_vs_ratio.legend()
    fig_success_vs_ratio.tight_layout()
    fig_success_vs_ratio
    return


@app.cell
def success_rate_heatmap(VARIANT_STYLES, np, plt, stats):
    # (loading array size, contraction ratio) isn't a regular grid — each
    # size only has two ratios (no_margin/with_margin) — so a scatter
    # colored by success rate stands in for an imshow-style heatmap
    from matplotlib.colors import LinearSegmentedColormap

    _cmap = LinearSegmentedColormap.from_list(
        "purple_to_green", ["tab:purple", "tab:red", "tab:green"]
    )
    _variants = tuple(_style[0] for _style in VARIANT_STYLES)

    def _column(key):
        return np.array(
            [stats[_size][_variant][key] for _size in stats
             for _variant in _variants]
        )

    _sizes = np.array([_size for _size in stats for _ in _variants])
    _ratios = _column("contraction_ratio")
    _rates = _column("success_rate")
    # relative precision of each point, which unlike the spread of the
    # Bernoulli draws does say how well the rate is known
    _relative_errors = 100 * _column("standard_error") / _rates

    fig_heatmap, (ax_heatmap, ax_heatmap_error) = plt.subplots(
        1, 2, figsize=(16, 6)
    )
    _scatter = ax_heatmap.scatter(
        _sizes, _ratios, c=_rates, cmap=_cmap, vmin=0.5, vmax=1, s=40
    )
    fig_heatmap.colorbar(_scatter, ax=ax_heatmap, label="Success rate")
    ax_heatmap.set_xlabel("Loading array size $L$")
    ax_heatmap.set_ylabel("Contraction ratio")
    ax_heatmap.set_title("Success rate")

    _scatter_error = ax_heatmap_error.scatter(
        _sizes, _ratios, c=_relative_errors, cmap="viridis", s=40
    )
    fig_heatmap.colorbar(
        _scatter_error,
        ax=ax_heatmap_error,
        label="standard error / success rate (%)",
    )
    ax_heatmap_error.set_xlabel("Loading array size $L$")
    ax_heatmap_error.set_ylabel("Contraction ratio")
    ax_heatmap_error.set_title("Relative precision of each point")

    fig_heatmap.tight_layout()
    fig_heatmap
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Cost of a configuration

    Everything above is Python and not heavily optimized, so this says more about the implementation than about the algorithm. It is worth looking at anyway, if only to see how little of it is the algorithm.

    The cost is the CPU time a worker spent, divided by the number of configurations it examined. Below $L \approx 30$ it barely moves: a row costs a fixed handful of NumPy calls whatever its length, and that overhead swamps the work. The fit is therefore restricted to the large end, where the per-site work has taken over. Even there the exponent stays close to 1 rather than the 2 the $L^2$ sites would suggest, so the overhead never really lets go over this range.

    Individual points scatter by a few tens of percent. CPU time removes the effect of a worker being descheduled, but not the effect of sharing a physical core with another worker: two threads on one core each retire fewer instructions per second, and the time still counts. Shuffling the task order turns that into noise rather than a trend against $L$, which is the most that can be done without a serial timing pass.
    """)
    return


@app.cell
def computation_cost_plot(VARIANT_STYLES, np, plt, sizes_arr, stats):
    FIT_FROM = 30

    fig_cost, (ax_cost, ax_cost_per_row) = plt.subplots(
        1, 2, figsize=(11, 4)
    )
    _fit_sizes, _fit_costs = [], []
    for _variant, _label, _marker in VARIANT_STYLES:
        _costs = np.array(
            [
                stats[_size][_variant]["microseconds_per_configuration"]
                for _size in sizes_arr
            ]
        )
        ax_cost.loglog(
            sizes_arr, _costs, _marker, markersize=4, alpha=0.75, label=_label
        )
        ax_cost_per_row.semilogx(
            sizes_arr,
            _costs / sizes_arr,
            _marker,
            markersize=4,
            alpha=0.75,
            label=_label,
        )
        _mask = sizes_arr >= FIT_FROM
        _fit_sizes.append(np.log(sizes_arr[_mask]))
        _fit_costs.append(np.log(_costs[_mask]))

    _log_sizes = np.concatenate(_fit_sizes)
    _log_costs = np.concatenate(_fit_costs)
    _fit_range = sizes_arr[sizes_arr >= FIT_FROM]
    if _log_sizes.size > 2:
        (_slope, _intercept), _covariance = np.polyfit(
            _log_sizes, _log_costs, 1, cov=True
        )
        ax_cost.loglog(
            _fit_range,
            np.exp(_intercept) * _fit_range**_slope,
            "k--",
            label=(
                rf"$L^{{{_slope:.2f}({np.sqrt(_covariance[0, 0]):.2f})}}$"
                rf", fitted from $L={FIT_FROM}$"
            ),
        )

    ax_cost.set_xlabel("Loading array size $L$")
    ax_cost.set_ylabel("CPU time per configuration (µs)")
    ax_cost.set_title("Cost of one configuration")
    ax_cost.grid(alpha=0.3, which="both")
    ax_cost.legend()

    ax_cost_per_row.set_xlabel("Loading array size $L$")
    ax_cost_per_row.set_ylabel("CPU time per row (µs)")
    ax_cost_per_row.set_title("Per-row cost, flat because it is overhead")
    ax_cost_per_row.grid(alpha=0.3, which="both")
    ax_cost_per_row.legend()

    fig_cost.tight_layout()
    fig_cost
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Rearrangement cost

    Computation time is not what [[1]](https://doi.org/10.1103/PhysRevApplied.19.054032) reports, and it is not what dominates an experimental cycle either: moving atoms takes milliseconds, deciding where to move them takes microseconds. The quantity the paper fits is the number of *parallel displacements*, the sum over moves of the largest single-atom displacement in each, since all the atoms of a row travel at once. There are about $L$ row moves and $\ell$ column moves, so the count scales at most as $L^2 \propto N$.

    Their Monte Carlo gives $N^{1.03(7)}$ for the compact geometry with the Tetris algorithm, against $N^{1.6(1)}$ for the Hungarian algorithm, which is the strict optimum without parallel moves and is the number this notebook previously compared itself against by mistake. $N$ is the number of atoms in the target array, $\ell^2$, not the linear size.

    Each point below averages 10000 accepted configurations, a fixed count rather than a fixed time, matching the paper.
    """)
    return


@app.cell
def benchmark_displacements(
    ALGORITHM_SOURCES,
    BenchmarkParameters,
    DISPLACEMENT_SAMPLES,
    DISPLACEMENT_SIZES,
    DISPLACEMENT_STATS_PATH,
    N_WORKERS,
    algorithm_digest,
    load_results,
    np,
    run_displacement_benchmark,
    save_results,
):
    displacement_parameters = BenchmarkParameters(
        sizes=DISPLACEMENT_SIZES,
        margin_variants=(True,),
        seconds_per_task=0.0,
        samples_per_task=DISPLACEMENT_SAMPLES,
        workers=N_WORKERS,
        algorithm_digest=algorithm_digest(*ALGORITHM_SOURCES),
    )
    displacements, _reason = load_results(
        DISPLACEMENT_STATS_PATH, displacement_parameters
    )
    if displacements is None:
        print(f"counting displacements, {_reason}")
        _entropy = np.random.SeedSequence().entropy
        displacements = run_displacement_benchmark(
            sizes=DISPLACEMENT_SIZES,
            samples_per_task=DISPLACEMENT_SAMPLES,
            workers=N_WORKERS,
            entropy=_entropy,
        )
        save_results(
            DISPLACEMENT_STATS_PATH,
            displacement_parameters,
            _entropy,
            displacements,
        )
    return (displacements,)


@app.cell
def displacement_plot(displacements, np, plt):
    PAPER_TETRIS_EXPONENT = 1.03
    PAPER_HUNGARIAN_EXPONENT = 1.6

    _atoms = np.array(
        [
            displacements[_size]["with_margin"]["target_number_of_atoms"]
            for _size in sorted(displacements)
        ]
    )
    _means = np.array(
        [
            displacements[_size]["with_margin"]["mean"]
            for _size in sorted(displacements)
        ]
    )
    _errors = np.array(
        [
            displacements[_size]["with_margin"]["standard_error"]
            for _size in sorted(displacements)
        ]
    )

    fig_displacements, ax_displacements = plt.subplots(figsize=(7, 5))
    ax_displacements.errorbar(
        _atoms,
        _means,
        yerr=_errors,
        fmt="o",
        markersize=4,
        capsize=2,
        label="This implementation, with margin",
    )
    if _atoms.size > 2:
        (_slope, _intercept), _covariance = np.polyfit(
            np.log(_atoms), np.log(_means), 1, cov=True
        )
        ax_displacements.plot(
            _atoms,
            np.exp(_intercept) * _atoms**_slope,
            "k--",
            label=(
                rf"Fit: "
                rf"$N^{{{_slope:.2f}({np.sqrt(_covariance[0, 0]):.2f})}}$"
            ),
        )
    for _exponent, _label, _style in (
        (PAPER_TETRIS_EXPONENT, "Paper, Tetris: $N^{1.03(7)}$", "-"),
        (PAPER_HUNGARIAN_EXPONENT, "Paper, Hungarian: $N^{1.6(1)}$", ":"),
    ):
        # anchored on the first point, only the slope is being compared
        ax_displacements.plot(
            _atoms,
            _means[0] * (_atoms / _atoms[0]) ** _exponent,
            _style,
            alpha=0.6,
            label=_label,
        )

    ax_displacements.set_xscale("log")
    ax_displacements.set_yscale("log")
    ax_displacements.set_xlabel("Target atom number $N = \\ell^2$")
    ax_displacements.set_ylabel("Parallel displacements")
    ax_displacements.set_title("Rearrangement cost vs. atom number")
    ax_displacements.grid(alpha=0.3, which="both")
    ax_displacements.legend()

    fig_displacements.tight_layout()
    fig_displacements
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Where the structure in the no-margin curve comes from

    The target side is a floor, so as $L$ grows the fractional part of $L/\sqrt{2}$ sweeps through $[0, 1)$ and the contraction ratio $(\ell/L)^2$ oscillates around $1/2$ instead of sitting on it. When the fractional part is small, $\ell$ is close to $L/\sqrt{2}$ and the target asks for about as many atoms as the loading provides, which is exactly when the rejection rate is worst.
    """)
    return
@app.cell
def floor_effect_plot(np, plt, sqrt):
    _sizes = np.arange(4, 101, 1)
    _target_sizes = np.floor(_sizes / sqrt(2))

    fig_floor, (ax_fractional, ax_ratio) = plt.subplots(
        1, 2, figsize=(11, 4)
    )
    ax_fractional.plot(_sizes, _sizes / sqrt(2) - _target_sizes, marker=".")
    ax_fractional.set_xlabel("Loading array size $L$")
    ax_fractional.set_ylabel(r"$L/\sqrt{2} - \ell$")
    ax_fractional.set_title("Fractional part lost to the floor")
    ax_fractional.grid(alpha=0.3)

    ax_ratio.plot(_sizes, (_target_sizes / _sizes) ** 2, marker=".")
    ax_ratio.axhline(0.5, color="tab:red", ls="--", label="1/2")
    ax_ratio.set_xlabel("Loading array size $L$")
    ax_ratio.set_ylabel(r"$(\ell/L)^2$")
    ax_ratio.set_title("Contraction ratio, no margin")
    ax_ratio.grid(alpha=0.3)
    ax_ratio.legend()

    fig_floor.tight_layout()
    fig_floor
    return
@app.cell(hide_code=True)
def references(mo):
    mo.md(r"""
    ## References

    1. Wang, S., Zhang, W., Zhang, T., Mei, S., Wang, Y., Hu, J., & Chen, W. (2023). *Accelerating the Assembly of Defect-Free Atomic Arrays with Maximum Parallelisms*. **Physical Review Applied, 19**(5), 054032. https://doi.org/10.1103/PhysRevApplied.19.054032
    2. Schlosser, N., Reymond, G., & Grangier, P. (2002). *Collisional Blockade in Microscopic Optical Dipole Traps*. **Physical Review Letters, 89**(2), 023005. https://doi.org/10.1103/PhysRevLett.89.023005
    3. Schlosser, N., Reymond, G., Protsenko, I., et al. (2001). *Sub-poissonian loading of single atoms in a microscopic dipole trap*. **Nature, 411**, 1024–1027. https://doi.org/10.1038/35082512
    """)
    return


if __name__ == "__main__":
    app.run()
