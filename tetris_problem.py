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
    import json
    import multiprocessing as mp
    import os
    from datetime import datetime, timezone
    from math import sqrt
    from pathlib import Path
    from time import monotonic
    from typing import Literal

    # third-party
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.collections import PatchCollection
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Rectangle
    from numpy.typing import NDArray
    from rich.console import Console
    from rich.table import Table

    return (
        Axes,
        Circle,
        Console,
        Figure,
        Literal,
        NDArray,
        PatchCollection,
        Path,
        Rectangle,
        Table,
        datetime,
        json,
        mo,
        monotonic,
        mp,
        np,
        os,
        plt,
        sqrt,
        timezone,
    )


@app.cell
def notebook_constants(Path):
    # general static parameters for the notebook (usually default values)
    DEFAULT_LOADING_ARRAY_SIZE: int = 18

    # cache for the benchmark cell below, so re-running the notebook
    # (e.g. run-all) doesn't retrigger the multi-minute computation
    BENCHMARK_STATS_PATH: Path = Path(__file__).parent / "benchmark_stats.json"
    return BENCHMARK_STATS_PATH, DEFAULT_LOADING_ARRAY_SIZE


@app.cell(hide_code=True)
def atoms_configuration_docs(mo):
    mo.md(r"""
    We implement a `AtomsConfiguration` class in charge of:

    1. Generating a random occupation matrix (probability half for each loading site) – or alternatively loading a manually chosen one.
    2. Simulating the tetris algorithm execution flow on the generated occupation matrix. This is achieved by the `construct_tetriminoes` method which creates several attributes: `tetriminoes_matrix` (the matrix with atoms horizontally compactified), `tetriminoes_motions` (the list of motions to realize so as to prepare the tetrominoes matrix, starting from the loaded matrix).
    3. A `configuration_kept` attribute is also created after the execution of `construct_tetriminoes`, to know if a loaded configuration should be discarded before starting the vertical stacking of the atoms. This happens if – after `construct_tetriminoes` execution – at least one column of the loading array does not contain enough atoms to fill the corresponding column of the target array.
    4. Plotting the loaded and tetriminoes configurations, with the `plot_configuration` method.
    """)
    return


@app.cell
def atoms_configuration_class(
    Axes,
    Circle,
    Console,
    Figure,
    Literal,
    NDArray,
    PatchCollection,
    Rectangle,
    Table,
    np,
    plt,
):
    type OccupationMatrix = NDArray[np.bool_] | list[list[bool]]

    class AtomsConfiguration:
        """Occupation of a square loading array by atoms, with a target
        array inscribed diagonally (side = loading_array_size / sqrt(2))."""

        loading_array_size: int
        occupation_matrix: np.ndarray
        target_size: int
        target_start: int
        tetriminoes_matrix: np.ndarray | None
        tetriminoes_motions: list[list[tuple[int, int]]] | None
        configuration_kept: bool | None

        def __init__(
            self,
            loading_array_size: int | None = None,
            occupation_matrix: OccupationMatrix | None = None,
            margin: bool = True,
        ):
            match (loading_array_size, occupation_matrix):
                case (int() as size, None):
                    matrix = self._generate_occupation_matrix(size)

                case (None, np.ndarray() as matrix) if matrix.dtype == bool:
                    pass

                case (None, list() as matrix):
                    matrix = np.asarray(matrix, dtype=bool)

                case (int(), np.ndarray() | list()):
                    raise ValueError(
                        "loading_array_size and occupation_matrix are mutually exclusive"
                    )

                case (None, None):
                    raise ValueError(
                        "Either loading_array_size or occupation_matrix must be provided"
                    )

                case _:
                    raise TypeError(
                        "occupation_matrix must be a boolean NumPy array "
                        "or a list of lists of booleans"
                    )

            self.occupation_matrix = matrix
            self.loading_array_size = matrix.shape[0]
            self.margin = margin

            self.target_size = int(self.loading_array_size / np.sqrt(2))

            self.target_start = (
                self.loading_array_size - self.target_size
            ) // 2
            if margin:
                self.target_size -= 1
            self.contraction_ratio = (
                self.target_size / self.loading_array_size
            ) ** 2
            # Populated by construct_tetriminoes()
            self.tetriminoes_matrix = None
            self.tetriminoes_motions = None
            self.configuration_kept = None

        @staticmethod
        def _generate_occupation_matrix(loading_array_size: int) -> np.ndarray:
            return np.random.randint(
                low=0, high=2, size=(loading_array_size, loading_array_size)
            ).astype(bool)

        @staticmethod
        def _format_row(
            row: np.ndarray, target_start: int, target_size: int
        ) -> str:
            """Render a boolean row as Rich markup: a bullet per atom
            (green inside the target region, yellow outside — matching
            plot_configuration's colors), a dot per empty site, and '|'
            markers delimiting the target region."""
            target_end = target_start + target_size
            symbols = []
            for i, occupied in enumerate(row):
                if not occupied:
                    symbols.append("[dim]·[/dim]")
                elif target_start <= i < target_end:
                    symbols.append("[green]●[/green]")
                else:
                    symbols.append("[yellow]●[/yellow]")
            symbols.insert(target_end, "[bold cyan]|[/bold cyan]")
            symbols.insert(target_start, "[bold cyan]|[/bold cyan]")
            return "  ".join(symbols)

        @staticmethod
        def _format_motions(motions: list[tuple[int, int]]) -> str:
            """Render (source, destination) column motions as Rich markup
            arrows, skipping atoms that didn't actually move
            (source == destination)."""
            moved = [
                (source, destination)
                for source, destination in motions
                if source != destination
            ]
            if not moved:
                return "[dim]—[/dim]"
            return "  ".join(
                f"[cyan]{source}[/cyan][dim]→[/dim][magenta]{destination}[/magenta]"
                for source, destination in moved
            )

        @staticmethod
        def _pack_row(
            loaded_row: np.ndarray,
            shift: int,
            loading_array_size: int,
            target_size: int,
            target_start: int,
        ) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
            """Rearrange one row's atoms into a contiguous block (tetrimino
            construction). Returns the corrected row, the shift to carry
            into the next row, and the list of (source, destination)
            column motions for that row's atoms."""
            n_row = int(loaded_row.sum())
            loaded_atoms_positions = np.flatnonzero(loaded_row)

            if n_row < target_size:
                # Pack the n_row atoms into a contiguous block within a
                # target_size-wide window. `shift`, carried over from the
                # previous row, offsets that block to build a staircase
                # (tetrimino) pattern across rows; embed the window in the
                # full-width row at target_start.
                window = np.roll(
                    np.pad(
                        np.ones(n_row, dtype=bool), (0, target_size - n_row)
                    ),
                    shift=shift,
                )
                next_shift = (n_row + shift) % target_size
                row = np.roll(
                    np.pad(window, (0, loading_array_size - target_size)),
                    shift=target_start,
                )
            else:
                # More atoms than the target needs: just center them in the
                # full row. `shift` is intentionally left untouched — a row
                # with enough atoms doesn't hand off a staircase offset to
                # the next row.
                row = np.roll(
                    np.pad(
                        np.ones(n_row, dtype=bool),
                        (0, loading_array_size - n_row),
                    ),
                    shift=(loading_array_size - n_row) // 2,
                )
                next_shift = shift
            final_atoms_positions = np.flatnonzero(row)
            motions = [
                (int(source), int(destination))
                for source, destination in zip(
                    loaded_atoms_positions, final_atoms_positions
                )
            ]
            return row, next_shift, motions

        @staticmethod
        def _find_deficient_columns(
            matrix: np.ndarray, target_start: int, target_size: int
        ) -> list[tuple[int, int]]:
            """Return (column index, atom count) for each target-region
            column holding fewer than target_size atoms. Column indices
            are relative to the target zone (0 is target_start in the
            full loading array). A configuration is kept only if this
            list is empty."""
            target_columns = matrix[
                :, target_start : target_start + target_size
            ]
            column_counts = target_columns.sum(axis=0)
            deficient = np.flatnonzero(column_counts < target_size)
            return [
                (int(column), int(column_counts[column]))
                for column in deficient
            ]

        def construct_tetriminoes(self, verbose: bool = False) -> None:
            """Tetrimino construction: pack each row's atoms into a
            contiguous block, staircased across rows via a running shift.
            Sets tetriminoes_matrix (the packed occupation),
            tetriminoes_motions (per-row (source, destination) column
            pairs), and configuration_kept (whether every target-region
            column holds enough atoms) on self."""
            shift = 0
            tetriminoes_matrix = np.empty_like(
                self.occupation_matrix, dtype=bool
            )
            tetriminoes_motions: list[list[tuple[int, int]]] = []
            total_atom_count = 0
            total_motion_count = 0

            console = Console(force_terminal=True) if verbose else None

            if console:
                info = Table(show_header=False, box=None, pad_edge=False)
                info.add_column(style="bold")
                info.add_row(
                    "Loading array size", str(self.loading_array_size)
                )
                info.add_row("Target size", str(self.target_size))
                info.add_row("Target start", str(self.target_start))
                console.print(info)

            for idx, loaded_row in enumerate(self.occupation_matrix):
                row, shift, motions = self._pack_row(
                    loaded_row,
                    shift,
                    self.loading_array_size,
                    self.target_size,
                    self.target_start,
                )
                tetriminoes_matrix[idx] = row
                tetriminoes_motions.append(motions)
                row_motion_count = sum(
                    1
                    for source, destination in motions
                    if source != destination
                )
                total_atom_count += int(loaded_row.sum())
                total_motion_count += row_motion_count

                if console:
                    console.rule(
                        f"[bold blue]Row {idx}[/bold blue]", style="blue"
                    )
                    console.print(
                        f"[bold]{'Loaded':<10}[/bold] "
                        f"{self._format_row(loaded_row, self.target_start, self.target_size)}"
                        f"  [dim]({loaded_row.sum()} atoms)[/dim]"
                    )
                    console.print(
                        f"[bold]{'Corrected':<10}[/bold] "
                        f"{self._format_row(row, self.target_start, self.target_size)}"
                        f"  [dim]({row_motion_count} motions)[/dim]"
                    )
                    console.print(
                        f"[bold]{'Motions':<10}[/bold] {self._format_motions(motions)}"
                    )
                    console.print(f"[bold]{'Next shift':<10}[/bold] {shift}")

            deficient_columns = self._find_deficient_columns(
                tetriminoes_matrix, self.target_start, self.target_size
            )
            configuration_kept = not deficient_columns

            if console:
                ratio = (
                    total_motion_count / total_atom_count
                    if total_atom_count
                    else 0.0
                )
                console.rule(style="blue")
                console.print(f"[bold]Total atoms:[/bold] {total_atom_count}")
                console.print(
                    f"[bold]Total motions:[/bold] {total_motion_count}  [dim](ratio: {ratio:.3f})[/dim]"
                )
                status_style = "green" if configuration_kept else "red"
                console.print(
                    f"[bold]Configuration kept:[/bold] [{status_style}]{configuration_kept}[/{status_style}]"
                )
                if not configuration_kept:
                    console.print(
                        f"[bold]Deficient columns:[/bold] "
                        + "  ".join(
                            f"[red]{column}[/red] [dim]({count} atoms)[/dim]"
                            for column, count in deficient_columns
                        )
                    )

            self.tetriminoes_matrix = tetriminoes_matrix
            self.tetriminoes_motions = tetriminoes_motions
            self.configuration_kept = configuration_kept

        def _draw_configuration(
            self,
            ax: Axes,
            matrix: np.ndarray,
            title: str,
            atom_radius: float,
            show_ylabel: bool = True,
        ) -> None:
            """Draw one occupation matrix onto ax, highlighting atoms
            inside/outside the target region."""
            loading_array_size = self.loading_array_size
            target_size = self.target_size
            target_start = self.target_start

            # --------------------------------------------------
            # Occupation matrix
            # --------------------------------------------------

            ax.imshow(matrix, cmap="gray", interpolation="none")

            # --------------------------------------------------
            # Target region
            # --------------------------------------------------

            ax.add_patch(
                Rectangle(
                    (target_start - 0.5, target_start - 0.5),
                    target_size,
                    target_size,
                    fill=False,
                    edgecolor="blue",
                    linestyle="--",
                    linewidth=2,
                )
            )

            # --------------------------------------------------
            # Find occupied sites, split inside/outside the target
            # --------------------------------------------------

            occupied_yx = np.argwhere(matrix)

            target_mask = np.zeros_like(matrix, dtype=bool)
            target_mask[
                target_start : target_start + target_size,
                target_start : target_start + target_size,
            ] = True

            inside = target_mask[occupied_yx[:, 0], occupied_yx[:, 1]]
            atoms_inside = occupied_yx[inside]
            atoms_outside = occupied_yx[~inside]

            # --------------------------------------------------
            # Draw atoms as circles, colored by target membership
            # --------------------------------------------------

            inside_circles = [
                Circle((x, y), atom_radius) for y, x in atoms_inside
            ]
            outside_circles = [
                Circle((x, y), atom_radius) for y, x in atoms_outside
            ]

            ax.add_collection(
                PatchCollection(
                    inside_circles,
                    facecolor="none",
                    edgecolor="green",
                    linewidth=2,
                )
            )
            ax.add_collection(
                PatchCollection(
                    outside_circles,
                    facecolor="none",
                    edgecolor="red",
                    linewidth=2,
                )
            )

            # --------------------------------------------------
            # Axes
            # --------------------------------------------------

            ax.set_title(title)
            ax.set_xlabel("X")
            if show_ylabel:
                ax.set_ylabel("Y")

            ax.set_xlim(-0.5, loading_array_size - 0.5)
            ax.set_ylim(loading_array_size - 0.5, -0.5)
            ax.set_aspect("equal")

            # At most ~10 ticks
            max_ticks = 10
            tick_step = max(1, int(np.ceil(loading_array_size / max_ticks)))
            ticks = np.arange(0, loading_array_size, tick_step)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

        def plot_configuration(
            self,
            which: Literal["loading", "tetriminoes", "all"] = "loading",
            atom_radius: float = 0.32,
            size: float = 5.0,
        ) -> Figure:
            """Plot the loading array (`which="loading"`), the
            tetrimino-packed array (`which="tetriminoes"`), or both side
            by side sharing the Y axis (`which="all"`), highlighting
            atoms inside/outside the target region. Runs
            construct_tetriminoes() first if it hasn't been called yet
            and a tetriminoes plot is requested. `size` sizes one
            plot; in `which="all"` mode the width is doubled to fit the
            pair, height unchanged."""
            if (
                which in ("tetriminoes", "all")
                and self.tetriminoes_matrix is None
            ):
                self.construct_tetriminoes()

            if which == "all":
                fig, (loading_ax, tetriminoes_ax) = plt.subplots(
                    1, 2, figsize=(2 * size, size), sharey=True
                )
                self._draw_configuration(
                    loading_ax,
                    self.occupation_matrix,
                    "Initial atom configuration",
                    atom_radius,
                )
                self._draw_configuration(
                    tetriminoes_ax,
                    self.tetriminoes_matrix,
                    "Tetrimino-packed configuration",
                    atom_radius,
                    show_ylabel=False,
                )
            else:
                matrix = (
                    self.tetriminoes_matrix
                    if which == "tetriminoes"
                    else self.occupation_matrix
                )
                title = (
                    "Tetrimino-packed configuration"
                    if which == "tetriminoes"
                    else "Initial atom configuration"
                )
                fig, ax = plt.subplots(figsize=(size, size))
                self._draw_configuration(ax, matrix, title, atom_radius)

            plt.tight_layout()
            return fig

    return (AtomsConfiguration,)


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

    ### Computation

    We take $L \in [4, 100]$, and $\ell$ with and without securtity margin. For each $(L, \ell)$ configuration, we execute runs for 60s, to build a statistical sample. The computation is executed in parallel (using the `multiprocessing` package) with one worker per $(L, \ell)$ configuration.
    """)
    return


@app.cell
def benchmark_rejection_rate(
    AtomsConfiguration,
    BENCHMARK_STATS_PATH: "Path",
    datetime,
    json,
    monotonic,
    mp,
    np,
    os,
    timezone,
):
    if BENCHMARK_STATS_PATH.exists():
        # skip the multi-minute computation on a plain re-run (e.g.
        # run-all) — delete the cache file to force a recompute
        with open(BENCHMARK_STATS_PATH) as f:
            cache = json.load(f)
        stats = {
            int(size): variants for size, variants in cache["results"].items()
        }
    else:
        MIN_DURATION_PER_SIZE_S = 60
        N_WORKERS = max(1, (os.cpu_count() or 1) - 1)

        sizes = np.arange(start=4, stop=101, step=1)
        margin_variants = (False, True)

        # one task per (loading_array_size, margin) combo — each gets
        # its own worker and its own MIN_DURATION_PER_SIZE_S budget
        tasks = [
            (int(size), margin) for size in sizes for margin in margin_variants
        ]

        def _collect_configuration_kept(
            loading_array_size: int,
            margin: bool,
            deadline: float,
            result_queue: mp.Queue,
        ) -> None:
            # runs in a worker process (fork): loops on a single
            # (size, margin) combo until the deadline, then reports the
            # booleans plus the combo's derived quantities back
            start = monotonic()
            config = AtomsConfiguration(
                loading_array_size=loading_array_size, margin=margin
            )
            target_size = config.target_size
            contraction_ratio = config.contraction_ratio
            config.construct_tetriminoes()
            results = [config.configuration_kept]
            while monotonic() < deadline:
                config = AtomsConfiguration(
                    loading_array_size=loading_array_size, margin=margin
                )
                config.construct_tetriminoes()
                results.append(config.configuration_kept)
            elapsed = monotonic() - start
            result_queue.put(
                (
                    loading_array_size,
                    margin,
                    results,
                    target_size,
                    contraction_ratio,
                    elapsed,
                )
            )

        mp_ctx = mp.get_context("fork")

        # every combo takes the same fixed duration, so instead of
        # piling several workers on one combo at a time, each worker
        # takes a whole combo for itself and N_WORKERS combos run side
        # by side per batch
        stats = dict()
        for batch_start in range(0, len(tasks), N_WORKERS):
            batch = tasks[batch_start : batch_start + N_WORKERS]
            deadline = monotonic() + MIN_DURATION_PER_SIZE_S
            result_queue = mp_ctx.Queue()
            workers = [
                mp_ctx.Process(
                    target=_collect_configuration_kept,
                    args=(size, margin, deadline, result_queue),
                )
                for size, margin in batch
            ]
            for worker in workers:
                worker.start()

            for _ in workers:
                (
                    size,
                    margin,
                    results,
                    target_size,
                    contraction_ratio,
                    _elapsed,
                ) = result_queue.get()
                results = np.array(results)
                _variant = "with_margin" if margin else "no_margin"
                stats.setdefault(size, {})[_variant] = {
                    "success_rate": float(results.mean()),
                    "std": float(results.std()),
                    "N": int(results.size),
                    "contraction_ratio": float(contraction_ratio),
                    "target_number_of_atoms": int(target_size**2),
                    "number_of_loading_sites": int(size**2),
                }
            for worker in workers:
                worker.join()

        # every combo runs against the same fixed budget, so a single
        # value covers all of them instead of one entry per combo
        stats = dict(sorted(stats.items()))
        cache = {
            "meta": {
                "date": datetime.now(timezone.utc).isoformat(),
                "computation_time_per_size": MIN_DURATION_PER_SIZE_S,
            },
            "results": {
                str(size): variants for size, variants in stats.items()
            },
        }
        with open(BENCHMARK_STATS_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    return (stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting the results
    """)
    return


@app.cell
def _(np, stats):
    # Generic quantities and numpy arrays that shall be used in several cells of what follows
    sizes_arr = np.array(sorted(stats.keys()))
    return (sizes_arr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Computation time and sample size $N_s$

    Everything implemented above is Python and not heavily optimized. Hence I do no expect blazing fast computation, but we must anyway check the sample size – that we will write $N_s$ – obtained for each configuration, after one minute of calculation. It would be also interesting to have an insight of the scaling law, for the per row processing time, as a function of the array size.
    """)
    return


@app.cell
def _(np, plt, sizes_arr, stats):
    fig_sample_size, ax_sample_size = plt.subplots(figsize=(6, 4))
    _all_log_sizes = []
    _all_log_samples = []
    for _variant, _label, _marker in (
        ("no_margin", "No margin", "o"),
        ("with_margin", "With margin", "s"),
    ):
        _samples = np.array([stats[size][_variant]["N"] for size in sizes_arr])

        ax_sample_size.semilogy(
            sizes_arr,
            _samples,
            marker=_marker,
            label=_label,
            ls="None",
            alpha=0.75,
        )

        _all_log_sizes.append(np.log(sizes_arr))
        _all_log_samples.append(np.log(_samples))

    # single fit N = alpha * L^(-beta) across both variants' data, linear
    # in log-log: log(N) = log(alpha) - beta*log(L)
    _log_sizes = np.concatenate(_all_log_sizes)
    _log_samples = np.concatenate(_all_log_samples)
    (_slope, _intercept), _cov = np.polyfit(_log_sizes, _log_samples, 1, cov=True)
    _beta = -_slope
    _beta_std = np.sqrt(_cov[0, 0])
    _alpha = np.exp(_intercept)
    ax_sample_size.semilogy(
        sizes_arr,
        _alpha * sizes_arr ** (-_beta),
        label=rf"Fit: $\beta$={_beta:.2f} ({_beta_std:.2f})",
        # ls="--",
    )

    ax_sample_size.semilogy(
        sizes_arr,
        2 * 10**5 * sizes_arr ** (-0.8),
        label="Paper scaling",
        color="tab:red",
    )

    ax_sample_size.set_xlabel("Loading array size $L$")
    ax_sample_size.set_ylabel("Sample size $N_s$")
    ax_sample_size.set_title("Sample sizes after 1 min of computation")
    ax_sample_size.grid()
    ax_sample_size.legend()

    fig_sample_size.tight_layout()
    fig_sample_size
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is not easy to infer a scaling law at this stage. In [[1]](https://doi.org/10.1103/PhysRevApplied.19.054032) the authors claim a fitted rearrangement complexety of $N^{1.6}$ ($N$ being the number of atoms), hence in our case $L^{0.8}$.

    With a naive back of the enveloppe reasoning, we could think:
    \[
        \text{Rearrangement complexity} \propto {N_s}^{-1}
    \]
    maybe this reasoning is naive, or the authors have a better implementation of the algorithm than the one I made above, but we do not retrieve that scaling here. Our own fit gives
    \[
        N_s \propto N^{-2.18(0.02)}
    \]

    For the computation time per row, with 1 minute of computation for each configuration, we obtain the following:
    """)
    return


@app.cell
def _(np, plt, sizes_arr, stats):
    fig_time_per_row, ax_time_per_row = plt.subplots(figsize=(6, 4))
    for _variant, _label, _marker in (
        ("no_margin", "No margin", "o"),
        ("with_margin", "With margin", "s"),
    ):
        _samples = np.array([stats[size][_variant]["N"] for size in sizes_arr])
        # 1 min of computation per configuration, spread over N
        # configurations of L rows each
        _time_per_row_us = 60e6 / (_samples * sizes_arr)

        ax_time_per_row.plot(
            sizes_arr,
            _time_per_row_us,
            marker=_marker,
            label=_label,
            ls="None",
            alpha=0.75,
        )

    ax_time_per_row.set_xlabel("Loading array size $L$")
    ax_time_per_row.set_ylabel("Computation time per row (µs)")
    ax_time_per_row.set_title("Estimated per-row computation time")
    ax_time_per_row.legend()

    fig_time_per_row.tight_layout()
    fig_time_per_row
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Success rate

    We can start by brutally plotting the raw results, separating the cases with and without the security margin.
    """)
    return


@app.cell
def success_rate_plot(np, plt, stats):
    _sizes_arr = np.array(sorted(stats.keys()))

    fig_success_rate, ax_success_rate = plt.subplots(figsize=(10, 4))
    for _variant, _label, _marker in (
        ("no_margin", "No margin", "o"),
        ("with_margin", "With margin", "s"),
    ):
        _means = np.array(
            [stats[size][_variant]["success_rate"] for size in _sizes_arr]
        )
        _stds = np.array([stats[size][_variant]["std"] for size in _sizes_arr])
        ax_success_rate.errorbar(
            _sizes_arr,
            _means,
            yerr=_stds,
            fmt=_marker,
            capsize=3,
            label=_label,
        )
    ax_success_rate.set_xlabel("Loading array size")
    ax_success_rate.set_ylabel("Success rate")
    ax_success_rate.set_title("Success rate vs. loading array size")
    ax_success_rate.legend()

    fig_success_rate.tight_layout()
    fig_success_rate
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Without security margin, the sucess rate of the procedure can be really bad, exhibiting structures that we can easily imagine coming from the truncation happening when we convert the loading size into the target size (effect of the `floor` function).

    The behaviour seems more reasonable
    """)
    return


@app.cell
def success_rate_vs_contraction_ratio_plot(np, plt, stats):
    fig_success_vs_ratio, ax_success_vs_ratio = plt.subplots(figsize=(10, 4))
    for _variant, _label, _marker in (
        ("no_margin", "No margin", "o"),
        ("with_margin", "With margin", "s"),
    ):
        _entries = sorted(
            (
                stats[size][_variant]["contraction_ratio"],
                stats[size][_variant]["success_rate"],
                stats[size][_variant]["std"],
            )
            for size in stats
        )
        _ratios, _means, _stds = (np.array(col) for col in zip(*_entries))
        ax_success_vs_ratio.errorbar(
            _ratios, _means, yerr=_stds, fmt=_marker, capsize=3, label=_label
        )
    ax_success_vs_ratio.set_xlabel("Contraction ratio")
    ax_success_vs_ratio.set_ylabel("Success rate")
    ax_success_vs_ratio.set_title("Success rate vs. contraction ratio")
    ax_success_vs_ratio.legend()
    fig_success_vs_ratio
    return


@app.cell
def success_rate_heatmap(np, plt, stats):
    # (loading array size, contraction ratio) isn't a regular grid — each
    # size only has two ratios (no_margin/with_margin) — so a scatter
    # colored by success rate stands in for an imshow-style heatmap
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "purple_to_green", ["tab:purple", "tab:red", "tab:green"]
    )

    _sizes = np.array(
        [size for size in stats for _ in ("no_margin", "with_margin")]
    )
    _ratios = np.array(
        [
            stats[size][_variant]["contraction_ratio"]
            for size in stats
            for _variant in ("no_margin", "with_margin")
        ]
    )
    _success_rates = np.array(
        [
            stats[size][_variant]["success_rate"]
            for size in stats
            for _variant in ("no_margin", "with_margin")
        ]
    )
    _stds = np.array(
        [
            stats[size][_variant]["std"]
            for size in stats
            for _variant in ("no_margin", "with_margin")
        ]
    )
    # normalized standard deviation, in percent of the success rate
    _relative_stds = 100 * _stds / _success_rates

    fig_heatmap, (ax_heatmap, ax_heatmap_std) = plt.subplots(
        1, 2, figsize=(18, 6)
    )

    _scatter = ax_heatmap.scatter(
        _sizes,
        _ratios,
        c=_success_rates,
        cmap=cmap,
        vmin=0.5,
        vmax=1,
        s=40,
    )
    fig_heatmap.colorbar(_scatter, ax=ax_heatmap, label="Success rate")
    ax_heatmap.set_xlabel("Loading array size")
    ax_heatmap.set_ylabel("Contraction ratio")
    ax_heatmap.set_title("Success rate heatmap")

    _scatter_std = ax_heatmap_std.scatter(
        _sizes,
        _ratios,
        c=_relative_stds,
        cmap="viridis",
        s=40,
    )
    fig_heatmap.colorbar(
        _scatter_std, ax=ax_heatmap_std, label=r"$\sigma$ / success rate (%)"
    )
    ax_heatmap_std.set_xlabel("Loading array size")
    ax_heatmap_std.set_ylabel("Contraction ratio")
    ax_heatmap_std.set_title("Normalized standard deviation heatmap")

    fig_heatmap
    return


@app.cell
def _(np, plt, sqrt):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    sizes_sample = np.arange(4, 51, 1)
    target_sizes = np.floor(sizes_sample / sqrt(2))
    fractional_part = sizes_sample / sqrt(2) - target_sizes

    axs[0].plot(
        sizes_sample,
        fractional_part,
        marker=".",
    )

    axs[1].plot(
        sizes_sample,
        (target_sizes / sizes_sample) ** 2,
        marker=".",
    )
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
