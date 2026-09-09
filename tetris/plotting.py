"""Matplotlib rendering of a loading array and of its tetrimino-packed
counterpart."""

from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

__all__ = ["draw_configuration", "plot_configuration"]

MAX_TICKS = 10


def draw_configuration(
    ax: Axes,
    matrix: np.ndarray,
    title: str,
    target_start: int,
    target_size: int,
    atom_radius: float,
    show_ylabel: bool = True,
) -> None:
    """Draw one occupation matrix, highlighting atoms inside and outside
    the target region."""
    loading_array_size = matrix.shape[0]
    ax.imshow(matrix, cmap="gray", interpolation="none")
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

    occupied_yx = np.argwhere(matrix)
    target_mask = np.zeros_like(matrix, dtype=bool)
    target_mask[
        target_start : target_start + target_size,
        target_start : target_start + target_size,
    ] = True
    inside = target_mask[occupied_yx[:, 0], occupied_yx[:, 1]]

    for atoms, colour in (
        (occupied_yx[inside], "green"),
        (occupied_yx[~inside], "red"),
    ):
        ax.add_collection(
            PatchCollection(
                [Circle((x, y), atom_radius) for y, x in atoms],
                facecolor="none",
                edgecolor=colour,
                linewidth=2,
            )
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    if show_ylabel:
        ax.set_ylabel("Y")
    ax.set_xlim(-0.5, loading_array_size - 0.5)
    ax.set_ylim(loading_array_size - 0.5, -0.5)
    ax.set_aspect("equal")

    tick_step = max(1, int(np.ceil(loading_array_size / MAX_TICKS)))
    ticks = np.arange(0, loading_array_size, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def plot_configuration(
    config,
    which: Literal["loading", "tetriminoes", "all"] = "loading",
    atom_radius: float = 0.32,
    size: float = 5.0,
) -> Figure:
    """Plot the loading array, the packed array, or both side by side
    sharing the Y axis. `size` sizes one panel; in `all` mode the width
    is doubled to fit the pair."""
    if which in ("tetriminoes", "all") and config.tetriminoes_matrix is None:
        config.construct_tetriminoes()

    geometry = (config.target_start, config.target_size)
    if which == "all":
        fig, (loading_ax, packed_ax) = plt.subplots(
            1, 2, figsize=(2 * size, size), sharey=True
        )
        draw_configuration(
            loading_ax,
            config.occupation_matrix,
            "Initial atom configuration",
            *geometry,
            atom_radius,
        )
        draw_configuration(
            packed_ax,
            config.tetriminoes_matrix,
            "Tetrimino-packed configuration",
            *geometry,
            atom_radius,
            show_ylabel=False,
        )
    else:
        packed = which == "tetriminoes"
        fig, ax = plt.subplots(figsize=(size, size))
        draw_configuration(
            ax,
            config.tetriminoes_matrix if packed else config.occupation_matrix,
            (
                "Tetrimino-packed configuration"
                if packed
                else "Initial atom configuration"
            ),
            *geometry,
            atom_radius,
        )

    plt.tight_layout()
    return fig
