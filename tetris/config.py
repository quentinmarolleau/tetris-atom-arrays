"""Occupation of a square loading array by atoms, and the tetrimino
construction of Wang et al. 2023 that packs it towards a target array
inscribed diagonally in it."""

from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from rich.console import Console

from tetris import display, plotting
from tetris.fast import parallel_displacements, target_geometry

__all__ = ["AtomsConfiguration", "OccupationMatrix"]

type OccupationMatrix = NDArray[np.bool_] | list[list[bool]]


class AtomsConfiguration:
    """A loaded configuration and the rearrangement strategy for it.

    The target array is a square of side `loading_array_size / sqrt(2)`,
    centred in the loading array, so that it holds about half the sites
    and therefore about as many sites as the 50 percent loading provides
    atoms. With `margin`, one site is taken off that side first, which
    is the convention Wang et al. use in their own simulations.
    """

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
        rng: np.random.Generator | None = None,
    ):
        self.rng = rng if rng is not None else np.random.default_rng()

        match (loading_array_size, occupation_matrix):
            case (int() as size, None):
                matrix = self._generate_occupation_matrix(size)

            case (None, np.ndarray() as matrix) if matrix.dtype == bool:
                pass

            case (None, list() as matrix):
                matrix = np.asarray(matrix, dtype=bool)

            case (int(), np.ndarray() | list()):
                raise ValueError(
                    "loading_array_size and occupation_matrix are "
                    "mutually exclusive"
                )

            case (None, None):
                raise ValueError(
                    "Either loading_array_size or occupation_matrix "
                    "must be provided"
                )

            case _:
                raise TypeError(
                    "occupation_matrix must be a boolean NumPy array "
                    "or a list of lists of booleans"
                )

        self.occupation_matrix = matrix
        self.loading_array_size = matrix.shape[0]
        self.margin = margin
        self.target_size, self.target_start = target_geometry(
            self.loading_array_size, margin
        )
        self.contraction_ratio = (
            self.target_size / self.loading_array_size
        ) ** 2

        # Populated by construct_tetriminoes()
        self.tetriminoes_matrix = None
        self.tetriminoes_motions = None
        self.configuration_kept = None

    def _generate_occupation_matrix(self, size: int) -> np.ndarray:
        return self.rng.integers(low=0, high=2, size=(size, size)).astype(bool)

    @staticmethod
    def _pack_row(
        loaded_row: np.ndarray,
        shift: int,
        target_size: int,
        target_start: int,
    ) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
        """Rearrange one row's atoms towards the target columns. Returns
        the packed row, the shift to carry into the next one, and the
        (source, destination) column pairs for the atoms that move.

        Sources and destinations are both in ascending order, so the
        atoms never cross and can all be moved in a single parallel
        step."""
        positions = np.flatnonzero(loaded_row)
        atoms_in_row = positions.size
        packed = np.zeros_like(loaded_row)

        if atoms_in_row < target_size:
            # Deal the atoms round the target window from the running
            # shift, which spreads successive rows over the columns that
            # are furthest behind. Wrapping is deliberate: it is what
            # makes the staircase of tetriminoes.
            columns = target_start + (shift + np.arange(atoms_in_row)) % (
                target_size
            )
            packed[columns] = True
            destinations = np.sort(columns)
            moving = positions
            next_shift = (shift + atoms_in_row) % target_size
        else:
            # The row already has enough atoms to serve every target
            # column. Fill the window and leave the surplus atoms where
            # they are rather than dragging them along; a row can only
            # ever give one atom per column, so nothing is lost.
            destinations = target_start + np.arange(target_size)
            first = min(
                max(int(np.searchsorted(positions, target_start)), 0),
                atoms_in_row - target_size,
            )
            moving = positions[first : first + target_size]
            packed[destinations] = True
            packed[positions[:first]] = True
            packed[positions[first + target_size :]] = True
            # a row that serves every column hands on no staircase offset
            next_shift = shift

        motions = [
            (int(source), int(destination))
            for source, destination in zip(moving, destinations)
        ]
        return packed, next_shift, motions

    @staticmethod
    def _find_deficient_columns(
        matrix: np.ndarray, target_start: int, target_size: int
    ) -> list[tuple[int, int]]:
        """(column index, atom count) for each target column holding
        fewer than target_size atoms. Indices are relative to the target
        window. A configuration is kept only if this list is empty."""
        target_columns = matrix[:, target_start : target_start + target_size]
        column_counts = target_columns.sum(axis=0)
        deficient = np.flatnonzero(column_counts < target_size)
        return [
            (int(column), int(column_counts[column])) for column in deficient
        ]

    def construct_tetriminoes(self, verbose: bool = False) -> None:
        """Pack each row's atoms towards the target columns, staircased
        across rows by a running shift. Sets `tetriminoes_matrix`,
        `tetriminoes_motions` and `configuration_kept`."""
        shift = 0
        tetriminoes_matrix = np.empty_like(self.occupation_matrix, dtype=bool)
        tetriminoes_motions: list[list[tuple[int, int]]] = []
        total_atoms = 0
        total_motions = 0
        console = Console(force_terminal=True) if verbose else None

        if console:
            display.print_header(
                console,
                self.loading_array_size,
                self.target_size,
                self.target_start,
            )

        for index, loaded_row in enumerate(self.occupation_matrix):
            packed, shift, motions = self._pack_row(
                loaded_row, shift, self.target_size, self.target_start
            )
            tetriminoes_matrix[index] = packed
            tetriminoes_motions.append(motions)
            total_atoms += int(loaded_row.sum())
            if console:
                total_motions += sum(
                    1
                    for source, destination in motions
                    if source != destination
                )
                display.print_row(
                    console,
                    index,
                    loaded_row,
                    packed,
                    motions,
                    shift,
                    self.target_start,
                    self.target_size,
                )

        deficient_columns = self._find_deficient_columns(
            tetriminoes_matrix, self.target_start, self.target_size
        )
        configuration_kept = not deficient_columns

        if console:
            display.print_summary(
                console,
                total_atoms,
                total_motions,
                configuration_kept,
                deficient_columns,
            )

        self.tetriminoes_matrix = tetriminoes_matrix
        self.tetriminoes_motions = tetriminoes_motions
        self.configuration_kept = configuration_kept

    def count_parallel_displacements(self) -> int:
        """Parallel displacements needed to reach the target array: the
        sum, over the row moves and then the column moves, of the largest
        single atom displacement in each. This is the cost Wang et al.
        report against atom number, since it is what the rearrangement
        time is dominated by."""
        if self.tetriminoes_matrix is None:
            self.construct_tetriminoes()
        if not self.configuration_kept:
            raise ValueError("configuration would have been discarded")

        total = sum(
            max(
                (abs(source - destination) for source, destination in motions),
                default=0,
            )
            for motions in self.tetriminoes_motions
        )
        window = self.tetriminoes_matrix[
            :, self.target_start : self.target_start + self.target_size
        ]
        destinations = self.target_start + np.arange(self.target_size)
        for column in range(self.target_size):
            rows = np.flatnonzero(window[:, column])
            first = min(
                max(int(np.searchsorted(rows, self.target_start)), 0),
                rows.size - self.target_size,
            )
            moving = rows[first : first + self.target_size]
            total += int(np.abs(moving - destinations).max(initial=0))
        return int(total)

    def plot_configuration(
        self,
        which: Literal["loading", "tetriminoes", "all"] = "loading",
        atom_radius: float = 0.32,
        size: float = 5.0,
    ) -> Figure:
        """Plot the loading array, the packed array, or both side by
        side. Runs `construct_tetriminoes` first if it has not run."""
        return plotting.plot_configuration(self, which, atom_radius, size)


# kept importable next to the class it belongs with
_ = parallel_displacements
