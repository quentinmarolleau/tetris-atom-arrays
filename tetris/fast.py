"""Array-level tetrimino construction, without the bookkeeping the
pedagogical class carries.

`AtomsConfiguration` builds a full occupation matrix, a per-row motion
table and a Rich rendering path for every configuration it examines. A
benchmark examines millions of them and needs none of that, so the two
quantities that the benchmark does need are recomputed here directly
from the loaded matrix. About sixteen times faster at L = 30 and
twenty-five times at L = 100, the gap widening because the class pays
per site and this pays per row.

The two implementations are held equal by the tests.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["TargetWindow", "configuration_kept", "parallel_displacements",
           "target_geometry"]


@dataclass(frozen=True)
class TargetWindow:
    """Where the target array sits inside the loading array.

    Rows and columns are tracked separately. They are equal for the
    square targets the paper uses, but the two play different parts:
    `columns` is how many atoms a row can place, `rows` is how many
    atoms each column has to supply.
    """

    rows: int
    columns: int
    row_start: int
    column_start: int

    @property
    def sites(self) -> int:
        return self.rows * self.columns


def target_geometry(
    loading_array_size: int,
    reshape: tuple[int, int] = (0, 0),
    shape: tuple[int, int] | None = None,
) -> TargetWindow:
    """The target window inscribed in a loading array of the given size.

    The natural side is `L / sqrt(2)` rounded down, which asks for about
    half the loading sites and so for about as many atoms as a 50 percent
    loading provides. `reshape` adds to that side, one entry per axis, so
    that (0, -1) drops a column and (-1, -1) drops a row and a column,
    the convention Wang et al. use. `shape` names the two sides outright
    instead, and cannot be combined with a non-zero `reshape`.

    The window is centred on each axis independently once its sides are
    settled.
    """
    if loading_array_size < 1:
        raise ValueError(
            f"a loading array needs at least one site, got "
            f"{loading_array_size}"
        )

    if shape is not None:
        if tuple(reshape) != (0, 0):
            raise ValueError(
                "a target shape and a reshape cannot both be given: "
                f"got shape {tuple(shape)} and reshape {tuple(reshape)}"
            )
        rows, columns = (int(side) for side in shape)
        because = "the requested target shape"
    else:
        side = int(loading_array_size / np.sqrt(2))
        rows, columns = (side + int(delta) for delta in reshape)
        because = (
            "the natural target side"
            if tuple(reshape) == (0, 0)
            else f"the natural target side reshaped by {tuple(reshape)}"
        )

    for axis, extent in (("rows", rows), ("columns", columns)):
        if extent < 1:
            raise ValueError(
                f"a loading array of size {loading_array_size} leaves no "
                f"target array to fill: {because} gives {extent} {axis}"
            )
        if extent > loading_array_size:
            raise ValueError(
                f"a target of {rows} rows by {columns} columns does not fit "
                f"in a loading array of size {loading_array_size}: "
                f"{because} gives {extent} {axis}"
            )

    return TargetWindow(
        rows=rows,
        columns=columns,
        row_start=(loading_array_size - rows) // 2,
        column_start=(loading_array_size - columns) // 2,
    )


def _row_columns(atoms_in_row: int, shift: int, columns: int) -> range:
    """Target columns, relative to the window, that a row carrying
    `atoms_in_row` atoms fills. Blocks are dealt round the window like
    cards, which for a uniform target is the same assignment as the
    paper's rule of always serving the least filled columns first."""
    return range(shift, shift + min(atoms_in_row, columns))


def configuration_kept(matrix: np.ndarray, window: TargetWindow) -> bool:
    """Whether every target column ends up holding enough atoms to fill
    its share of the target array, which is the condition to proceed to
    the vertical compression instead of discarding the loading."""
    columns, rows = window.columns, window.rows
    if columns < 1 or rows < 1:
        return False

    # occupancy is accumulated as a difference array over the target
    # columns, so a row costs O(1) instead of O(columns)
    boundaries = np.zeros(columns + 1, dtype=np.int32)
    full_rows = 0
    shift = 0
    for atoms_in_row in matrix.sum(axis=1).tolist():
        if atoms_in_row >= columns:
            full_rows += 1
            continue
        end = shift + atoms_in_row
        if end <= columns:
            boundaries[shift] += 1
            boundaries[end] -= 1
        else:
            boundaries[shift] += 1
            boundaries[columns] -= 1
            boundaries[0] += 1
            boundaries[end - columns] -= 1
        shift = end % columns

    occupancy = np.cumsum(boundaries[:columns]) + full_rows
    return bool(occupancy.min() >= rows)


def _movers(positions: np.ndarray, window_start: int, count: int) -> int:
    """Index of the first of `count` consecutive atoms to assign to a
    window starting at `window_start`. Picking a consecutive run keeps
    the assignment non-crossing, and starting it at the window keeps the
    atoms that stay behind outside it."""
    first = int(np.searchsorted(positions, window_start))
    return min(max(first, 0), positions.size - count)


def _window_displacement(
    positions: np.ndarray, window_start: int, window_size: int
) -> int:
    """Largest distance any atom travels when `window_size` of the atoms
    at `positions` are compressed into the window. Atoms move in
    parallel, so the largest of them is what the move costs."""
    if positions.size < window_size:
        raise ValueError("not enough atoms to fill the window")
    first = _movers(positions, window_start, window_size)
    moving = positions[first : first + window_size]
    destinations = np.arange(window_start, window_start + window_size)
    return int(np.abs(moving - destinations).max(initial=0))


def parallel_displacements(
    matrix: np.ndarray, window: TargetWindow
) -> int:
    """Number of parallel displacements needed to reach the target
    array, the quantity Wang et al. 2023 report against atom number.

    Every atom in a row moves at once, so a row move costs the largest
    single displacement in it; likewise for the column compression that
    follows. The total is the sum over the L row moves and the
    `window.columns` column moves.

    The row moves are measured against the column extent of the window
    and the column moves against its row extent. Those coincide for a
    square target, which is why one number stood in for both until the
    window was allowed to be rectangular.
    """
    if not configuration_kept(matrix, window):
        raise ValueError("configuration would have been discarded")

    columns, rows = window.columns, window.rows
    loading_array_size = matrix.shape[0]
    # occupancy of the target window only, rows by target columns
    occupancy = np.zeros((loading_array_size, columns), dtype=bool)
    total = 0
    shift = 0
    for index, row in enumerate(matrix):
        positions = np.flatnonzero(row)
        atoms_in_row = positions.size
        filled = np.fromiter(
            (
                (column % columns)
                for column in _row_columns(atoms_in_row, shift, columns)
            ),
            dtype=np.intp,
            count=min(atoms_in_row, columns),
        )
        occupancy[index, filled] = True
        if atoms_in_row >= columns:
            total += _window_displacement(
                positions, window.column_start, columns
            )
        else:
            destinations = window.column_start + np.sort(filled)
            total += int(np.abs(positions - destinations).max(initial=0))
            shift = (shift + atoms_in_row) % columns

    for column in range(columns):
        total += _window_displacement(
            np.flatnonzero(occupancy[:, column]), window.row_start, rows
        )
    return total
