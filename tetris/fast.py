"""Array-level tetrimino construction, without the bookkeeping the
pedagogical class carries.

`AtomsConfiguration` builds a full occupation matrix, a per-row motion
table and a Rich rendering path for every configuration it examines. A
benchmark examines millions of them and needs none of that, so the two
quantities that the benchmark does need are recomputed here directly
from the loaded matrix. Roughly fifty times faster at L = 100.

The two implementations are held equal by the tests.
"""

import numpy as np

__all__ = ["configuration_kept", "parallel_displacements", "target_geometry"]


def target_geometry(
    loading_array_size: int, margin: bool = True
) -> tuple[int, int]:
    """Side and first index of the target window inscribed in a loading
    array of the given size. With `margin`, the side is shortened by one
    site before the window is centred."""
    target_size = int(loading_array_size / np.sqrt(2))
    if margin:
        target_size -= 1
    target_start = (loading_array_size - target_size) // 2
    return target_size, target_start


def _row_columns(atoms_in_row: int, shift: int, target_size: int) -> range:
    """Target columns, relative to the window, that a row carrying
    `atoms_in_row` atoms fills. Blocks are dealt round the window like
    cards, which for a uniform target is the same assignment as the
    paper's rule of always serving the least filled columns first."""
    return range(shift, shift + min(atoms_in_row, target_size))


def configuration_kept(matrix: np.ndarray, target_size: int) -> bool:
    """Whether every target column ends up holding enough atoms to fill
    its share of the target array, which is the condition to proceed to
    the vertical compression instead of discarding the loading."""
    if target_size < 1:
        return False

    # occupancy is accumulated as a difference array over the target
    # columns, so a row costs O(1) instead of O(target_size)
    boundaries = np.zeros(target_size + 1, dtype=np.int32)
    full_rows = 0
    shift = 0
    for atoms_in_row in matrix.sum(axis=1).tolist():
        if atoms_in_row >= target_size:
            full_rows += 1
            continue
        end = shift + atoms_in_row
        if end <= target_size:
            boundaries[shift] += 1
            boundaries[end] -= 1
        else:
            boundaries[shift] += 1
            boundaries[target_size] -= 1
            boundaries[0] += 1
            boundaries[end - target_size] -= 1
        shift = end % target_size

    occupancy = np.cumsum(boundaries[:target_size]) + full_rows
    return bool(occupancy.min() >= target_size)


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
    matrix: np.ndarray, target_size: int, target_start: int
) -> int:
    """Number of parallel displacements needed to reach the target
    array, the quantity Wang et al. 2023 report against atom number.

    Every atom in a row moves at once, so a row move costs the largest
    single displacement in it; likewise for the column compression that
    follows. The total is the sum over the L row moves and the
    target_size column moves.
    """
    if not configuration_kept(matrix, target_size):
        raise ValueError("configuration would have been discarded")

    loading_array_size = matrix.shape[0]
    # occupancy of the target window only, rows by target columns
    window = np.zeros((loading_array_size, target_size), dtype=bool)
    total = 0
    shift = 0
    for index, row in enumerate(matrix):
        positions = np.flatnonzero(row)
        atoms_in_row = positions.size
        columns = np.fromiter(
            (
                (column % target_size)
                for column in _row_columns(atoms_in_row, shift, target_size)
            ),
            dtype=np.intp,
            count=min(atoms_in_row, target_size),
        )
        window[index, columns] = True
        if atoms_in_row >= target_size:
            total += _window_displacement(
                positions, target_start, target_size
            )
        else:
            destinations = target_start + np.sort(columns)
            total += int(np.abs(positions - destinations).max(initial=0))
            shift = (shift + atoms_in_row) % target_size

    for column in range(target_size):
        total += _window_displacement(
            np.flatnonzero(window[:, column]), target_start, target_size
        )
    return total
