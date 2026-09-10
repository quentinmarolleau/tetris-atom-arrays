"""Parallel displacement counting, the quantity Wang et al. 2023 report."""

import numpy as np
import pytest

from tetris import AtomsConfiguration
from tetris.fast import (
    configuration_kept,
    parallel_displacements,
    target_geometry,
)

FULL_MARGIN = (-1, -1)


def test_already_dense_configuration_needs_no_motion() -> None:
    """A loading array whose target window is already full and whose
    other sites are empty needs no horizontal and no vertical move."""
    window = target_geometry(12, FULL_MARGIN)
    matrix = np.zeros((12, 12), dtype=bool)
    matrix[
        window.row_start : window.row_start + window.rows,
        window.column_start : window.column_start + window.columns,
    ] = True
    config = AtomsConfiguration(
        occupation_matrix=matrix, reshape_target=FULL_MARGIN
    )
    config.construct_tetriminoes()
    assert config.configuration_kept
    assert config.count_parallel_displacements() == 0


def test_single_column_shift_is_counted_once() -> None:
    """One atom per row, all one site left of the target window: every
    row move displaces by one, no column move is needed."""
    window = target_geometry(6)
    matrix = np.zeros((6, 6), dtype=bool)
    start = window.column_start
    matrix[:, start - 1 : start - 1 + window.columns] = True
    config = AtomsConfiguration(occupation_matrix=matrix)
    config.construct_tetriminoes()
    assert config.configuration_kept
    # six rows each shifted by one site, then no vertical compression
    assert config.count_parallel_displacements() == 6


def test_fast_path_matches_the_class() -> None:
    rng = np.random.default_rng(31337)
    checked = 0
    for size in range(6, 51, 2):
        for _ in range(12):
            matrix = rng.integers(0, 2, size=(size, size)).astype(bool)
            config = AtomsConfiguration(
                occupation_matrix=matrix, reshape_target=FULL_MARGIN
            )
            config.construct_tetriminoes()
            if not config.configuration_kept:
                continue
            checked += 1
            assert config.count_parallel_displacements() == (
                parallel_displacements(matrix, config.window)
            ), size
    assert checked > 100


@pytest.mark.parametrize("shape", [(9, 11), (11, 9), (8, 13), (7, 14)])
def test_fast_path_matches_the_class_on_a_rectangular_target(shape) -> None:
    """The row moves span the window's columns and the column moves span
    its rows. Those are the same number for a square target, so only a
    rectangular one can tell a swap of the two apart."""
    rng = np.random.default_rng(80808)
    checked = 0
    for _ in range(4000):
        matrix = rng.integers(0, 2, size=(14, 14)).astype(bool)
        config = AtomsConfiguration(
            occupation_matrix=matrix, target_shape=shape
        )
        config.construct_tetriminoes()
        if not config.configuration_kept:
            continue
        checked += 1
        assert config.count_parallel_displacements() == (
            parallel_displacements(matrix, config.window)
        ), shape
    assert checked > 50, (shape, checked)


def test_displacements_are_only_defined_for_kept_configurations() -> None:
    matrix = np.zeros((10, 10), dtype=bool)
    config = AtomsConfiguration(
        occupation_matrix=matrix, reshape_target=FULL_MARGIN
    )
    config.construct_tetriminoes()
    assert not config.configuration_kept
    with pytest.raises(ValueError):
        config.count_parallel_displacements()
    assert not configuration_kept(matrix, config.window)


def test_scaling_stays_close_to_linear_in_atom_number() -> None:
    """The paper fits N^1.03(7) for compact geometry. A coarse check that
    the implementation is in that regime rather than, say, quadratic."""
    rng = np.random.default_rng(5)
    atom_numbers, means = [], []
    for size in (20, 40, 80):
        window = target_geometry(size, FULL_MARGIN)
        samples = []
        while len(samples) < 60:
            matrix = rng.integers(0, 2, size=(size, size)).astype(bool)
            if configuration_kept(matrix, window):
                samples.append(parallel_displacements(matrix, window))
        atom_numbers.append(window.sites)
        means.append(float(np.mean(samples)))
    slope = np.polyfit(np.log(atom_numbers), np.log(means), 1)[0]
    assert 0.7 < slope < 1.4, (atom_numbers, means, slope)
