"""Parallel displacement counting, the quantity Wang et al. 2023 report."""

import numpy as np
import pytest

from tetris import AtomsConfiguration
from tetris.fast import configuration_kept, parallel_displacements


def test_already_dense_configuration_needs_no_motion() -> None:
    """A loading array whose target window is already full and whose
    other sites are empty needs no horizontal and no vertical move."""
    probe = AtomsConfiguration(loading_array_size=12, margin=True)
    start, size = probe.target_start, probe.target_size
    matrix = np.zeros((12, 12), dtype=bool)
    matrix[start : start + size, start : start + size] = True
    config = AtomsConfiguration(occupation_matrix=matrix, margin=True)
    config.construct_tetriminoes()
    assert config.configuration_kept
    assert config.count_parallel_displacements() == 0


def test_single_column_shift_is_counted_once() -> None:
    """One atom per row, all one site left of the target window: every
    row move displaces by one, no column move is needed."""
    probe = AtomsConfiguration(loading_array_size=6, margin=False)
    start, size = probe.target_start, probe.target_size
    matrix = np.zeros((6, 6), dtype=bool)
    for row in range(6):
        matrix[row, :] = False
    matrix[:, start - 1 : start - 1 + size] = True
    config = AtomsConfiguration(occupation_matrix=matrix, margin=False)
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
            config = AtomsConfiguration(occupation_matrix=matrix, margin=True)
            config.construct_tetriminoes()
            if not config.configuration_kept:
                continue
            checked += 1
            assert config.count_parallel_displacements() == (
                parallel_displacements(matrix, config.target_size,
                                       config.target_start)
            ), size
    assert checked > 100


def test_displacements_are_only_defined_for_kept_configurations() -> None:
    matrix = np.zeros((10, 10), dtype=bool)
    config = AtomsConfiguration(occupation_matrix=matrix, margin=True)
    config.construct_tetriminoes()
    assert not config.configuration_kept
    with pytest.raises(ValueError):
        config.count_parallel_displacements()
    assert not configuration_kept(matrix, config.target_size)


def test_scaling_stays_close_to_linear_in_atom_number() -> None:
    """The paper fits N^1.03(7) for compact geometry. A coarse check that
    the implementation is in that regime rather than, say, quadratic."""
    rng = np.random.default_rng(5)
    atom_numbers, means = [], []
    for size in (20, 40, 80):
        probe = AtomsConfiguration(loading_array_size=size, margin=True)
        samples = []
        while len(samples) < 60:
            matrix = rng.integers(0, 2, size=(size, size)).astype(bool)
            if configuration_kept(matrix, probe.target_size):
                samples.append(
                    parallel_displacements(
                        matrix, probe.target_size, probe.target_start
                    )
                )
        atom_numbers.append(probe.target_size**2)
        means.append(float(np.mean(samples)))
    slope = np.polyfit(np.log(atom_numbers), np.log(means), 1)[0]
    assert 0.7 < slope < 1.4, (atom_numbers, means, slope)
