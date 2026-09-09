"""Correctness of the tetrimino construction and of the acceptance rule."""

import numpy as np
import pytest

from tetris import AtomsConfiguration
from tetris.fast import configuration_kept


def random_matrix(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.integers(0, 2, size=(size, size)).astype(bool)


def paper_rule_kept(matrix: np.ndarray, target_size: int) -> bool:
    """Direct transcription of the assignment rule of Wang et al. 2023:
    each row's atoms are assigned to the target columns whose smallest
    unassigned target row index is lowest, i.e. the least filled ones."""
    counts = np.zeros(target_size, dtype=np.int64)
    for atoms_in_row in matrix.sum(axis=1).tolist():
        if atoms_in_row >= target_size:
            counts += 1
        else:
            least_filled = np.argsort(counts, kind="stable")[:atoms_in_row]
            counts[least_filled] += 1
    return bool(counts.min() >= target_size)


@pytest.mark.parametrize("margin", [False, True])
def test_target_window_is_centred(margin: bool) -> None:
    for size in range(4, 101):
        config = AtomsConfiguration(loading_array_size=size, margin=margin)
        left = config.target_start
        right = size - (config.target_start + config.target_size)
        assert abs(left - right) <= 1, (size, margin, left, right)


@pytest.mark.parametrize("margin", [False, True])
def test_full_row_covers_every_target_column(margin: bool) -> None:
    """A row holding exactly target_size atoms must reach every target
    column: it is the boundary case that the off-centre target window
    used to break."""
    for size in range(4, 61):
        probe = AtomsConfiguration(loading_array_size=size, margin=margin)
        target_size, target_start = probe.target_size, probe.target_start
        if target_size < 1:
            continue
        for offset in range(size - target_size + 1):
            row = np.zeros(size, dtype=bool)
            row[offset : offset + target_size] = True
            matrix = np.tile(row, (size, 1))
            config = AtomsConfiguration(
                occupation_matrix=matrix, margin=margin
            )
            config.construct_tetriminoes()
            window = config.tetriminoes_matrix[
                :, target_start : target_start + target_size
            ]
            assert window.all(), (size, margin, offset)


@pytest.mark.parametrize("margin", [False, True])
def test_matches_the_paper_assignment_rule(margin: bool) -> None:
    rng = np.random.default_rng(20260909)
    for size in range(4, 71):
        for _ in range(20):
            matrix = random_matrix(rng, size)
            config = AtomsConfiguration(
                occupation_matrix=matrix, margin=margin
            )
            config.construct_tetriminoes()
            assert config.configuration_kept == paper_rule_kept(
                matrix, config.target_size
            ), (size, margin)


@pytest.mark.parametrize("margin", [False, True])
def test_fast_path_matches_the_class(margin: bool) -> None:
    rng = np.random.default_rng(4242)
    for size in range(4, 71):
        for _ in range(20):
            matrix = random_matrix(rng, size)
            config = AtomsConfiguration(
                occupation_matrix=matrix, margin=margin
            )
            config.construct_tetriminoes()
            assert config.configuration_kept == configuration_kept(
                matrix, config.target_size
            ), (size, margin)


def test_atoms_are_conserved() -> None:
    rng = np.random.default_rng(1)
    for size in (7, 16, 31):
        for _ in range(20):
            matrix = random_matrix(rng, size)
            config = AtomsConfiguration(occupation_matrix=matrix)
            config.construct_tetriminoes()
            assert config.tetriminoes_matrix.sum() == matrix.sum()


def test_motions_never_cross() -> None:
    """Sources and destinations are paired in ascending order, which is
    what keeps atoms from colliding during a parallel row move."""
    rng = np.random.default_rng(2)
    for size in (9, 24, 40):
        config = AtomsConfiguration(occupation_matrix=random_matrix(rng, size))
        config.construct_tetriminoes()
        for motions in config.tetriminoes_motions:
            sources = [source for source, _ in motions]
            destinations = [destination for _, destination in motions]
            assert sources == sorted(sources)
            assert destinations == sorted(destinations)


def test_rejects_ambiguous_construction_arguments() -> None:
    with pytest.raises(ValueError):
        AtomsConfiguration()
    with pytest.raises(ValueError):
        AtomsConfiguration(loading_array_size=8, occupation_matrix=[[True]])


def test_generator_is_injectable_and_reproducible() -> None:
    first = AtomsConfiguration(
        loading_array_size=12, rng=np.random.default_rng(7)
    )
    second = AtomsConfiguration(
        loading_array_size=12, rng=np.random.default_rng(7)
    )
    third = AtomsConfiguration(
        loading_array_size=12, rng=np.random.default_rng(8)
    )
    assert np.array_equal(first.occupation_matrix, second.occupation_matrix)
    assert not np.array_equal(first.occupation_matrix, third.occupation_matrix)


@pytest.mark.parametrize(
    "size, expected",
    [(8, 1.0000), (30, 0.9995)],
)
def test_margin_success_rate_regression(size: int, expected: float) -> None:
    """Sizes where the target window used to sit half a site off centre.
    The old code reported 0.9133 at L=8 and 0.9905 at L=30."""
    rng = np.random.default_rng(90909)
    probe = AtomsConfiguration(loading_array_size=size, margin=True)
    samples = 4000
    kept = sum(
        configuration_kept(random_matrix(rng, size), probe.target_size)
        for _ in range(samples)
    )
    rate = kept / samples
    # Wilson half-width at three sigma, widened for the small counts
    tolerance = 3 * np.sqrt(max(expected * (1 - expected), 1e-4) / samples)
    assert abs(rate - expected) < tolerance + 0.002, rate


@pytest.mark.parametrize("size, margin", [(1, False), (1, True), (2, True)])
def test_rejects_a_loading_array_with_no_target(size, margin) -> None:
    with pytest.raises(ValueError, match="no target array"):
        AtomsConfiguration(loading_array_size=size, margin=margin)


def test_smallest_usable_arrays_are_accepted() -> None:
    for size, margin in ((2, False), (4, True)):
        config = AtomsConfiguration(loading_array_size=size, margin=margin)
        assert config.target_size >= 1
        config.construct_tetriminoes()
        assert config.configuration_kept in (True, False)
