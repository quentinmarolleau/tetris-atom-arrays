"""The closed forms in `tetris.analytic`, against enumeration, direct
summation and a seeded sample."""

import itertools
from math import comb, isclose

import numpy as np
import pytest

from tetris.analytic import (
    enough_atoms_probability,
    expected_wasted_atoms,
    expected_wasted_atoms_per_row,
    overfull_row_probability,
    salvageable_share,
)


def _direct_waste(loading_array_size: int, columns: int, p: float) -> float:
    """E[(K - columns)+] summed term by term, the definition the closed
    form replaces."""
    return sum(
        (k - columns)
        * comb(loading_array_size, k)
        * p**k
        * (1 - p) ** (loading_array_size - k)
        for k in range(columns + 1, loading_array_size + 1)
    )


@pytest.mark.parametrize("size", [2, 3])
def test_enough_atoms_probability_matches_enumeration(size: int) -> None:
    sites = size * size
    for target_sites in range(sites + 1):
        counted = sum(
            1
            for bits in itertools.product((0, 1), repeat=sites)
            if sum(bits) >= target_sites
        ) / 2**sites
        assert isclose(
            enough_atoms_probability(size, target_sites), counted, abs_tol=1e-12
        )


def test_overfull_row_probability_matches_enumeration() -> None:
    size = 6
    for columns in range(size + 1):
        counted = sum(
            1
            for bits in itertools.product((0, 1), repeat=size)
            if sum(bits) > columns
        ) / 2**size
        assert isclose(
            overfull_row_probability(size, columns), counted, abs_tol=1e-12
        )


@pytest.mark.parametrize("size", [12, 20])
@pytest.mark.parametrize("p", [0.5, 0.3, 0.75])
def test_expected_waste_matches_direct_summation(size: int, p: float) -> None:
    for columns in range(size + 1):
        assert isclose(
            expected_wasted_atoms_per_row(size, columns, p),
            _direct_waste(size, columns, p),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )


def test_expected_waste_is_the_per_row_figure_times_the_rows() -> None:
    for size in (8, 30, 100):
        columns = int(size / 2**0.5) - 1
        assert isclose(
            expected_wasted_atoms(size, columns),
            size * expected_wasted_atoms_per_row(size, columns),
            rel_tol=1e-12,
        )


def test_expected_waste_matches_a_seeded_sample() -> None:
    size, columns = 12, 8
    rng = np.random.default_rng(20260910)
    draws = rng.integers(0, 2, size=(40_000, size, size)).sum(axis=2)
    surplus = np.clip(draws - columns, 0, None).sum(axis=1)
    error = surplus.std(ddof=1) / np.sqrt(surplus.size)
    assert abs(surplus.mean() - expected_wasted_atoms(size, columns)) < 4 * error


def test_a_row_narrower_than_the_target_wastes_nothing() -> None:
    for size in (4, 17, 60):
        assert expected_wasted_atoms_per_row(size, size) == 0.0
        assert overfull_row_probability(size, size) == 0.0


def test_overfull_row_probability_falls_as_the_target_widens() -> None:
    size = 40
    probabilities = [overfull_row_probability(size, c) for c in range(size + 1)]
    assert all(
        later <= earlier
        for earlier, later in itertools.pairwise(probabilities)
    )
    assert all(0.0 <= value <= 1.0 for value in probabilities)


def test_a_target_nobody_could_fill_is_never_reachable() -> None:
    assert enough_atoms_probability(5, 26) == 0.0
    assert enough_atoms_probability(5, 0) == 1.0


def test_salvageable_share_is_zero_when_the_rate_meets_the_bound() -> None:
    assert salvageable_share(0.8, 0.8) == 0.0
    # a measured rate above its own bound is sampling noise, not a
    # negative share
    assert salvageable_share(0.999664, 0.999624) == 0.0


def test_salvageable_share_splits_the_rejections() -> None:
    assert isclose(salvageable_share(0.5, 0.75), 0.5)
    assert isclose(salvageable_share(0.0, 1.0), 1.0)
    assert salvageable_share(1.0, 1.0) == 0.0


@pytest.mark.parametrize(
    "size, columns", [(0, 1), (-1, 2), (10, -1)]
)
def test_degenerate_arguments_are_rejected(size: int, columns: int) -> None:
    with pytest.raises(ValueError):
        overfull_row_probability(size, columns)
