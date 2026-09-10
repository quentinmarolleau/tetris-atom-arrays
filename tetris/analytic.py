"""Closed forms for two questions the benchmark would otherwise have to
sample.

How many atoms does a loading waste? A row hands at most one atom to
each target column, so a row carrying more atoms than the target is wide
leaves the surplus where it lies. The count is a binomial tail.

How many discarded loadings held enough atoms to have filled the target
anyway? An accepted loading always does, so the share of rejections that
were salvageable follows from the measured success rate and a second
binomial tail, over the whole array this time. Nothing has to be drawn.

`total >= sites` is what an assembler free to move any atom anywhere
would need. A row-then-column scheme is more constrained than that, so
the share computed here is an upper bound on what any change to the
packing rule could recover.
"""

from functools import cache
from math import comb, exp, lgamma, log

__all__ = [
    "enough_atoms_probability",
    "expected_wasted_atoms",
    "expected_wasted_atoms_per_row",
    "overfull_row_probability",
    "salvageable_share",
]


def _check(draws: int, threshold: int) -> None:
    if draws < 1:
        raise ValueError(f"a binomial needs at least one draw, got {draws}")
    if threshold < 0:
        raise ValueError(f"a count cannot be negative, got {threshold}")


@cache
def _upper_tail(draws: int, threshold: int, p: float = 0.5) -> float:
    """P(X >= threshold) for X ~ Bin(draws, p).

    At p = 1/2 the tail is a sum of binomial coefficients over a power of
    two, so it is accumulated in exact integers and divided once. The
    coefficients come from the recurrence rather than from `comb` per
    term: C(n, j+1) = C(n, j) (n-j) / (j+1) divides exactly, and one
    multiply-divide beats recomputing a factorial ratio when the sum runs
    to thousands of terms. Any other p goes through logs.
    """
    if threshold <= 0:
        return 1.0
    if threshold > draws:
        return 0.0

    if p == 0.5:
        term = comb(draws, threshold)
        total = term
        for j in range(threshold, draws):
            term = term * (draws - j) // (j + 1)
            total += term
        return total / 2**draws

    log_p, log_q = log(p), log(1.0 - p)
    log_choose = lgamma(draws + 1)
    return sum(
        exp(
            log_choose
            - lgamma(j + 1)
            - lgamma(draws - j + 1)
            + j * log_p
            + (draws - j) * log_q
        )
        for j in range(threshold, draws + 1)
    )


def overfull_row_probability(
    loading_array_size: int, columns: int, p: float = 0.5
) -> float:
    """Chance that one loaded row carries more atoms than the target is
    wide, so that some of them cannot be placed."""
    _check(loading_array_size, columns)
    return _upper_tail(loading_array_size, columns + 1, p)


def expected_wasted_atoms_per_row(
    loading_array_size: int, columns: int, p: float = 0.5
) -> float:
    """Mean surplus of one row, E[(K - columns)+] for K ~ Bin(L, p).

    Splitting the expectation at the threshold gives
    E[(K-m)+] = E[K 1{K>m}] - m P(K>m), and the first term is
    L p P(Bin(L-1, p) >= m). Two tails instead of a sum over the tail.
    """
    _check(loading_array_size, columns)
    if columns >= loading_array_size:
        return 0.0
    carried = (
        loading_array_size
        * p
        * _upper_tail(loading_array_size - 1, columns, p)
    )
    return carried - columns * _upper_tail(loading_array_size, columns + 1, p)


def expected_wasted_atoms(
    loading_array_size: int, columns: int, p: float = 0.5
) -> float:
    """Mean number of atoms one whole loading leaves behind, summed over
    its rows."""
    return loading_array_size * expected_wasted_atoms_per_row(
        loading_array_size, columns, p
    )


def enough_atoms_probability(
    loading_array_size: int, target_sites: int, p: float = 0.5
) -> float:
    """Chance that a loading holds at least as many atoms as the target
    has sites, which is what any rearrangement needs and what the Tetris
    acceptance rule is bounded by."""
    _check(loading_array_size, target_sites)
    return _upper_tail(loading_array_size**2, target_sites, p)


def salvageable_share(success_rate: float, enough_atoms: float) -> float:
    """Share of the discarded loadings that held enough atoms to have
    filled the target.

    An accepted loading always holds enough, so the two events nest and
    the joint probability is the difference of the marginals. A measured
    rate sitting a hair above its own bound is sampling noise on a rate
    of 0.9997, not a negative share, so the result is clamped.
    """
    rejected = 1.0 - success_rate
    if rejected <= 0.0:
        return 0.0
    return min(1.0, max(0.0, (enough_atoms - success_rate) / rejected))
