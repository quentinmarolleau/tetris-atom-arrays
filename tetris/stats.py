"""Interval estimates for the benchmarked success rates."""

import numpy as np

__all__ = ["standard_error", "wilson_interval"]


def standard_error(successes: int, samples: int) -> float:
    """Standard error of a proportion. Not the standard deviation of the
    Bernoulli draws, which is larger by a factor sqrt(samples) and says
    nothing about how well the rate is known."""
    if samples <= 0:
        return float("nan")
    rate = successes / samples
    return float(np.sqrt(rate * (1.0 - rate) / samples))


def wilson_interval(
    successes: int, samples: int, z: float = 1.0
) -> tuple[float, float]:
    """Wilson score interval at `z` standard deviations.

    The success rates here run to within a few parts in ten thousand of
    one, where a symmetric interval would reach past one and understate
    how much room is left below. The Wilson interval stays inside [0, 1]
    and keeps its coverage at those extremes.
    """
    if samples <= 0:
        return float("nan"), float("nan")
    rate = successes / samples
    denominator = 1.0 + z**2 / samples
    centre = (rate + z**2 / (2 * samples)) / denominator
    spread = (
        z
        * np.sqrt(rate * (1 - rate) / samples + z**2 / (4 * samples**2))
        / denominator
    )
    low = float(max(0.0, centre - spread))
    high = float(min(1.0, centre + spread))
    # at a rate of exactly zero or one the analytic bound lands on the
    # observed rate, and rounding can put it a fraction on the wrong
    # side; an interval that excludes its own point estimate is worse
    # than one that is a few ulps wide
    return min(low, rate), max(high, rate)
