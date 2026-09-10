"""The benchmark driver, exercised at a budget small enough for CI."""

import pytest

from tetris.benchmark import (
    build_tasks,
    run_displacement_benchmark,
    run_success_rate_benchmark,
)

SIZES = (6, 9, 12)
VARIANTS = ((0, 0), (0, -1), (-1, -1))
KEYS = {"r0c0", "r0c-1", "r-1c-1"}


def test_tasks_cover_every_combination() -> None:
    tasks = build_tasks(SIZES, VARIANTS, entropy=1)
    assert len(tasks) == len(SIZES) * len(VARIANTS)
    assert {(size, reshape) for size, reshape, _ in tasks} == {
        (size, reshape) for size in SIZES for reshape in VARIANTS
    }


def test_task_order_is_shuffled_but_reproducible() -> None:
    sizes = tuple(range(4, 40))
    first = [
        (size, reshape) for size, reshape, _ in build_tasks(sizes, VARIANTS, 7)
    ]
    again = [
        (size, reshape) for size, reshape, _ in build_tasks(sizes, VARIANTS, 7)
    ]
    ordered = [(size, reshape) for size in sizes for reshape in VARIANTS]
    assert first == again
    assert first != ordered


def test_success_rate_run_is_well_formed() -> None:
    results = run_success_rate_benchmark(
        SIZES, VARIANTS, seconds_per_task=0.3, workers=2, entropy=11
    )
    assert set(results) == set(SIZES)
    for size, variants in results.items():
        assert set(variants) == KEYS
        for entry in variants.values():
            assert entry["samples"] > 0
            assert entry["kept"] <= entry["samples"]
            assert entry["wilson_low"] <= entry["success_rate"]
            assert entry["success_rate"] <= entry["wilson_high"]
            assert entry["cpu_seconds"] > 0
            assert entry["microseconds_per_row"] > 0
            assert entry["number_of_loading_sites"] == size**2


def test_cost_comes_from_measured_cpu_time() -> None:
    """The nominal budget must not be reused as the measured cost: that
    is what made an under-subscribed batch look like a faster algorithm."""
    budget = 0.3
    results = run_success_rate_benchmark(
        SIZES, ((-1, -1),), seconds_per_task=budget, workers=2, entropy=12
    )
    measured = []
    for variants in results.values():
        entry = variants["r-1c-1"]
        measured.append(entry["cpu_seconds"])
        assert entry["microseconds_per_configuration"] == pytest.approx(
            entry["cpu_seconds"] / entry["samples"] * 1e6
        )
        # the sampling loop tests its deadline before each draw, so it
        # always overruns rather than stopping short
        assert entry["wall_seconds"] >= budget
        # cpu_seconds is not compared against wall_seconds: process_time
        # sums the CPU of every thread in the process, and a threaded
        # BLAS underneath NumPy can put it above the wall clock. What
        # has to hold is that the number is measured rather than the
        # budget played back
        assert entry["cpu_seconds"] != pytest.approx(budget, abs=1e-6)
    # each task times itself, so the values are not one shared constant
    assert len(set(measured)) == len(measured)


def test_displacement_run_is_well_formed() -> None:
    results = run_displacement_benchmark(
        (10, 16), samples_per_task=40, workers=2, entropy=13
    )
    assert set(results) == {10, 16}
    for variants in results.values():
        entry = variants["r-1c-1"]
        assert entry["samples"] == 40
        assert entry["attempts"] >= entry["samples"]
        assert entry["mean"] > 0
        assert entry["target_number_of_atoms"] > 0
