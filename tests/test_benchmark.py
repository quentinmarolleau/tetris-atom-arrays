"""The benchmark driver, exercised at a budget small enough for CI."""

import pytest

from tetris.benchmark import (
    build_tasks,
    run_displacement_benchmark,
    run_success_rate_benchmark,
)

SIZES = (6, 9, 12)
VARIANTS = (False, True)


def test_tasks_cover_every_combination() -> None:
    tasks = build_tasks(SIZES, VARIANTS, entropy=1)
    assert len(tasks) == len(SIZES) * len(VARIANTS)
    assert {(size, margin) for size, margin, _ in tasks} == {
        (size, margin) for size in SIZES for margin in VARIANTS
    }


def test_task_order_is_shuffled_but_reproducible() -> None:
    sizes = tuple(range(4, 40))
    first = [(size, margin) for size, margin, _ in build_tasks(sizes, VARIANTS, 7)]
    again = [(size, margin) for size, margin, _ in build_tasks(sizes, VARIANTS, 7)]
    ordered = [(size, margin) for size in sizes for margin in VARIANTS]
    assert first == again
    assert first != ordered


def test_success_rate_run_is_well_formed() -> None:
    results = run_success_rate_benchmark(
        SIZES, VARIANTS, seconds_per_task=0.3, workers=2, entropy=11
    )
    assert set(results) == set(SIZES)
    for size, variants in results.items():
        assert set(variants) == {"no_margin", "with_margin"}
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
    results = run_success_rate_benchmark(
        SIZES, (True,), seconds_per_task=0.3, workers=2, entropy=12
    )
    measured = []
    for variants in results.values():
        entry = variants["with_margin"]
        measured.append(entry["cpu_seconds"])
        assert entry["microseconds_per_configuration"] == pytest.approx(
            entry["cpu_seconds"] / entry["samples"] * 1e6
        )
        assert entry["cpu_seconds"] <= entry["wall_seconds"] + 0.05
    # each task times itself, so the values are not one shared constant
    assert len(set(measured)) == len(measured)


def test_displacement_run_is_well_formed() -> None:
    results = run_displacement_benchmark(
        (10, 16), samples_per_task=40, workers=2, entropy=13
    )
    assert set(results) == {10, 16}
    for variants in results.values():
        entry = variants["with_margin"]
        assert entry["samples"] == 40
        assert entry["attempts"] >= entry["samples"]
        assert entry["mean"] > 0
        assert entry["target_number_of_atoms"] > 0
