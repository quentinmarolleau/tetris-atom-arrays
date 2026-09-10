"""Parallel benchmark driver.

Every worker holds its own random stream, sets its own deadline and
reports its own CPU time. Tasks are drawn from one queue in a shuffled
order, so that nothing about the machine's scheduling or its thermal
behaviour lines up with the array size being measured.
"""

import multiprocessing as mp
import random
from time import monotonic, process_time

import numpy as np

from tetris.fast import (
    configuration_kept,
    parallel_displacements,
    target_geometry,
)
from tetris.stats import standard_error, wilson_interval
from tetris.variants import variant_key

__all__ = [
    "build_tasks",
    "run_displacement_benchmark",
    "run_success_rate_benchmark",
    "task_generator",
    "task_seeds",
]

_SENTINEL = None


def task_seeds(entropy: int, count: int) -> list[np.random.SeedSequence]:
    """One independent child stream per task. The failure this replaces:
    forked workers inherited a single global NumPy state, so workers on
    the same array size drew identical configurations."""
    return np.random.SeedSequence(entropy).spawn(count)


def task_generator(seed: np.random.SeedSequence) -> np.random.Generator:
    return np.random.default_rng(seed)


def build_tasks(
    sizes: tuple[int, ...],
    reshape_variants: tuple[tuple[int, int], ...],
    entropy: int,
) -> list[tuple[int, tuple[int, int], np.random.SeedSequence]]:
    """Shuffled (size, reshape, seed) tasks.

    Running in size order lets any slow drift over the run, a laptop
    heating up over half an hour for instance, masquerade as a trend
    against array size. Shuffling turns it into noise. The shuffle is
    driven by the recorded entropy, so a run stays reproducible."""
    combinations = [
        (int(size), tuple(reshape))
        for size in sizes
        for reshape in reshape_variants
    ]
    seeds = task_seeds(entropy, len(combinations))
    tasks = [
        (size, reshape, seed)
        for (size, reshape), seed in zip(combinations, seeds)
    ]
    random.Random(entropy).shuffle(tasks)
    return tasks


def _sample_until(deadline: float, rng, size: int, window):
    """Draw configurations until the deadline, counting the kept ones."""
    kept = 0
    samples = 0
    while monotonic() < deadline:
        matrix = rng.integers(0, 2, size=(size, size)).astype(bool)
        kept += configuration_kept(matrix, window)
        samples += 1
    return kept, samples


def _success_rate_worker(tasks, results, seconds_per_task, active) -> None:
    while True:
        task = tasks.get()
        if task is _SENTINEL:
            return
        size, reshape, seed = task
        window = target_geometry(size, reshape)
        rng = task_generator(seed)

        with active.get_lock():
            active.value += 1
            concurrency_in = active.value
        cpu_start, wall_start = process_time(), monotonic()
        kept, samples = _sample_until(
            wall_start + seconds_per_task, rng, size, window
        )
        cpu_seconds = process_time() - cpu_start
        wall_seconds = monotonic() - wall_start
        with active.get_lock():
            concurrency_out = active.value
            active.value -= 1

        results.put(
            {
                "size": size,
                "reshape": list(reshape),
                "kept": kept,
                "samples": samples,
                "cpu_seconds": cpu_seconds,
                "wall_seconds": wall_seconds,
                "target_rows": window.rows,
                "target_columns": window.columns,
                "target_row_start": window.row_start,
                "target_column_start": window.column_start,
                "min_concurrency": min(concurrency_in, concurrency_out),
            }
        )


def _drain(worker, tasks_list, workers: int, *args) -> list[dict]:
    """Run `worker` over `tasks_list` on `workers` forked processes."""
    context = mp.get_context("fork")
    tasks: mp.Queue = context.Queue()
    results: mp.Queue = context.Queue()
    active = context.Value("i", 0)
    for task in tasks_list:
        tasks.put(task)
    for _ in range(workers):
        tasks.put(_SENTINEL)

    processes = [
        context.Process(target=worker, args=(tasks, results, *args, active))
        for _ in range(workers)
    ]
    for process in processes:
        process.start()
    collected = [results.get() for _ in tasks_list]
    for process in processes:
        process.join()
    return collected


def _summarise(record: dict) -> tuple[int, str, dict]:
    size, kept, samples = record["size"], record["kept"], record["samples"]
    sites = record["target_rows"] * record["target_columns"]
    low, high = wilson_interval(kept, samples)
    per_configuration = (
        record["cpu_seconds"] / samples * 1e6 if samples else float("nan")
    )
    return (
        size,
        variant_key(tuple(record["reshape"])),
        {
            "success_rate": kept / samples if samples else float("nan"),
            "standard_error": standard_error(kept, samples),
            "wilson_low": low,
            "wilson_high": high,
            "kept": kept,
            "samples": samples,
            "cpu_seconds": record["cpu_seconds"],
            "wall_seconds": record["wall_seconds"],
            "microseconds_per_configuration": per_configuration,
            "microseconds_per_row": per_configuration / size,
            "min_concurrency": record["min_concurrency"],
            "contraction_ratio": sites / size**2,
            "target_number_of_atoms": sites,
            "target_rows": record["target_rows"],
            "target_columns": record["target_columns"],
            "number_of_loading_sites": size**2,
        },
    )


def run_success_rate_benchmark(
    sizes: tuple[int, ...],
    reshape_variants: tuple[tuple[int, int], ...],
    seconds_per_task: float,
    workers: int,
    entropy: int,
) -> dict[int, dict]:
    tasks = build_tasks(sizes, reshape_variants, entropy)
    records = _drain(_success_rate_worker, tasks, workers, seconds_per_task)
    results: dict[int, dict] = {}
    for record in records:
        size, variant, summary = _summarise(record)
        results.setdefault(size, {})[variant] = summary
    return dict(sorted(results.items()))


def _displacement_worker(tasks, results, samples_per_task, active) -> None:
    while True:
        task = tasks.get()
        if task is _SENTINEL:
            return
        size, reshape, seed = task
        window = target_geometry(size, reshape)
        rng = task_generator(seed)

        with active.get_lock():
            active.value += 1
        displacements: list[int] = []
        attempts = 0
        cpu_start = process_time()
        while len(displacements) < samples_per_task:
            matrix = rng.integers(0, 2, size=(size, size)).astype(bool)
            attempts += 1
            if configuration_kept(matrix, window):
                displacements.append(
                    parallel_displacements(matrix, window)
                )
        cpu_seconds = process_time() - cpu_start
        with active.get_lock():
            active.value -= 1

        array = np.asarray(displacements, dtype=float)
        results.put(
            {
                "size": size,
                "reshape": list(reshape),
                "target_number_of_atoms": window.sites,
                "target_rows": window.rows,
                "target_columns": window.columns,
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)),
                "standard_error": float(array.std(ddof=1) / np.sqrt(len(array))),
                "samples": len(displacements),
                "attempts": attempts,
                "cpu_seconds": cpu_seconds,
            }
        )


DISPLACEMENT_RESHAPE = (-1, -1)


def run_displacement_benchmark(
    sizes: tuple[int, ...],
    samples_per_task: int,
    workers: int,
    entropy: int,
) -> dict[int, dict]:
    """Mean parallel displacements per accepted configuration, at a fixed
    sample count per size so that every point carries the same weight.
    Only the full margin is swept: that is the reservoir size Wang et al.
    use in their simulations, and it is their exponent this is compared
    against."""
    tasks = build_tasks(sizes, (DISPLACEMENT_RESHAPE,), entropy)
    records = _drain(_displacement_worker, tasks, workers, samples_per_task)
    key = variant_key(DISPLACEMENT_RESHAPE)
    return {
        record["size"]: {key: record}
        for record in sorted(records, key=lambda item: item["size"])
    }
