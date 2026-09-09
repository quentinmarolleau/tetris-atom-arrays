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
    margin_variants: tuple[bool, ...],
    entropy: int,
) -> list[tuple[int, bool, np.random.SeedSequence]]:
    """Shuffled (size, margin, seed) tasks.

    Running in size order lets any slow drift over the run, a laptop
    heating up over half an hour for instance, masquerade as a trend
    against array size. Shuffling turns it into noise. The shuffle is
    driven by the recorded entropy, so a run stays reproducible."""
    combinations = [
        (int(size), margin) for size in sizes for margin in margin_variants
    ]
    seeds = task_seeds(entropy, len(combinations))
    tasks = [
        (size, margin, seed)
        for (size, margin), seed in zip(combinations, seeds)
    ]
    random.Random(entropy).shuffle(tasks)
    return tasks


def _sample_until(deadline: float, rng, size: int, target_size: int):
    """Draw configurations until the deadline, counting the kept ones."""
    kept = 0
    samples = 0
    while monotonic() < deadline:
        matrix = rng.integers(0, 2, size=(size, size)).astype(bool)
        kept += configuration_kept(matrix, target_size)
        samples += 1
    return kept, samples


def _success_rate_worker(tasks, results, seconds_per_task, active) -> None:
    while True:
        task = tasks.get()
        if task is _SENTINEL:
            return
        size, margin, seed = task
        target_size, target_start = target_geometry(size, margin)
        rng = task_generator(seed)

        with active.get_lock():
            active.value += 1
            concurrency_in = active.value
        cpu_start, wall_start = process_time(), monotonic()
        kept, samples = _sample_until(
            wall_start + seconds_per_task, rng, size, target_size
        )
        cpu_seconds = process_time() - cpu_start
        wall_seconds = monotonic() - wall_start
        with active.get_lock():
            concurrency_out = active.value
            active.value -= 1

        results.put(
            {
                "size": size,
                "margin": margin,
                "kept": kept,
                "samples": samples,
                "cpu_seconds": cpu_seconds,
                "wall_seconds": wall_seconds,
                "target_size": target_size,
                "target_start": target_start,
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
    target_size = record["target_size"]
    low, high = wilson_interval(kept, samples)
    per_configuration = (
        record["cpu_seconds"] / samples * 1e6 if samples else float("nan")
    )
    return (
        size,
        "with_margin" if record["margin"] else "no_margin",
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
            "contraction_ratio": (target_size / size) ** 2,
            "target_number_of_atoms": target_size**2,
            "number_of_loading_sites": size**2,
        },
    )


def run_success_rate_benchmark(
    sizes: tuple[int, ...],
    margin_variants: tuple[bool, ...],
    seconds_per_task: float,
    workers: int,
    entropy: int,
) -> dict[int, dict]:
    tasks = build_tasks(sizes, margin_variants, entropy)
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
        size, margin, seed = task
        target_size, target_start = target_geometry(size, margin)
        rng = task_generator(seed)

        with active.get_lock():
            active.value += 1
        displacements: list[int] = []
        attempts = 0
        cpu_start = process_time()
        while len(displacements) < samples_per_task:
            matrix = rng.integers(0, 2, size=(size, size)).astype(bool)
            attempts += 1
            if configuration_kept(matrix, target_size):
                displacements.append(
                    parallel_displacements(matrix, target_size, target_start)
                )
        cpu_seconds = process_time() - cpu_start
        with active.get_lock():
            active.value -= 1

        array = np.asarray(displacements, dtype=float)
        results.put(
            {
                "size": size,
                "margin": margin,
                "target_number_of_atoms": target_size**2,
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)),
                "standard_error": float(array.std(ddof=1) / np.sqrt(len(array))),
                "samples": len(displacements),
                "attempts": attempts,
                "cpu_seconds": cpu_seconds,
            }
        )


def run_displacement_benchmark(
    sizes: tuple[int, ...],
    samples_per_task: int,
    workers: int,
    entropy: int,
) -> dict[int, dict]:
    """Mean parallel displacements per accepted configuration, at a fixed
    sample count per size so that every point carries the same weight.
    The margin variant is the one that matches the reservoir size Wang
    et al. use in their simulations."""
    tasks = build_tasks(sizes, (True,), entropy)
    records = _drain(_displacement_worker, tasks, workers, samples_per_task)
    return {
        record["size"]: {"with_margin": record}
        for record in sorted(records, key=lambda item: item["size"])
    }
