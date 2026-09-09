"""Benchmark cache metadata and its invalidation."""

import json

import pytest

from tetris.cache import BenchmarkParameters, load_results, save_results


def parameters(**overrides) -> BenchmarkParameters:
    defaults = {
        "sizes": (4, 5, 6),
        "margin_variants": (False, True),
        "seconds_per_task": 60.0,
        "workers": 7,
        "algorithm_digest": "abc123",
    }
    return BenchmarkParameters(**{**defaults, **overrides})


def test_round_trip(tmp_path) -> None:
    path = tmp_path / "stats.json"
    params = parameters()
    save_results(path, params, entropy=999, results={4: {"no_margin": 1.0}})
    results, reason = load_results(path, params)
    assert reason == ""
    assert results == {4: {"no_margin": 1.0}}


def test_missing_file_is_not_an_error(tmp_path) -> None:
    results, reason = load_results(tmp_path / "absent.json", parameters())
    assert results is None
    assert "no cache" in reason


@pytest.mark.parametrize(
    "override",
    [
        {"sizes": (4, 5, 6, 7)},
        {"margin_variants": (True,)},
        {"seconds_per_task": 30.0},
        {"algorithm_digest": "deadbeef"},
        {"samples_per_task": 10_000},
    ],
)
def test_parameter_change_invalidates(tmp_path, override) -> None:
    path = tmp_path / "stats.json"
    save_results(path, parameters(), entropy=1, results={4: {}})
    results, reason = load_results(path, parameters(**override))
    assert results is None
    assert "parameters" in reason


def test_worker_count_does_not_invalidate(tmp_path) -> None:
    """The worker count is recorded for provenance but does not change
    the sampled distribution, so it must not throw the cache away."""
    path = tmp_path / "stats.json"
    save_results(path, parameters(), entropy=1, results={4: {}})
    results, _ = load_results(path, parameters(workers=3))
    assert results is not None


def test_corrupt_file_is_reported(tmp_path) -> None:
    path = tmp_path / "stats.json"
    path.write_text("{not json")
    results, reason = load_results(path, parameters())
    assert results is None
    assert "unreadable" in reason


def test_entropy_is_recorded(tmp_path) -> None:
    path = tmp_path / "stats.json"
    save_results(path, parameters(), entropy=4242, results={4: {}})
    meta = json.loads(path.read_text())["meta"]
    assert meta["seed_entropy"] == 4242
    assert "cpu_count" in meta


def test_cpu_model_is_readable() -> None:
    from tetris.cache import cpu_model

    name = cpu_model()
    assert name
    assert name != "x86_64"
