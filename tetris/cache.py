"""On-disk cache for the benchmark, with enough provenance that a stale
file cannot be mistaken for a fresh one.

The benchmark takes about half an hour, so its results are kept on disk
and reused. The previous version keyed that reuse on the file merely
existing, which meant a changed size range or a changed algorithm was
silently served from an old run. Everything that can move the numbers is
folded into a fingerprint instead, and a mismatch reports why.
"""

import hashlib
import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "BenchmarkParameters",
    "algorithm_digest",
    "load_results",
    "save_results",
]


@dataclass(frozen=True)
class BenchmarkParameters:
    """Everything a benchmark run is characterised by. `workers` is
    recorded for provenance but stays out of the fingerprint: it changes
    how long the run takes, not the distribution it samples."""

    sizes: tuple[int, ...]
    margin_variants: tuple[bool, ...]
    seconds_per_task: float
    workers: int
    algorithm_digest: str

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "sizes": list(self.sizes),
                "margin_variants": list(self.margin_variants),
                "seconds_per_task": self.seconds_per_task,
                "algorithm_digest": self.algorithm_digest,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def algorithm_digest(*paths: Path) -> str:
    """Hash of the sources that decide what the benchmark measures, so
    that editing the algorithm invalidates the cached numbers."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()[:16]


def save_results(
    path: Path,
    parameters: BenchmarkParameters,
    entropy: int,
    results: dict,
    **extra_meta,
) -> None:
    payload = {
        "meta": {
            "date": datetime.now(timezone.utc).isoformat(),
            "fingerprint": parameters.fingerprint(),
            "sizes": list(parameters.sizes),
            "margin_variants": list(parameters.margin_variants),
            "seconds_per_task": parameters.seconds_per_task,
            "workers": parameters.workers,
            "algorithm_digest": parameters.algorithm_digest,
            "seed_entropy": entropy,
            "cpu_count": os.cpu_count(),
            "cpu_model": platform.processor() or platform.machine(),
            **extra_meta,
        },
        "results": {
            str(size): variants for size, variants in sorted(results.items())
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_results(
    path: Path, parameters: BenchmarkParameters
) -> tuple[dict | None, str]:
    """Cached results, or None and the reason they could not be used."""
    path = Path(path)
    if not path.exists():
        return None, f"no cache at {path.name}"
    try:
        payload = json.loads(path.read_text())
        meta = payload["meta"]
        results = payload["results"]
    except (json.JSONDecodeError, KeyError, OSError) as error:
        return None, f"cache unreadable: {error}"

    if meta.get("fingerprint") != parameters.fingerprint():
        return None, (
            "cache parameters differ from the current ones "
            f"({meta.get('fingerprint')} against "
            f"{parameters.fingerprint()})"
        )
    return {int(size): variants for size, variants in results.items()}, ""
