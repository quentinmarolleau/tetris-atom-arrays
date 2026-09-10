# Tetris algorithm for atomic array sorting

A Python reproduction of the rearrangement scheme of Wang *et al.*,
[*Accelerating the Assembly of Defect-Free Atomic Arrays with Maximum
Parallelisms*][paper], Phys. Rev. Applied **19**, 054032 (2023), together with
benchmarks of its success rate and of its rearrangement cost.

Individual atoms load into an array of optical tweezers with probability one
half per site, so preparing a dense, defect-free target array means moving
atoms after the loading image comes back. The paper's scheme does it row by
row, which suits a camera streaming its rows one at a time, and packs each row
against the ones before it the way a Tetris player packs a well. This
repository implements that, checks it against the paper's own numbers, and
writes the whole thing up as a marimo notebook.

The notebook is the point; the package exists so that the notebook can be
tested and benchmarked rather than only read.

[paper]: https://doi.org/10.1103/PhysRevApplied.19.054032

## Running it

Everything is driven by [uv](https://docs.astral.sh/uv/).

```sh
uv run marimo edit tetris_problem.py   # the notebook, live
uv run pytest                          # 59 tests
uv run ruff check tetris tests         # lint the package and the tests
```

Lint the package and the tests, not the notebook: ruff flags the trailing
`return` that closes every marimo cell, which is marimo's file format rather
than a finding.

## Layout

| Module | Description |
|---|:---|
| [`config`](tetris/config.py) | `AtomsConfiguration`: generates a loading, packs it into tetriminoes, renders it. Pedagogical, not fast. |
| [`fast`](tetris/fast.py) | The same acceptance rule and displacement count at constant cost per row, which is what makes the sweep affordable. |
| [`benchmark`](tetris/benchmark.py) | The parallel driver. Seeds each worker independently, shuffles task order, measures CPU time. |
| [`cache`](tetris/cache.py) | On-disk benchmark results, fingerprinted against their parameters and the algorithm source. |
| [`stats`](tetris/stats.py) | Wilson score intervals for the measured rates. |
| [`display`](tetris/display.py) | Rich rendering of a tetrimino construction, row by row. |
| [`plotting`](tetris/plotting.py) | Matplotlib rendering of a loading array and its packed counterpart. |

`benchmark_stats.json` and `displacement_stats.json` hold the two cached sweeps.
Each carries a fingerprint of the parameters it was run with and a hash of
`tetris/config.py` and `tetris/fast.py`, so a changed size range or a changed
algorithm is never served silently from an old run. Editing either of those two
files therefore costs a rerun, about half an hour.

`view_raw_frame.py` and `sample.tif` are unrelated to the algorithm: a raw
tweezer frame and a contrast-stretch viewer for it.

## What the numbers say

Against the paper, on the margin variant it uses:

| Quantity | Here | Wang *et al.* |
|---|---|---|
| Parallel displacements against atom number | *N*<sup>0.951(3)</sup> | *N*<sup>1.03(7)</sup>, and *N*<sup>1.6(1)</sup> for Hungarian |
| Acceptance over the displacement sweep | 0.99974 | 0.9982 |

Across loading sizes 4 to 100 the mean success rate is 0.9997 with the margin,
never dropping below 0.9973. Without it the mean is 0.8606 and the worst size
reaches 0.5070, because the target side is a floor and the contraction ratio
oscillates around one half instead of sitting on it.

## Licence

MIT. See [LICENSE](LICENSE).
