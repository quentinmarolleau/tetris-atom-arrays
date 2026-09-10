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
| [`analytic`](tetris/analytic.py) | Closed forms for the atoms a loading wastes and for the bound every rearrangement obeys. |
| [`variants`](tetris/variants.py) | Names for the target reshapes the sweep compares. |
| [`stats`](tetris/stats.py) | Wilson score intervals for the measured rates. |
| [`display`](tetris/display.py) | Rich rendering of a tetrimino construction, row by row. |
| [`plotting`](tetris/plotting.py) | Matplotlib rendering of a loading array and its packed counterpart. |

`benchmark_stats.json` and `displacement_stats.json` hold the two cached sweeps.
Each carries a fingerprint of the parameters it was run with and a hash of
`tetris/config.py` and `tetris/fast.py`, so a changed size range or a changed
algorithm is never served silently from an old run. Editing either of those two
files therefore costs a rerun, roughly three quarters of an hour on eight
threads.

## The target

The target array has a natural side of ⌊*L*/√2⌋, which asks for about half the
loading sites and so for about as many atoms as a half-filled loading provides.
`reshape_target` adjusts that side one axis at a time:

```python
AtomsConfiguration(loading_array_size=18)                       # (0, 0)
AtomsConfiguration(loading_array_size=18, reshape_target=(0, -1))
AtomsConfiguration(loading_array_size=18, reshape_target=(-1, -1))
AtomsConfiguration(loading_array_size=18, target_shape=(8, 12))
```

Rows and columns are not the same knob. A loaded row gives at most one atom to
each target column, so the width limits what a row can contribute while the
height sets what each column must supply. Two targets with the same number of
sites are therefore not equally easy: at *L* = 14 a 9 × 11 target is accepted
47.6 % of the time and its transpose, asking for the same 99 atoms, 36.3 %.

## What the numbers say

Against the paper, on the reshape it uses, one row and one column off:

| Quantity | Here | Wang *et al.* |
|---|---|---|
| Parallel displacements against atom number | *N*<sup>0.951(3)</sup> | *N*<sup>1.03(7)</sup>, and *N*<sup>1.6(1)</sup> for Hungarian |
| Acceptance over the displacement sweep | 0.99975 | 0.9982 |

Across loading sizes 4 to 100, mean success rate by reshape:

| Reshape | Mean | Worst |
|---|---|---|
| (0, 0), no margin | 0.8607 | 0.5082 at *L* = 99 |
| (0, −1), one column off | 0.9868 | 0.9234 at *L* = 17 |
| (−1, −1), one row and one column off | 0.9997 | 0.9974 at *L* = 17 |

Giving up one column of the target buys most of the distance on its own. The
no-margin curve swings that widely because the target side is a floor, so the
contraction ratio oscillates around one half instead of sitting on it.

Almost none of those failures are the algorithm's. Filling the target needs as
many atoms as it has sites, and comparing the measured rate against that bound
leaves 9.4 % of the rejections at *L* = 10 recoverable by any better packing,
and 0.4 % at *L* = 100.

## Licence

MIT. See [LICENSE](LICENSE).
