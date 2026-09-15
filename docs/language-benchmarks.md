# Language benchmark corpus

LuaPyre's permanent cross-runtime suite includes a bounded, deterministic
language-benchmark group inspired by the Computer Language Benchmarks Game.
It runs the same ordinary Lua source in LuaPyre, PUC Lua 5.5, and LuaJIT;
`native_headroom.py` also runs a reduced-contract Python implementation of the
same algorithm.

These are microbenchmarks, not application-performance predictions. The
parameters are intentionally smaller than submission-sized Benchmarks Game
runs so the interpreter backend and routine CI remain practical. Every timed
result is checked.

| Workload | Fixed parameter | Result checked | Primary pressure |
| --- | ---: | --- | --- |
| n-body | 1,000 steps, 5 bodies | final system energy | floating-point loops and record access |
| mandelbrot | 40 x 40, 50 iterations | packed-row checksum | scalar floating point, branches, and bit operations |
| spectral-norm | 30 x 30, 10 products | norm within `1e-12` | nested numeric loops and dense tables |
| fannkuch-redux | `n = 7` | checksum and maximum flips | permutation mutation and integer loops |
| binary-trees | 30 trees, depth 8 | aggregate node count | allocation, recursion, traversal, and GC |
| fasta | 5,000 bases | output length and byte checksum | LCG, weighted selection, strings, and concatenation |
| k-nucleotide | 1,300 bases, 7 k sizes | selected-count checksum | substring creation and hash-table updates |
| reverse-complement | 4,000 bases | position-weighted byte checksum | character mapping, reversal, and string assembly |

## Deliberate adaptations

The original output-oriented programs normally read or write streams. The
portable corpus instead constructs deterministic input in memory and validates
compact checksums. That keeps filesystem policy and terminal buffering out of
the language-runtime comparison while retaining the algorithms' string,
table, allocation, and numeric work. Mandelbrot is the scalar kernel; this
suite does not claim to measure threads or SIMD because neither is part of the
common Lua source.

The Python column is an engineering lower bound rather than a semantically
equivalent runtime. It omits Lua fuel, metatables, debug state, and Lua-level
GC, but it does not substitute closed forms, memoization, NumPy, or another
algorithm.

## Pidigits disposition

Pidigits is not included in the portable three-runtime corpus. Its defining
workload uses arbitrary-precision integers, commonly through GMP, whereas Lua
5.5 and LuaPyre expose fixed-width signed 64-bit Lua integers. A small bounded
spigot would cease to measure the stated arbitrary-precision workload, and a
host bigint binding would measure unequal external-library interfaces. If a
portable bigint library is added to all compared runtimes later, pidigits can
be admitted as a separately labelled library benchmark.

Run just this group with:

```bash
python benchmarks/compare_runtimes.py --group language --require-all
```

For LuaPyre, native Lua 5.5, and Python lower-bound comparisons, select the same
names with repeated `--case` arguments to `benchmarks/native_headroom.py`.

## Initial 0.40 baseline

The initial run used three isolated processes, rotating implementation order,
with 7 warmups and 31 checked samples per implementation. Times are medians of
the three process medians. The working tree was based on commit `0bb256d`.

| Workload | Python | LuaPyre | Native Lua 5.5 | Python lower bound | vs native | vs Python |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n-body | 3.13 | 762.422 ms | 2.091 ms | 4.882 ms | 364.56x | 156.18x |
| n-body | 3.14 | 694.521 ms | 2.150 ms | 5.048 ms | 323.05x | 137.59x |
| Mandelbrot | 3.13 | 436.954 ms | 1.222 ms | 4.268 ms | 357.64x | 102.39x |
| Mandelbrot | 3.14 | 401.662 ms | 1.234 ms | 4.152 ms | 325.60x | 96.74x |
| fannkuch-redux | 3.13 | 711.137 ms | 3.187 ms | 6.001 ms | 223.12x | 118.51x |
| fannkuch-redux | 3.14 | 636.539 ms | 3.227 ms | 6.080 ms | 197.25x | 104.69x |
| FASTA | 3.13 | 180.129 ms | 0.589 ms | 1.016 ms | 305.69x | 177.22x |
| FASTA | 3.14 | 159.929 ms | 0.596 ms | 0.903 ms | 268.22x | 177.06x |
| k-nucleotide | 3.13 | 138.801 ms | 0.559 ms | 0.940 ms | 248.21x | 147.73x |
| k-nucleotide | 3.14 | 130.030 ms | 0.535 ms | 0.926 ms | 243.07x | 140.43x |
| reverse-complement | 3.13 | 103.117 ms | 0.442 ms | 0.394 ms | 233.37x | 261.73x |
| reverse-complement | 3.14 | 96.919 ms | 0.453 ms | 0.357 ms | 214.13x | 271.66x |

Raw reports are retained in
[`language_benchmarks_040_313.json`](../benchmarks/results/language_benchmarks_040_313.json)
and
[`language_benchmarks_040_314.json`](../benchmarks/results/language_benchmarks_040_314.json).
