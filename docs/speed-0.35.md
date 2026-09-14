# LuaPyre 0.35 performance tranche

0.35 implements the measured Python-only work from the 0.35 roadmap. It keeps
source/AST generation and deliberately does not generate CPython bytecode.

## Changes

1. **Use a scalar compiled-leaf ABI at the Python boundary.** Certified
   one-integer, one-result leaves receive the argument as a Python local and
   return a scalar. The admitted path allocates no scratch register list, empty
   cell dictionary, or one-element result tuple. Live JIT, hook, re-entry,
   stack, fuel, conversion, and safepoint checks remain in the cached adapter.
2. **Batch equal-cost diamond fuel.** Fully funded range-proven diamonds charge
   their exact total outside the Python loop. A semantic side exit derives the
   completed iteration count from the induction value, while short budgets use
   the precise per-path fallback.
3. **Emit leaner scalar AST.** Immutable nil, Boolean, integer, and byte-string
   constants become Python literals; known nonzero modulo divisors lose their
   zero check; self-copies and repeated Boolean coercions disappear.
4. **Retain string facts through structured diamonds.** One dominating entry
   guard proves the accumulator representation and both arms emit direct
   `bytes + bytes`. This removes the generated state dispatcher and repeated
   conversion checks from the string-building loop.

An already-owned-root shortcut in `LuaGC.adopt` was measured and removed: it
changed binary trees by only about 1–2%, below the tranche's 5% acceptance gate.
The larger fixed-point CFG, nested-region, table-representation, and lazy
recursive-activation candidates remain future work; none is represented here
as accepted without cross-version evidence.

## Measurement

`benchmarks/speed_035_ab.py` ran in three paired processes per version in A/B,
B/A, A/B order, with seven warmups, 31 checked samples, fixed CPU affinity, and
`PYTHONHASHSEED=0`. The table reports the median of process medians; raw values
for all nine workloads are in
[`speed_035.json`](../benchmarks/results/speed_035.json).

| Workload | Python | 0.34 | 0.35 | Change |
| --- | --- | ---: | ---: | ---: |
| Typed branch | 3.13.15 | 4.270 ms | 2.200 ms | **48.5% faster** |
| Typed branch | 3.14.7 | 3.648 ms | 2.003 ms | **45.1% faster** |
| String building | 3.13.15 | 2.625 ms | 0.602 ms | **77.1% faster** |
| String building | 3.14.7 | 2.249 ms | 0.591 ms | **73.7% faster** |
| 1,000 Python calls | 3.13.15 | 0.931 ms | 0.759 ms | **18.5% faster** |
| 1,000 Python calls | 3.14.7 | 1.012 ms | 0.804 ms | **20.6% faster** |

Table mix, sieve, binary trees, recursive Fibonacci, spectral norm, and typed
arithmetic stayed within 3.0% of 0.34 in the aggregate and serve as regression
controls.

## Correctness gates

Focused tests cover code shape, dynamic boundary fallback, and every fuel
boundary of small integer and string diamonds.

- All 552 Python tests pass on CPython 3.13 and 3.14.
- All 24 required official Lua 5.5.1 probes pass unchanged on both versions.
- Penlight passes 23/23, luatest 5/5, LuaCov scanner specs 24/24, and Are We
  Fast Yet 11/11 on both versions. Established native-module and unsafe-I/O
  exclusions remain unchanged.
