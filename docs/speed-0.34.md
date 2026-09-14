# LuaPyre 0.34 performance tranche

0.34 implements the measured Python-only work from the performance roadmap.
It emits simpler Python for proven hot paths and relies on CPython's adaptive
specialization. It does not generate Python bytecode directly.

## Changes

1. **Concatenate proven strings directly.** Fully typed or guarded byte-string
   `CONCAT` sites emit Python `bytes + bytes` and avoid repeated general Lua
   string conversion.
2. **Cache GC pacing thresholds.** Allocation accounting compares against a
   cached threshold. Collection, mode changes, and public parameter changes
   refresh it explicitly.
3. **Batch proven diamond loops.** A fully funded, range-proven integer loop
   executes through one Python `range`, while short fuel budgets retain the
   exact per-path loop. Admission rejects bodies that write loop-control
   registers. Warmed disassembly on CPython 3.13 and 3.14 contains
   `FOR_ITER_RANGE` and `BINARY_OP_ADD_INT`.
4. **Pre-hash constant record writes.** Whole-function compilation stores a
   validated non-array key token and writes directly through it while retaining
   deletion successors, versioning, GC barriers, and allocation accounting.
5. **Cache typed Python entry adapters.** Bound one-integer leaf functions reuse
   a generated adapter that checks live fuel, JIT, hooks, re-entry, and stack
   state on every call. Other argument shapes preserve the generic conversion
   and diagnostic path. Recursive materialized calls also bind stable runner,
   prototype, environment-register, and pool references once.

Two broader table experiments were rejected. Adding modulo to the structured
table loop made table mix substantially slower, and dynamic dense-array access
caused frequent side exits and regressions in table mix, sieve, and spectral
norm. Direct float-division lowering did not improve spectral norm. These paths
were removed rather than retained behind optimistic heuristics.

## Measurement

`benchmarks/speed_034_ab.py` ran in three paired processes per version in A/B,
B/A, A/B order, with seven warmups, 31 checked samples, fixed CPU affinity, and
`PYTHONHASHSEED=0`. The table reports the median of the three process medians;
all raw process results are in
[`speed_034.json`](../benchmarks/results/speed_034.json).

| Workload | Python | 0.33 | 0.34 | Change |
| --- | --- | ---: | ---: | ---: |
| String building | 3.13.15 | 3.153 ms | 2.721 ms | **13.7% faster** |
| String building | 3.14.7 | 2.888 ms | 2.220 ms | **23.2% faster** |
| Binary trees | 3.13.15 | 134.371 ms | 118.305 ms | **12.0% faster** |
| Binary trees | 3.14.7 | 130.034 ms | 111.456 ms | **14.3% faster** |
| 1,000 Python calls | 3.13.15 | 1.441 ms | 0.965 ms | **33.0% faster** |
| 1,000 Python calls | 3.14.7 | 1.492 ms | 1.003 ms | **32.8% faster** |
| Typed branch | 3.13.15 | 8.037 ms | 4.321 ms | **46.2% faster** |
| Typed branch | 3.14.7 | 6.653 ms | 3.711 ms | **44.2% faster** |

Sieve improved 6.6% on Python 3.13 and 4.2% on 3.14. Table mix, recursive
Fibonacci, spectral norm, and typed arithmetic remained within 3.5% of the
baseline and serve as regression controls. The recursive-call binding cleanup
is retained as supporting work; it did not independently meet the 5% target.

## Correctness gates

The focused 0.34 tests cover direct string lowering, GC threshold refreshes,
full and partial-fuel diamond execution, warmed CPython specialization, dynamic
Python entry state, argument fallback, quota errors, and constant-key deletion
successors.

- All 549 Python tests pass on CPython 3.13 and 3.14.
- All 24 required official Lua 5.5.1 probes pass unchanged on both versions.
- Penlight passes 23/23, luatest 5/5, LuaCov scanner specs 24/24, and Are We
  Fast Yet 11/11 on both versions. Established native-module and unsafe-I/O
  exclusions remain unchanged.
