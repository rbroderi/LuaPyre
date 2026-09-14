# LuaPyre 0.33 performance tranche

0.33 implements the measured opportunities found after the 0.32 merge. It
reduces table allocation, guard, compiled recursion, Python entry, and warmed
small-callee costs while preserving the same interpreter fallbacks.

## Changes

1. **Complete constant guard lowering.** Region, whole-function, and virtual
   frame emitters expose immutable type names to the shared AST guard inliner.
2. **Register fresh empty tables directly.** New tables retain the same
   safepoint, ownership, generation, byte accounting, and GC debt without a
   general object-graph traversal.
3. **Fuse recycled one-argument entries.** Cached materialized call entries bind
   and release the common recursive frame shape directly while retaining exact
   validation, stack, suspension, and pool limits.
4. **Cache bound Python leaf adapters.** `LuaFunction` retains its compiled leaf
   descriptor. In-range typed integer arguments reuse their existing tuple and
   primitive results take the exact scalar conversion path. Dynamic fuel,
   hooks, re-entry, stack limits, and GC safepoints are checked on every call.
5. **Inline safe callees after whole-function promotion.** Stable upvalue-loaded,
   fully typed, straight-line callees with no captures or children are guarded
   by immutable Proto identity and spliced into warmed whole functions. Their
   bytecode fuel and logical stack checks remain explicit. A numeric division
   uses direct Python division only when its divisor is a proven nonzero
   constant and the numerator is converted to float first.

The attempted structured table/modulo path was removed after it failed the
cross-version performance gate.

Further measured bottlenecks, comparisons with direct Python, and proposed
work are in [`performance-roadmap-0.34.md`](performance-roadmap-0.34.md).

## Measurement

The reusable benchmark is `benchmarks/speed_033_ab.py`. Release measurements use
three paired processes per version in A/B, B/A, A/B order, seven warmups, and 31
unprofiled samples. Every workload execution validates its result. The table
reports the median of three process medians; raw results are retained in
[`speed_033.json`](../benchmarks/results/speed_033.json).

| Workload | Python | 0.32 | 0.33 | Improvement |
| --- | --- | ---: | ---: | ---: |
| Table mix | 3.13.15 | 37.084 ms | 34.319 ms | **7.5% faster** |
| Table mix | 3.14.7 | 27.945 ms | 25.490 ms | **8.8% faster** |
| Binary trees | 3.13.15 | 169.339 ms | 138.143 ms | **18.4% faster** |
| Binary trees | 3.14.7 | 161.218 ms | 131.962 ms | **18.1% faster** |
| Recursive Fibonacci | 3.13.15 | 38.941 ms | 34.779 ms | **10.7% faster** |
| Recursive Fibonacci | 3.14.7 | 35.641 ms | 30.581 ms | **14.2% faster** |
| Spectral norm | 3.13.15 | 35.962 ms | 31.951 ms | **11.2% faster** |
| Spectral norm | 3.14.7 | 33.979 ms | 28.754 ms | **15.4% faster** |
| 1,000 Python calls | 3.13.15 | 2.661 ms | 1.418 ms | **46.7% faster** |
| 1,000 Python calls | 3.14.7 | 2.706 ms | 1.407 ms | **48.0% faster** |

Arithmetic moved by +4.6% on Python 3.13 and -1.0% on 3.14. A higher-sample
five-pair rerun of the initially noisy branch control measured 1.1% faster on
Python 3.13 and 0.5% slower on 3.14, within the 5% rejection threshold.

## Correctness gates

The 0.33 regression suite covers literal guard generation, fresh-table GC
accounting, virtual and materialized argument errors, changing fuel, cached
bound entries, warmed whole-function inlining, constant division, and every
small fuel outcome around an inlined call. Existing stack, debug hook,
deoptimization, GC, coroutine, interop, and compatibility suites remain gates.

- All 542 Python tests pass on CPython 3.13 and 3.14.
- All 24 required official Lua 5.5.1 probes pass unchanged on both versions.
- Penlight passes 23/23, luatest 5/5, LuaCov scanner specs 24/24, and Are We
  Fast Yet 11/11 on both versions. Established native-module and unsafe-I/O
  exclusions remain unchanged.
