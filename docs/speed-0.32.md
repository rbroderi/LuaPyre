# LuaPyre 0.32 performance tranche

0.32 implements the compiled-call and virtual-frame roadmap selected after
0.31 profiling. The release keeps Lua's exact fuel, stack, error, table, and
integer behavior while reducing Python work on typed calls and numerical loops.

## Changes

1. **Cache final compiled call entries.** Each immutable Proto and call shape
   resolves once to either a virtual runner or a materialized compiled runner.
   Monomorphic generated call sites invoke that final entry directly.
2. **Generate fixed-arity adapters.** Known argument counts use direct tuple
   binding. Whole-function CALL IR selects unchecked adapters only when the
   current typed-IR state proves every actual argument; uncertain inputs keep
   runtime validation. The common one-result path avoids a length comparison.
3. **Batch pure block accounting.** Virtual-frame and whole-function emitters
   charge runs of pure operations together. Calls, guards, allocations,
   deoptimization, errors, and suspension still flush exact consumed prefixes.
4. **Spill continuation-live state.** Compiled child calls use CFG liveness plus
   active debug locals to update the parent Frame. Side exits retain full
   spills, and result slots visible to GC are cleared until the child returns.
5. **Inline small numerical upvalue callees.** Structured typed loops now admit
   stable GETUPVAL call targets and proven generic numeric arithmetic. The
   spectral-norm helper is guarded by immutable Proto identity and inlined with
   its original bytecode cost and logical stack-limit side exit.

Unsupported, dynamic, capturing, effectful, or insufficiently typed shapes
continue through the existing compiled or interpreter paths.

## Measurement

The reusable benchmark is `benchmarks/speed_032_ab.py`. Each workload validates
its result. Release measurements use CPython 3.14.7 on the same machine, three paired
processes per revision, 7 warmups, and 31 unprofiled samples per process. The
table reports the median of the three process medians; raw results are in
[`speed_032_ab.json`](../benchmarks/results/speed_032_ab.json).

| Workload | 0.31 | 0.32 | Improvement |
| --- | ---: | ---: | ---: |
| Recursive Fibonacci | 46.077 ms | 36.231 ms | **21.4% faster** |
| Spectral norm | 47.461 ms | 34.348 ms | **27.6% faster** |
| Binary trees | 181.657 ms | 165.332 ms | **9.0% faster** |
| Table mix | 27.696 ms | 28.245 ms | 2.0% slower |

Table mix is retained as a regression control; its 2.0% movement is below the
5% rejection threshold and isolated repeat measurements overlapped.

## Correctness gates

The 0.32 regression cases cover final-entry reuse, IR-proven unchecked
arguments, missing/extra/boolean arguments, every small fuel boundary for an
inlined numerical call, logical stack limits, and upvalue-loaded callee
selection. The established deoptimization, GC, traceback, integer-loop, and
0.30/0.31 performance-boundary suites remain part of the full test run.

- All 534 Python tests pass on CPython 3.13 and 3.14.
- All 24 required official Lua 5.5.1 probes pass unchanged.
- Penlight passes 23/23, luatest 5/5, LuaCov scanner specs 24/24, and Are We
  Fast Yet 11/11. The established native-module and unsafe-I/O exclusions are
  unchanged.
