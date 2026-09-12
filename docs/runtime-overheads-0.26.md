# LuaPyre 0.26: lower trace, call, and source overhead

LuaPyre 0.26 removes three costs on the fully typed path while preserving the
same interpreter fallback, fuel accounting, stack limits, and deoptimization
state.

## Trace code generation

CFG traces now pass through the generated-AST optimization pipeline used by
the other typed tiers. Lua registers that remain live in a trace become Python
fast locals, with exact spills at side exits, returns, and budget exits.
Conservative integer range analysis also removes 64-bit wrapping only where an
operation is proved unable to overflow. The generated trace retains exact OSR
PCs and fuel checks.

## Recursive compiled calls

Each compiled function owns a small pool of inactive exact `Frame` objects.
Recursive calls allocate only to the maximum simultaneously active depth;
successful returns recycle those Frames for later calls. Keeping the pool on
the compiled specialization avoids a per-call hash lookup. Suspended, failed,
or active calls are never recycled, so tracebacks, stack overflow, and Tier 0
resumption still see the real Lua call stack.

For `fib(20)`, the first execution allocates 19 compiled Frames for its maximum
depth instead of one for every call. Later executions require no new compiled
Frames.

## Bounded source cache

`LuaRuntime` now keeps an LRU cache of compiled source chunks. Its default
capacity is 128 and `source_cache_size=0` disables it. Cache keys include the
source, chunk-name type, and chunk name. Returning the same `Proto` matters
beyond parse time: its inline caches, hot counters, and compiled tiers stay
warm across repeated `execute()` calls.

`source_cache_info` reports hits, misses, current size, and capacity.
`clear_source_cache()` discards entries and resets those counters.

## Same-machine comparison

`benchmarks/runtime_overheads_ab.py` was run on CPython 3.13 with 6 warmups
and 31 timed samples. Values are median milliseconds; lower is better.

| Workload | 0.25 | 0.26 | Change |
|---|---:|---:|---:|
| typed CFG trace | 21.283 | 17.574 | 17.4% faster |
| recursive `fib(18)` | 21.434 | 19.707 | 8.1% faster |
| 500 repeated source executions | 10.270 | 2.432 | 76.3% faster |

Packed numeric table storage is deliberately outside this release. A local
CPython probe reduced memory substantially but made indexed numeric reads and
adds 16–43% slower than lists, which conflicts with LuaPyre's speed target.
