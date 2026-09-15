# LuaPyre 0.39 performance tranche

0.39 implements the two dominant paths identified by profiling typed Sieve.
Both remain Python/AST based; no CPython bytecode or native runtime component is
generated.

## Accepted changes

1. **First-iteration compiled entry.** Once a numeric loop has a cached compiled
   region, its next invocation enters that region immediately after `FORPREP`
   normalizes the loop registers. The interpreter charges `FORPREP`; generated
   code charges the body and backedge through the existing synchronized fuel
   budget. Zero-trip loops, active debug hooks, cold loops and unsupported loop
   shapes stay on the interpreter path.
2. **Sparse primitive-table regions.** A generated region may select direct
   integer dictionary storage when the observed table is hash-only, has no
   metatable or traversal state, and every access is compatible with an
   integer-to-boolean map. Runtime identity, key and value guards remain at each
   access. Generic mutation materializes tagged hash entries before applying
   the ordinary Lua array/hash, deletion and iteration rules.
3. **Primitive barrier elimination.** Direct sparse writes skip the GC barrier
   only when the runtime key is an exact integer and the value is an exact
   boolean. Version increments and allocation accounting remain intact.

## Measurements

The baseline is merged 0.38 `162193820804911f012976aa85f72cb237375347`.
Each figure is the median of three process medians, with seven warmups and 31
checked samples per process. The process was pinned to one CPU, Python hash seed
was fixed, and Python GC was disabled only during steady timing. Full samples
are retained in [`speed_039.json`](../benchmarks/results/speed_039.json); the
reusable harness is [`speed_039_ab.py`](../benchmarks/speed_039_ab.py).

| Python | 0.38 | 0.39 | Improvement | Python lower bound | 0.39 / Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3.13.15 | 22.906 ms | 8.161 ms | **64.4% faster** | 0.573 ms | 14.23× |
| 3.14.7 | 18.764 ms | 6.138 ms | **67.3% faster** | 0.512 ms | 11.98× |

The CPython 3.14 profile for ten warmed executions fell from 2,930,560 calls
and 0.810 profiled seconds to 225,751 calls and 0.142 profiled seconds.
The steady generated path eliminates all 80,870 generic `rawset` calls, 49,990
generic `rawget` calls and 80,870 table write-barrier calls seen in the former
ten-execution profile. The generated loop is now the clear remaining cost.

## Correctness gates

- Cached entry is tested against the interpreter at every fuel value through
  completion, including the exact completion boundary.
- Zero-trip loops and active debug hooks do not enter compiled code.
- Sparse storage is tested for integer/integral-float reads, iteration,
  versioning, generic materialization, dense growth, deletion and GC adoption.
- Generated code-shape coverage verifies direct dictionary access and the
  absence of the table barrier from the primitive path.

## Validation

- All 606 Python tests pass on CPython 3.13.15 and 3.14.7.
- All 24 required unchanged official Lua 5.5.1 probes pass on both versions.
- Penlight passes 23/23, luatest 5/5, LuaCov scanner specs 24/24, and Are We
  Fast Yet 11/11 on both versions. Established native-module and unsafe-I/O
  exclusions remain unchanged.
- The 0.39.0a1 wheel builds successfully.

## Remaining headroom

After these changes, general VM dispatch and table methods are no longer in the
warmed Sieve profile. Most remaining time is inside the generated Python loop:
dynamic table/key guards, `isinstance` checks, dictionary lookups and allocation
accounting. Any next Sieve work should start with region-wide guard hoisting and
safe batching of allocation accounting, while preserving exact side exits and
memory-limit behavior.
