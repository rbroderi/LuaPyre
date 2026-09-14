# LuaPyre 0.37 performance tranche

0.37 accepts two Python-only optimization groups from the
[0.37 roadmap](performance-roadmap-0.37.md): indexed table iteration/deletion
and broader allocation-free scalar entry from Python. It continues to emit
Python source/AST and relies on CPython's adaptive specialization; it does not
generate bytecode directly.

## Accepted changes

1. **Index table iteration order once.** `next` no longer rebuilds and scans a
   complete entry list for every returned pair. Each table lazily caches its
   current iteration keys and token positions. Independent callers pass their
   own keys, values remain live, and any uncoordinated generated write changes
   the table version and forces a safe rebuild.
2. **Keep deletion sweeps linear.** Raw and pre-hashed deletion use the same
   index to record the deleted key's successor. The index remains valid while
   removed keys are skipped, preserving LuaPyre's `next(table, deleted_key)`
   behavior without rebuilding the table per deletion. Reinsertions and layout
   changes invalidate it. GC pacing charges index allocation.
3. **Broaden scalar Python entry.** Certified one-float and two-integer,
   one-result leaves now receive positional Python scalars and return a scalar.
   They avoid argument indexing and result tuples while retaining live JIT,
   hook, re-entry, stack, fuel, conversion, overflow, safepoint and type checks.
4. **Remove unreachable direct-leaf cell work.** Direct leaf ASTs delete LOCAL
   cell-materialization branches when no cell dictionary exists. Generated
   pre-hashed deletion routes through the semantic helper so iteration
   continuation metadata is not bypassed.

## Measurements

The baseline is merged `main` at
`20fb9756cb65b55f29607aacc2ae5aa7ca80a2f4`, which contains the 0.35 runtime
and the 0.36/0.37 planning work. Each row is the median of three process
medians on an isolated CPU, using A/B, B/A, A/B ordering, seven warmups and 31
checked samples. Python GC is disabled only during steady timing; Lua GC stays
active. Full process medians are in
[`speed_037.json`](../benchmarks/results/speed_037.json).

| Target | Python | Baseline | 0.37 | Improvement | 0.37 / Python |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense `next` traversal, 2,048 entries ×4 | 3.13.15 | 2,092.126 ms | 63.961 ms | **96.9% faster** | 450.84× |
| Dense `next` traversal, 2,048 entries ×4 | 3.14.7 | 1,992.279 ms | 59.119 ms | **97.0% faster** | 532.50× |
| Hash deletion sweep, 2,048 entries | 3.13.15 | 93.027 ms | 8.619 ms | **90.7% faster** | 17.62× |
| Hash deletion sweep, 2,048 entries | 3.14.7 | 94.614 ms | 7.699 ms | **91.9% faster** | 14.26× |
| 1,000 strict-float Python calls | 3.13.15 | 1.662 ms | 1.110 ms | **33.2% faster** | 13.21× |
| 1,000 strict-float Python calls | 3.14.7 | 1.551 ms | 1.073 ms | **30.8% faster** | 13.58× |
| 1,000 two-integer Python calls | 3.13.15 | 2.161 ms | 1.205 ms | **44.3% faster** | 13.04× |
| 1,000 two-integer Python calls | 3.14.7 | 2.064 ms | 1.195 ms | **42.1% faster** | 14.83× |

The very large remaining traversal ratio is useful evidence: after eliminating
quadratic list construction/search, generic-for VM and host-function calls
dominate this reduced-contract comparison. It is not evidence that table
iteration still has quadratic scaling. Direct Python omits Lua fuel, iterator
protocol, multiple values, callability checks, and table semantics.

## Native Lua 5.5 and Python lower bounds

The reusable [`native_headroom.py`](../benchmarks/native_headroom.py) harness
also compares the nine established algorithms with native Lua 5.5 and direct
Python. Native execution uses the untyped version of the same Lua source
through Lupa 2.8's pinned `lupa.lua55` runtime. The Python functions preserve
the algorithm but omit Lua fuel, values, tables, metatables, stack/debug state,
GC, dynamic guards, and other runtime contracts; they are engineering lower
bounds, not equivalent implementations or promised attainable speeds.

Each result is the median of three process medians on one isolated CPU. The
three implementations occupy each position once in a Latin-square order, with
seven warmups and 31 checked samples. Python GC is disabled during timing while
both Lua collectors remain active. Each sample crosses Python once to invoke
the workload; the explicit 1,000-call case crosses the boundary 1,000 times.

| Workload | Python | LuaPyre | Native Lua 5.5 | Python lower bound | vs native | vs Python |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Typed arithmetic | 3.13.15 | 0.994 ms | 0.126 ms | 0.726 ms | 7.87× | 1.37× |
| Typed arithmetic | 3.14.7 | 0.722 ms | 0.126 ms | 0.489 ms | 5.73× | 1.48× |
| Typed branch | 3.13.15 | 2.215 ms | 0.263 ms | 1.460 ms | 8.43× | 1.52× |
| Typed branch | 3.14.7 | 1.856 ms | 0.301 ms | 1.143 ms | 6.16× | 1.62× |
| Recursive Fibonacci | 3.13.15 | 30.326 ms | 0.317 ms | 0.791 ms | 95.62× | 38.33× |
| Recursive Fibonacci | 3.14.7 | 24.270 ms | 0.324 ms | 0.620 ms | 74.87× | 39.17× |
| Binary trees | 3.13.15 | 102.174 ms | 1.751 ms | 2.106 ms | 58.37× | 48.51× |
| Binary trees | 3.14.7 | 86.256 ms | 1.761 ms | 1.981 ms | 48.98× | 43.53× |
| Sieve | 3.13.15 | 23.015 ms | 0.197 ms | 0.586 ms | 116.85× | 39.25× |
| Sieve | 3.14.7 | 19.681 ms | 0.194 ms | 0.509 ms | 101.48× | 38.67× |
| Table mix | 3.13.15 | 29.971 ms | 0.587 ms | 3.331 ms | 51.09× | 9.00× |
| Table mix | 3.14.7 | 22.066 ms | 0.586 ms | 2.508 ms | 37.66× | 8.80× |
| String build | 3.13.15 | 0.549 ms | 1.171 ms | 0.381 ms | **0.47×** | 1.44× |
| String build | 3.14.7 | 0.523 ms | 1.167 ms | 0.372 ms | **0.45×** | 1.41× |
| Spectral norm | 3.13.15 | 28.717 ms | 0.832 ms | 2.004 ms | 34.51× | 14.33× |
| Spectral norm | 3.14.7 | 23.595 ms | 0.836 ms | 2.407 ms | 28.21× | 9.80× |
| 1,000 Python calls | 3.13.15 | 0.675 ms | 0.200 ms | 0.051 ms | 3.38× | 13.30× |
| 1,000 Python calls | 3.14.7 | 0.598 ms | 0.197 ms | 0.041 ms | 3.04× | 14.70× |

The results split the remaining work cleanly. Straight typed loops are already
within 1.37–1.62× of reduced-contract Python, so their 5.73–8.43× native-Lua
gap is primarily the ceiling of executing the loop as Python. Recursive frame
creation and Lua table semantics dominate Fibonacci, binary trees, sieve, and
spectral norm. The Python-to-Lua boundary is about 3× native Lua but already
far below the cost of 1,000 ordinary Python calls through the full LuaPyre
contract. String construction is the exception: LuaPyre is about 2.1–2.2×
faster than native Lua for this repeated-concatenation shape.

Raw samples and per-process medians are retained in
[`native_headroom_037_313.json`](../benchmarks/results/native_headroom_037_313.json)
and
[`native_headroom_037_314.json`](../benchmarks/results/native_headroom_037_314.json).

## Rejected and deferred work

A conservative fixed-point constant analysis across whole-function blocks was
implemented as an experiment, corrected to invalidate loop-written registers,
and removed. `record_continuations` changed by about +1% on 3.13 and -7% on
3.14 in diagnostic runs, failing the cross-version 5% gate.

Longer nested regions, general table-representation proofs, scalar internal
compiled-call interfaces, and lazy recursive activations remain deferred. They
need narrower recovery/effect contracts and independent measurements; 0.37
does not label them complete. Direct bytecode generation remains out of scope.

## Validation

`benchmarks/speed_037_ab.py` retains all 0.36 cases and adds dense/hash
traversal and deletion probes. `inspect_codegen.py --suite 037` audits the new
scalar runners with adaptive disassembly; the final 3.13 and 3.14 audits are
[`codegen_037_313.json`](../benchmarks/results/codegen_037_313.json) and
[`codegen_037_314.json`](../benchmarks/results/codegen_037_314.json).
Deterministic tests cover cold and
warmed results, independent traversal, live value replacement, numeric key
normalization, invalid keys, deleted-successor chains, reinsertion, pre-hashed
deletion, scalar code shape, argument validation, JIT toggles, and the
successor-preserving generated deletion route.

- All 592 Python tests pass on CPython 3.13.15 and 3.14.7.
- All 24 required unchanged official Lua 5.5.1 probes pass on both versions.
- Penlight passes 23/23, luatest 5/5, LuaCov scanner specs 24/24, and Are We
  Fast Yet 11/11 on both versions. Established native-module and unsafe-I/O
  exclusions remain unchanged.
