# Performance roadmap: 0.37

## Goal and actual starting point

Reduce the remaining cost of table operations, compiled calls, and control
flow while keeping LuaPyre entirely in Python. Continue generating lean
Python source/AST and inspect warmed code with `dis(..., adaptive=True)`.
Direct CPython bytecode generation is outside this plan.

The original planning baseline is `main` at
`20fb9756cb65b55f29607aacc2ae5aa7ca80a2f4`, tree
`f57d313df710d58258981202d934b66e99c6abd7`, after
[PR #45](https://github.com/rbroderi/LuaPyre/pull/45). The runtime is still
**0.35.0a1**: 0.36 added measurements, probes, and a roadmap, not a runtime
tranche. The implemented subset and measurements are now recorded in
[`speed-0.37.md`](speed-0.37.md); unfinished items below remain future work.

The [0.36 roadmap](performance-roadmap-0.36.md) remains the detailed source
for unfinished prerequisites. This plan can start from current main. If a
0.36 runtime tranche lands first, remeasure it and remove only prerequisites
that its code and tests actually satisfy. Do not count the same improvement
in both releases.

## Where the gap remains

Recorded medians from the 0.36 planning pass, running the 0.35 runtime:

| Workload | LuaPyre / Python, 3.13 | LuaPyre / Python, 3.14 | Primary investigation |
| --- | ---: | ---: | --- |
| Binary trees | 47.34× | 45.13× | Calls, record writes, allocation |
| Sieve | 40.42× | 43.58× | Nested control flow and table operations |
| Recursive Fibonacci | 41.43× | 36.83× | Compiled entries and frame materialization |
| Table mix | 8.99× | 9.76× | Repeated representation checks and writes |
| Spectral norm | 14.06× | 11.14× | Tier differences, helper calls, numeric proofs |
| Python two-integer leaf | 27.91× | 29.74× | Entry validation, argument/result containers |

Sources: [original workload samples](../benchmarks/results/python_headroom_035.json),
[focused probe samples](../benchmarks/results/speed_036_baseline.json),
[profiles](../benchmarks/results/profile_035.json), and
[generated-code audits](../benchmarks/results/codegen_036_baseline.json).
The boundary row uses the newer harness's identical per-call checking driver;
its ratio is not directly comparable to the older Python-call benchmark.

Arithmetic, branches, and string building are already approximately 1.3–1.7×
their direct Python references. Prioritize the larger gaps. These references
use the same application algorithms but omit parts of Lua's contract, including
metamethods, fuel, hooks, GC accounting, and conversion. Their timings are
comparison points, not a promise that every workload can reach 1×.

## Ordered work

Each row is a separately measured change. Finish only the prerequisite needed
for that change; avoid making every optimization depend on a complete new IR.

| Step | Deliverable | Relationship to 0.36 | Dependency |
| --- | --- | --- | --- |
| 0 | Baseline, size sweeps, and cost attribution | Reuses its 20 workloads and phase-aware harness | None |
| 1 | Efficient table traversal and deletion bookkeeping | New source-backed candidate | Step 0 |
| 2 | Facts across call continuations and longer structured regions | Explicit carryover | Step 0 |
| 3 | Scalar compiled-call interfaces and broader Python entry | New internal interface; boundary cleanup is carryover | Required call facts from step 2 |
| 4 | Table access proofs across a region | Carryover, using the shared facts | Step 2 |
| 5 | Lazy scalar recursive activations | Carryover, separately gated prototype | Proven call/recovery contract from step 3 |

### 0. Establish an attributable baseline

Keep all nine original controls and eleven 0.36 probes. Record the actual
runtime revision/tree, harness revision, Python build, input size, selected
tier, and counter changes. A revision label does not select a source checkout.
Separate setup, first execution, warmup, and steady state; add isolated JIT
compile timing before evaluating larger generated functions.

For table and call probes, add diagnostic Python variants using `LuaTable`
operations or an explicit argument/result wrapper, alongside the direct
Python algorithm. These comparisons help locate representation and wrapper
costs, but do not reproduce the entire Lua contract. Do not subtract their
timings as if all costs were independent or advertise them as native Python.

Measure allocations and retained memory separately from elapsed time. Count
entry-wrapper calls, argument/result containers, frame acquisitions, spills,
state transitions, generic table operations, and entry-list construction.
Use `dis(adaptive=True)` to check specialization after warmup; fewer opcodes
alone do not establish a speedup.

### 1. Remove repeated whole-table work from iteration and deletion

Current source exposes a candidate absent from the 0.36 probe set:
[`next_fn`](../src/luapyre/stdlib.py) builds `list(table.items())` on **every**
call and scans it for the previous key. Traversing an unchanged table of
`n` entries therefore does quadratic work. Both `rawset` and
`rawset_prehashed` in [`table.py`](../src/luapyre/table.py) also materialize the
entries when deleting a present key to record its successor. Repeated deletion
can have the same scaling problem. This is a source-derived diagnosis;
there is no measured 0.37 iteration speedup yet.

First benchmark dense, hash, and mixed tables across sizes. Prototype a
table-owned order/successor index that can be reused across `next` calls and
maintained on deletion. A cached list with a linear search is insufficient.
Target linear total work for an unchanged full traversal and a deletion sweep,
including index construction. Start with unchanged traversal; accept mutation
support as a second change only if its maintenance cost is justified.

Preserve independent/interleaved traversals, invalid-key errors, deletion of
the current key, chains of deleted successors, nil versus false, numeric key
normalization, and `pairs` metamethod behavior. Do not introduce a single
mutable cursor shared by all callers. Value overwrite must return the current
value; an index must not freeze a snapshot of values.

Audit every mutation route, including generated writes, raw writes, array/hash
migration, weak-table clearing, and supported Python access. The existing
`version` increments on every write; it cannot by itself distinguish an
unchanged layout from a changed value. Fall back for unproved mutation paths.
Test metadata lifetime, object keys, weak tables and finalizers: the collector
currently traces deleted-successor references. An index must neither retain
dead objects indefinitely nor bypass required roots/accounting.

Acceptance evidence: scaling curves and allocation counts, traversal/deletion
speed, plus unchanged-table read/write and construction controls showing the
cost of maintaining any new metadata. Do not change the user's algorithm or
public iteration API to obtain the gain.

### 2. Carry facts and structured execution across calls and nested loops

[`function_jit.py`](../src/luapyre/function_jit.py) still starts each block
with an empty `known_constants`. Implement conservative fixed-point facts
using the existing CFG/value IR: intersect at joins, widen at loops, and
invalidate heap/upvalue facts on effects. Preserve an uncaptured scalar or
constant key across a call only when its definition remains valid.

Land constant record-key propagation first, with a branch using different
keys as a negative test. Extend argument/return and numeric range facts only
with explicit proofs. Then use these facts in shared structured lowering for
nested reducible loops and call-free continuations, across both the loop and
whole-function tiers. Target `record_continuations`, `sparse_nested`, sieve,
and the `numeric_helper`/`numeric_inline` pair.

The helper/inline difference combines tier selection and type/range contracts;
hold both constant in diagnostic probes. Do not disable whole-function
promotion globally to favor one benchmark. Preserve exact fuel, PCs, error
locations, pending results, and live registers at every suspension or guard
miss. Later-iteration exits are required tests.

### 3. Remove containers and redundant work at compiled call boundaries

The whole-function CALL emitter currently packs arguments into a tuple, invokes
an entry wrapper, unpacks status/results, and extracts a scalar from the result
sequence. Real-frame entries also acquire/recycle a frame. Caching a target
does not remove these per-call costs.

Add an explicitly admitted fixed-arity scalar interface for compiled-to-compiled
calls: begin with one/two scalar arguments and one scalar result. Retain real
frames and the existing suspension path in the first experiment, so argument
and result packaging can be measured independently of frame elimination.
Then evaluate fixed two-result calls separately. Keep dynamic arity, open
results, varargs, unknown targets, and observable captures on existing paths.

Do not reinterpret `DirectCallSite` as permission to erase a frame: its current
contract proves identity only. Extend admission/effect/recovery facts
explicitly in [`call_ir.py`](../src/luapyre/call_ir.py) and
[`escape_analysis.py`](../src/luapyre/escape_analysis.py). Small pure callees
can instead be inlined when a bounded cost model proves that profitable;
benchmark calls too large for current inlining to expose the new interface.

In a separate change, finish 0.36's direct-leaf definite-assignment cleanup
and strict float/two-integer Python adapters. Reuse `python_leaf_local`,
`python_leaf_float`, and `python_leaf_binary`. Fuse an adapter and body only
if profiles after cleanup show enough removable work to pass the speed gate.

Every path retains live JIT/hook/re-entry checks, fuel and stack limits,
argument/result conversion, target mutation handling, safepoints, and roots.
Do not cache mutable runtime settings as immutable contracts. Distinguish
nil, false, empty returns, multiple returns, suspension, and DEOPT. Check
invalid arguments and int64 boundaries as well as successful calls.

### 4. Prove table representation once per admitted region

Start with 0.36's read-only dense region, then existing-slot primitive writes
through aliases. Retain bounds/representation facts across the region only
while effects prove them stable. Aliasing may invalidate a cached value even
when storage and bounds remain valid. Repeated per-access shape guards failed
the earlier speed gate; merely moving that approach into a new helper is not
a new justification.

Use step 2's key facts for record writes across recursive call continuations
before attempting a different record storage layout. Preserve write versions,
GC barriers/accounting, nil deletion, metatable fallback, and array/hash
migration. Growth, holes, weak modes, collectable writes and callbacks need
proved handling or the existing path. Coordinate any structural token with
step 1 instead of adding incompatible invalidation schemes.

Targets: `dense_read`, `dense_alias_write`, `record_continuations`, table mix,
binary trees and sieve. Attribute wins to checks, hashing, bookkeeping, or
dispatch using counters and code audits. A primitive-array microbenchmark
alone does not justify a global table representation change.

### 5. Prototype lazy scalar recursion only after recovery is proven

The recorded Fibonacci profile still includes roughly 21,888 materialized
entry-wrapper calls per execution. Existing escape analysis admits typed
lexical leaves, not general recursive frame elimination.

Begin with fixed-arity scalar self-recursion whose only capture is a guarded
recursive target. Keep activation state in Python locals and materialize exact
logical Lua frames on suspension/error when required. Count logical frames
even while physical Lua frames are absent. Guard against reaching CPython's
own recursion limit before the Lua limit, with a proven transition to the
existing execution path before Python stack exhaustion.

Compare linear and balanced recursion at multiple depths; exhaust small fuel
budgets and test lowered frame limits, target mutation, exceptions, and
resumption through every ancestor. A returned value alone is insufficient
evidence. Effectful binary-tree construction, mutual recursion, escaping
captures, and yields are later admissions, not part of the first prototype.
Reject the prototype if recovery or retained-memory costs erase the win.

## Planned probes and correctness coverage

The reusable [`speed_037_ab.py`](../benchmarks/speed_037_ab.py) harness retains
the 0.36 probes and adds table traversal/deletion cases. The code audit now
accepts `inspect_codegen.py --suite 037`, and
[`test_performance_037.py`](../tests/test_performance_037.py) supplies
deterministic semantics. The remaining rows describe coverage required when
their deferred optimizations are attempted.

| Probe family | Inputs and comparison | Required failure/mutation cases |
| --- | --- | --- |
| `next` / `pairs` scaling | Dense, hash, mixed; 32, 256, 2,048 entries; checked count and checksum | Interleaved traversal, invalid key, `__pairs`, value overwrite |
| Deletion scaling | Forward/reverse/alternating deletion; same sizes; raw and prehashed paths | Delete current/next, delete/reinsert, nil versus false, weak keys |
| Compiled scalar calls | Unary, binary, two-result callees over the inline budget; 100/1,000/10,000 calls | Target change, short fuel, wrong type, exception after earlier effects |
| Continuation/region facts | Same numeric contracts with helper versus inline body; varying trip counts | Conflicting join facts, nested early exit, later-iteration guard miss |
| Table proof lifetime | Existing 0.36 dense/record probes plus repeated value overwrite | Alias deletion, growth, metatable change, callback, GC collection |
| Recursive recovery | Existing linear/balanced probes plus varying depth | Every small fuel budget, frame limit, target mutation, ancestor resume |
| Python boundary | Existing local/float/binary probes; same checking driver on both sides | Live config changes, bool/int distinction, conversion, re-entry |

Use the same input and algorithm in A/B and Python-reference runs. Keep
scaling separate from fixed-work throughput. Do not compare a fully checked
Lua API call with an unchecked Python call in the new boundary probes.

## Acceptance and release gates

- Establish baseline and candidate in three independent paired processes per
  CPython version (3.13 and 3.14), in A/B, B/A, A/B order, with seven warmups,
  31 checked samples, fixed CPU affinity and `PYTHONHASHSEED=0`. Run elapsed-time
  measurements sequentially; profile and audit separately. Require zero
  steady-state compilation or report and resolve incomplete warmup.
- Accept an optimization only with at least 5% elapsed-time reduction on its
  named target on both versions. Investigate process spread and reject
  repeatable control regressions above 5%. For a broad call/frame or table
  rewrite, require a representative workload win as well as its isolated probe.
- Report both speedup and the remaining Python ratio. When baseline exceeds
  the same-run Python reference, also report the fraction of excess time
  removed: `(baseline - candidate) / (baseline - Python)`. Use paired current
  measurements, not a historical denominator. Never equate that fraction with
  the proportion of all Lua semantics that can be eliminated.
- Report preparation/compile cost, cold execution, generated source/bytecode
  size, allocations, retained memory, cache limits and break-even invocation
  count. A larger specialization must justify its startup and memory costs.
- Differentially test interpreter, cold JIT and warmed JIT, then every fuel
  budget through completion for small admitted shapes. Check errors, side
  effects, hooks, GC roots/finalizers and recovery, not just final results.
- Before a runtime release, run the full Python suite on both versions (576
  tests at this planning baseline, plus new tests), all 24 required official
  Lua 5.5.1 probes, Penlight 23/23, luatest 5/5, LuaCov scanner 24/24 and
  Are We Fast Yet 11/11. Preserve established exclusions.

Ship accepted changes independently and record rejected/deferred experiments
in `docs/speed-0.37.md` when implementation occurs. Bump the runtime version
only with that tranche. This plan does not claim that every candidate must
land, or that 1× direct Python is a reachable universal floor.

## Keep out of the critical path

Do not revisit the sub-5% already-owned GC-root shortcut without new profiles.
Defer custom record layouts and temporary-table scalar replacement until the
safer table/call work is measured; allocation and identity can affect GC,
weak tables and finalizers. Apply direct division only after finite/nonzero
and conversion/range proofs justify it, preserving NaN, infinities, signed
zero and error order. No memoization, changed benchmark algorithms, removed
fuel/GC/hook semantics, batch-call substitutions, native extensions, or direct
bytecode generation are performance shortcuts in this roadmap.
