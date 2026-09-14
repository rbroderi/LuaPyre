# Performance opportunities after 0.32

Status: investigation and implementation plan. Runtime source is unchanged.
Baseline: merged 0.32 commit `3ac664934ff200e87233b66b423ee57c2c77198c`.

Three small changes have promising unprofiled results on both supported Python
versions: expose constant type guards to the shared AST inliner, register fresh
empty tables without a general graph traversal, and specialize materialized
frame argument rebinding. These should lead the next pass.

## Evidence

[`profile_032.json`](../benchmarks/results/profile_032.json) records seven
warmups followed by three profiled executions per workload on CPython 3.14.7.
Profile call counts below are normalized to one workload execution. Self-time
shares are instrumented attribution, not predicted speedups.

[`probe_033.py`](../benchmarks/probe_033.py) contains isolated, process-local
experiments. Every benchmark execution checks its expected result. Each case
uses three baseline/probe process pairs per case and Python version, in A/B,
B/A, then A/B order,
seven warmups and 31 timed samples. CPU affinity and `PYTHONHASHSEED=0` are held
constant. Python GC is disabled only during timed samples; Lua GC remains active.
The table reports the median of each variant's three process medians.
All individual runs, including outliers, are retained in
[`probes_033.json`](../benchmarks/results/probes_033.json).

| Experiment / workload | Python 3.14 baseline → probe | Less elapsed time | Python 3.13 baseline → probe | Less elapsed time |
| --- | ---: | ---: | ---: | ---: |
| Inline constant guards / table mix | 28.504 → 25.616 ms | 10.1% | 38.423 → 34.676 ms | 9.8% |
| Register fresh empty tables / binary trees | 166.169 → 142.691 ms | 14.1% | 170.279 → 147.431 ms | 13.4% |
| Rebind one argument directly / Fibonacci | 36.078 → 34.168 ms | 5.3% | 39.259 → 36.642 ms | 6.7% |
| Prefer existing loop compilation / spectral norm | 34.104 → 33.257 ms | 2.5% | 36.313 → 34.871 ms | 4.0% |

These gains apply to different workloads and cannot be added. The probes have
not passed production conformance gates. The frame result on 3.14 is marginal:
one pair improved by only 1.8%, so require another stable A/B measurement before
accepting it. Spectral tier selection falls below the 5% acceptance target;
one 3.13 probe process also had a 65.815 ms timing outlier. Do not ship that
workload-specific selection rule.

## Ordered 0.33 implementation candidates

### 1. Finish constant guard lowering across the region backends

Table mix still calls `type_matches` 24,000 times per run. The cause is concrete:
`AstPythonJIT._emit_instruction` emits `_type_matches(consts[index], value)`,
while `SemanticFastPathOptimizer` recognizes only a literal type-name argument.
The guard is known at compilation, but this expression hides it from the
existing AST transformation.

Emit the immutable type name as a literal and reuse the shared primitive/union
guard implementation. The probe changes only that emission; it retains the
check and the existing deoptimization and fuel accounting. Apply the same
audit to virtual-frame GUARD sites and other remaining emitter variants.

Primary files: `ast_jit.py`, `ast_backend.py`, `typed_ir_jit.py`,
`virtual_frame.py`.

Gate: changing table-derived values, strict bool/int/float distinctions, unions,
metamethod fallback, and every small fuel budget must still match the interpreter.

### 2. Give fresh allocations a dedicated GC registration path

Binary trees makes 30,637 `adopt` calls per execution; adoption alone accounts
for 12.6% of profiled self time. `_new_table` constructs an empty `LuaTable`, then
uses `adopt` to walk a graph that has no outgoing edges. The prototype directly
sets ownership and accounts the same allocation and object size after the same
safepoint. It removes the traversal without postponing collection.

Implement a GC-owned registration helper for fresh empty tables. Separately
consider an early return when adopting an already-owned object, and cache the
GC threshold only with explicit invalidation for mode, parameters, and heap-size
updates. Neither extension is included in the measured fresh-table probe.

Primary files: `gcvm.py`, `gc.py`, `table.py`.

Gate: exact allocation/debt accounting, old-to-young barriers, weak tables,
finalizers, automatic GC pacing, cross-runtime ownership, and collections during
callbacks. The general adoption path must still handle incoming populated graphs.

### 3. Specialize materialized frame adapters, then fuse entry overhead

The 0.32 fixed-arity optimization reaches virtual frames; the materialized path
still loops over `param_count` and tests `len(args)` inside
`_acquire_compiled_frame`. Fibonacci makes 21,890 acquisitions and releases per
run. The remaining call-entry wrapper, acquire, and release functions together
account for roughly 49.5% of profiled self time.

Generate direct register rebinding for common arities in the final call entry,
retaining checked and IR-proven unchecked variants. The probe specializes just
the one-argument materialized case. Further fusion could remove the separate
acquire/release Python calls and repeated pool-limit calculations; measure those
independently rather than assuming their profile shares are recoverable speed.

Primary files: `function_jit.py`, `optimizing_jitvm.py`.

Gate: missing/extra arguments, invalid input ordering, child errors, suspension,
hooks/re-entry, changing max_frames, exact Lua stack depth, and inactive pool
retention. Clear stale collectable references according to ownership/liveness
facts; fewer writes must not keep large temporary graphs alive indefinitely.

### 4. Add reusable Python call adapters

For 1,000 Python→Lua calls, `LuaRuntime.call` and `call_compiled_leaf` together
consume about 51.1% of profiled self time, with more time in wrapper dispatch and
conversion. `LuaFunction.__call__` repeatedly resolves a function already bound
to a runtime, constructs generic argument containers, validates parameters, and
looks up the same leaf runner.

Cache a runtime-owned adapter keyed by closure, arity, and conversion shape.
Use direct scalar conversion/binding for common signatures. Refresh JIT state,
hooks, re-entry status, fuel, and stack limits every call, and retain exact
fallbacks for changing arguments and conversion types. This has profile evidence
but no measured prototype in this investigation.

Primary files: `interop.py`, `runtime.py`, `optimizing_jitvm.py`.

Gate: strict LuaInt range rules, unions, containers, ignored extra arguments,
callbacks, multiple results, result conversion errors, and bounded cache lifetime.

### 5. Integrate structured loops with whole-function compilation

Fresh warmed profiling corrects an overly broad interpretation of the 0.32
inlining result: spectral norm still executes **18,000 virtual helper calls per
run**. The small-call inliner exists in the loop compiler, but once `multiplyAv`
and `multiplyAtv` become hot, whole-function compilation runs their loops through
the generic CALL path. This happens after warmup with both threshold 1 and the
default threshold 32. The one-execution 0.32 inlining test does not cover this
tier transition.

The diagnostic probe refuses whole-function compilation for those two named
benchmark functions. It removes the 18,000 virtual entries but increases loop
runner entries from 2 to 602 per run and leaves the outer work in the interpreter.
Its small timing gain demonstrates why refusal alone is insufficient.

Integrate the existing pure-callee analysis and structured-loop lowering into
whole-function generation. Preserve the outer compiled execution while removing
the inner virtual calls. Then admit integer MOD in structured table-loop shapes:
it currently excludes table mix from `_STRUCTURED_OPS`, leaving 76.8% of its
profile in the generic generated region with per-instruction fuel and state work.
Spectral's 36,001 `_float_divide` calls are a subsequent specialization candidate;
any direct division must prove a nonzero divisor and preserve float conversion,
signed zero, infinity, and NaN behavior.

Primary files: `function_jit.py`, `structured_jit.py`, `super_jit.py`,
`call_ir.py`, `typed_ir_jit.py`.

Gate: test both the first hot loop and fully warmed function tier, target/upvalue
changes, exact partial fuel, logical stack limits, aliasing, wrapping integers,
strict float types, and generated code size. Never select production paths by
benchmark function name.

## Subsequent opportunities

- **Sieve and broader control flow:** 26.3% of profiled self time remains in the
  interpreter, with 5,001 CFG branch checks and 92,559 Frame.proto accesses per
  run. Extend reducible nested `while`/branch coverage with exact side exits.
  Removing one property lookup does not address the larger missing coverage.
- **Typed strings:** string build still uses 6,000 `_to_lua_string` calls per
  run (12.0% of profiled self time). Proven string operands can bypass numeric
  conversion machinery. Buffer/join transformations require a separate proof
  that intermediate values and allocation behavior cannot be observed.
- **Compact recursive activations:** defer until ordinary frame adapters and
  entry fusion have been measured. Exact reconstruction, unwind, hooks, and GC
  roots make this substantially more complex than the first three candidates.

## Acceptance and reproduction

Implement and measure each candidate separately. Require a repeatable gain
around 5% or more on a named target, no unexplained regression above 5% on the
other Python version or important controls, and the Python 3.13/3.14, official
Lua 5.5.1, and pinned upstream compatibility gates. Report cold execution,
compilation cost/code size, warmed throughput, allocations, and retained memory.
Profile attribution is for choosing work; unprofiled A/B runs decide acceptance.

```bash
PYTHONPATH=src PYTHONHASHSEED=0 python benchmarks/profile_hotpaths.py \
  --revision 3ac6649 --warmups 7 --executions 3 --json /tmp/profile032.json
PYTHONPATH=src PYTHONHASHSEED=0 python benchmarks/probe_033.py \
  --case table_guards --variant baseline --json /tmp/baseline.json
PYTHONPATH=src PYTHONHASHSEED=0 python benchmarks/probe_033.py \
  --case table_guards --variant probe --json /tmp/probe.json
```

Repeat baseline/probe processes in alternating order, and repeat with the other
supported interpreter. Available cases are `table_guards`, `table_allocation`,
`frame_rebind`, and `spectral_tiers`.
