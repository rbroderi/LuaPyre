# Performance roadmap: 0.36

## Starting point and scope

The accumulated 0.33–0.35 performance work is merged into `main` by
[PR #44](https://github.com/rbroderi/LuaPyre/pull/44), merge commit
`d44ab6e493870f0341c37add6821fef0127649c3`.
Its tree is `7cefbea700e1fc738096db2538afdc80a7dece72`, identical to the
published 0.35 head `4f46911324d24bf6fc4d394ba78f73dc92611766`.
Tests, upstream compatibility, and official Lua conformance CI passed before
the merge. Older superseded/squashed branches are not new release work.

This is a tests-and-plan pass, not a 0.36 runtime release. The package remains
`0.35.0a1`. 0.35 improved scalar branch/string loops and one-integer Python
entry; it did **not** complete general CFG facts, nested regions, table proofs,
or lazy recursion. Those items remain real implementation work, not completed
milestones.

Planning-pass validation: all 576 tests (552 existing plus 24 new) pass on
CPython 3.13.15 and 3.14.7. Both the existing code-audit corpus and its new
`--suite 036` route execute successfully. No runtime source file is changed.

Continue emitting Python source/AST, inspect warmed code with
`dis(..., adaptive=True)`, and use CPython's existing specialization.
Direct bytecode generation, native extensions, alternative algorithms,
memoization, and batch-call API substitutions are outside this performance
comparison. Preserve the explicit fast-`integer` contract and ordinary
`integer_lua` wrapping semantics.

## Updated distance from direct Python

Three independent processes on each CPython version, seven warmups and 31
checked samples, fixed affinity and `PYTHONHASHSEED=0`. Execution order is
Lua/Python, Python/Lua, Lua/Python. The table uses medians of process medians
and ratios computed before rounding. Python GC is disabled only while timing;
Lua GC remains active. Raw samples and counters are in
[`python_headroom_035.json`](../benchmarks/results/python_headroom_035.json).

| Workload | 3.13 Lua / Python ms | Ratio | 3.14 Lua / Python ms | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Typed arithmetic | 1.158 / 0.894 | 1.30× | 1.013 / 0.626 | 1.62× |
| Typed branch | 2.108 / 1.556 | 1.35× | 2.021 / 1.214 | 1.67× |
| Fibonacci | 33.677 / 0.813 | 41.43× | 29.892 / 0.812 | 36.83× |
| Binary trees | 116.855 / 2.469 | 47.34× | 112.339 / 2.489 | 45.13× |
| Sieve | 27.430 / 0.679 | 40.42× | 25.108 / 0.576 | 43.58× |
| Table mix | 34.112 / 3.793 | 8.99× | 25.472 / 2.609 | 9.76× |
| String building | 0.592 / 0.403 | 1.47× | 0.551 / 0.374 | 1.47× |
| Spectral norm | 31.107 / 2.212 | 14.06× | 28.801 / 2.585 | 11.14× |
| 1,000 Python calls | 0.760 / 0.056 | 13.67× | 0.806 / 0.045 | 17.89× |

Arithmetic, branches, and byte-string building are now roughly 1.3–1.7× their
reference times. Concentrate on the much larger table, recursive, nested-loop,
and call-boundary gaps before another scalar arithmetic micro-pass.

These are reduced-contract same-algorithm references, not available speedup
promises or a mathematical ceiling. Python containers omit Lua metatables,
nil deletion, weak modes, GC barriers, and versioning; Python calls omit Lua
fuel, frame reconstruction, hooks, and interop validation. Costs vary by
workload and input shape.

## New speed tests

[`speed_036_ab.py`](../benchmarks/speed_036_ab.py) adds 11 focused shapes.
Each has a direct Python reference, a fixed checked result, and interpreter /
cold JIT / warmed JIT differential coverage. Bound-call cases check every
individual return, including its exact Python type.

The harness records preparation time, first-execution time, warmup samples,
steady samples, and JIT counter deltas for each phase. Both implementations use
the same call-checking driver in boundary probes. These new ratios therefore
must not be compared directly with the older `python_calls_1000` ratio.

| New case | 3.13 Lua ms | / Python | 3.14 Lua ms | / Python |
| --- | ---: | ---: | ---: | ---: |
| `recursive_linear` | 0.193 | 50.33× | 0.183 | 56.67× |
| `recursive_balanced` | 1.820 | 38.87× | 1.787 | 44.84× |
| `record_continuations` | 8.181 | 53.08× | 8.140 | 50.19× |
| `dense_read` | 1.536 | 21.01× | 1.269 | 19.36× |
| `dense_alias_write` | 3.774 | 30.05× | 3.200 | 26.42× |
| `sparse_nested` | 4.919 | 36.86× | 4.628 | 41.59× |
| `numeric_helper` | 0.827 | 8.27× | 0.863 | 7.93× |
| `numeric_inline` | 1.107 | 13.94× | 0.975 | 11.12× |
| `python_leaf_local` | 0.884 | 7.89× | 0.921 | 8.29× |
| `python_leaf_float` | 2.071 | 23.92× | 1.964 | 21.43× |
| `python_leaf_binary` | 2.741 | 27.91× | 2.923 | 29.74× |

The same three-process protocol produced these baseline measurements; no
optimization was installed or measured as a candidate. All new cases recorded
zero compile-related counter changes during the steady samples in all six
processes. Full samples, cold/warm phases, and counters are in
[`speed_036_baseline.json`](../benchmarks/results/speed_036_baseline.json).

Preparation time includes runtime creation, parsing/source compilation, and,
for bound leaves, executing the function-producing chunk. It is not isolated
JIT compilation time. Warmup counters identify tier changes, but phase timings
do not identify the first hot instruction. Add compiler timing instrumentation
when evaluating transformations that increase compilation cost.

## Evidence and ordered implementation path

Fresh profiles of the original workloads are in
[`profile_035.json`](../benchmarks/results/profile_035.json).
New probe AST/source and generic/adaptive disassembly counts on both versions
are in [`codegen_036_baseline.json`](../benchmarks/results/codegen_036_baseline.json).
Counts below are per checked execution, dividing the three-execution profile
counts; profiling times are diagnostic, not performance estimates.

### 1. Finish direct-leaf cleanup and broaden scalar entry, separately gated

The new `python_leaf_local` audit still has `cells = {}`, six initial
`None` assignments, and an unreachable cell-allocation/spill branch. 0.35's
allocation-free claim applies to its simple expression leaf, not every admitted
leaf containing `LOCAL`. The scalar generated function is 308 bytecode bytes
on 3.13 and 342 on 3.14, with 13 locals.

Compile direct leaves from parameter locals and definite-assignment/liveness
facts. Eliminate impossible cell branches before register promotion; preserve
live recovery values and do not apply this to captured/materialized locals.
Keep frame-backed and restartable direct-leaf recovery distinct.

Then add independent scalar adapters for strict float and two-integer entries.
Their current direct runners still accept an argument sequence and return a
tuple. Test fusing the certified body into an adapter only after measuring the
allocation cleanup: the original one-integer profile attributes just ~5% of
self time to its scalar runner, so deleting that call alone may not clear 5%
elapsed-time reduction.

Targets: `python_leaf_local`, `python_leaf_float`, `python_leaf_binary`,
and original `python_calls_1000`. Files: `dense_jit.py`, `jit_codegen.py`,
`runtime.py`, `interop.py`. Require live fuel, hook, JIT toggle, re-entry,
frame-limit, conversion, overflow, GC-root and safepoint behavior. Preserve
nil versus empty/multiple returns and the distinct DEOPT sentinel.

### 2. Shared fixed-point facts across joins and call continuations

The whole-function emitter still resets `known_constants` for each block.
Binary trees still performs about 15,300 generic `rawset` calls per execution.
The record-continuation probe confirms that the problem is not limited to
one benchmark's node layout.

Build conservative CFG dataflow using existing IR and effect analysis.
Intersect constant/type facts at joins, widen ranges at cycles, and retain
uncaptured immutable scalar locals over calls unless they are overwritten.
Invalidate heap/upvalue facts at mutation or unknown calls. Never infer
invariance solely from a sampled value or a previous loop iteration.

First retain record-key tokens over recursive calls; then extend argument,
return-type, and range facts into shared numeric lowering. Keep branch paths
whose keys differ as a required negative test. Do not patch named workloads
or hard-coded registers.

Targets: `record_continuations`, binary trees, and the numerical pair.
Files: `typed_ir.py`, `cfg_value_ir.py`, `range_analysis.py`,
`function_jit.py`, `typed_ir_function_jit.py`.
Diagnostic: surviving constant record writes avoid generic `rawset`, without
skipping deletion successors, barriers, version increments or accounting.

### 3. Longer reducible regions, starting with sparse nested loops

Sieve still reads `Frame.proto` about 92,559 times and enters roughly 5,001
CFG branch transitions per run (the latter appears among 3.14's top rows).
The sparse probe retains state-dispatched AST loops. Merely adding more
opcodes to a straight-line allowlist does not solve this.

Lower a bounded nested numeric-for / while region to real Python control flow.
Use a shared lowering so normal loops, inlined bodies, and whole-function
promotion preserve the same optimized shape. Retain the existing state-machine
fallback for unsupported/irreducible control flow. Sieve admission must not
depend on dense storage.

The numerical pair is a useful diagnostic: on these fixed inputs, the Lua
helper version is faster than manually spelling its expression in the loop.
The audit shows different generated tiers (structured helper loop versus a
hot state-dispatched inline loop). Explicit parameter contracts/range facts
also differ; isolate these effects before attributing the gap to one cause.
Do not disable whole-function promotion as the production optimization.

Targets: `sparse_nested`, sieve, `numeric_helper`, `numeric_inline`,
spectral norm. Files: `structured_jit.py`, `cfg_loop_opt_jit.py`,
`typed_ir_jit.py`, `function_jit.py`, `super_jit.py`, `jitvm.py`.
All exits must reconstruct exact PC, consumed fuel prefix, registers and
pending operations. Equal-cost batching is not permission to charge skipped
instructions or hide effects.

### 4. Region-level table representation proofs

Table mix still performs about 60,141 `len` and 54,017 `isinstance` calls.
Both new dense probes retain hot state-dispatched loops. Start with a
metatable-free, read-only dense region whose integer bounds and representation
stability are proved. Bind array/length once, reject unsupported layouts at
entry, and guard contents independently when their types are not proved.

Only then admit existing-slot primitive writes and aliases. An alias can
preserve representation while invalidating a loaded value; the
`dense_alias_write` read-after-write test prevents hoisting stale contents.
Growth, holes, trailing deletion, array replacement, metamethods, weak modes,
callbacks and collectable writes keep full bookkeeping until separately
proved. The ordinary per-write version is not a stable representation token.

Targets: `dense_read`, `dense_alias_write`, table mix, spectral norm;
sparse sieve remains a mandatory negative/control workload.
Files: `table.py`, shared IR/effect passes, loop and function backends.
Do not repeat 0.34's failed dynamic dense-array guard sequence or 0.35's
sub-5% same-owner GC shortcut without a materially different hypothesis.

### 5. Numerical lowering after proofs, not dynamic divide branches

Spectral norm still calls `_float_divide` about 18,001 times per execution.
Propagate facts from step 2 through inlining and region joins before changing
this. Only emit a simpler sequence when denominator bounds prove nonzero and
the required finite/conversion/evaluation-order behavior. Check zero and signed
zero, NaN/infinity, division overflow, and integer-to-float conversion.

Targets: numerical pair and spectral norm; float interop is a control.
Files: `range_analysis.py`, shared numerical IR, and function/loop emitters.
A per-operation dynamic division branch already failed the earlier gate.

### 6. Lazy scalar recursion as an isolated prototype

Fibonacci still enters about 21,888 materialized wrappers per execution and
does roughly 43,800 list pops and appends each. The new linear and balanced
shapes also show large gaps; this is structural activation work, not another
pool-binding tweak.

Start with pure, fixed-arity scalar recursion and a guarded call-target
capture, no heap effects, host callbacks, yields, or mutable data captures.
Specify logical depth/fuel accounting and reconstruction before optimizing.
Use compact activations and lazy materialization with exact parent/child order,
call destinations, continuation PCs and error unwinding. Guard mutable targets
on every relevant entry. Respect Lua frame limits separately from CPython
recursion limits and fall back before either is exceeded.

Targets: Fibonacci, `recursive_linear`, `recursive_balanced`; binary trees
is an exclusion/control until effectful activations have their own proof.
Files: `escape_analysis.py`, `virtual_frame.py`, `function_jit.py`,
`optimizing_jitvm.py`, `gc.py`, `diagnostics.py`.
Reject the prototype if reconstruction/cross-version speed fails; do not mark
this item implemented merely because it was reviewed or deferred.

## Acceptance and safety gates

- Keep `speed_035_ab.py` unchanged as the original nine-case control corpus.
  Use the same new `speed_036_ab.py` file against baseline and candidate source
  trees. Measure each candidate separately, then the combined tranche.
- Require at least 5% elapsed-time reduction on a named target on **both**
  CPython 3.13 and 3.14, with three A/B, B/A, A/B paired processes, seven
  warmups, 31 samples, fixed CPU affinity/hash seed. Do not run competing
  benchmarks pinned to the same CPU in parallel. Investigate process spread
  and repeat any noisy result; reject repeatable control regressions over 5%.
- Report cold/preparation costs, code size, actual tier counters and rejected
  experiments alongside wins. Profile/audit separately from elapsed-time runs.
  Add isolated compilation timings and tracemalloc/retention/lifetime checks
  before accepting extra specializations. Existing pooled-slot counts are not
  transitive heap measurements or a leak test.
- The 24 new tests check all 11 references against the interpreter and cold /
  warmed JIT; harness sampling and GC restoration; every fuel budget through
  completion for three small region shapes; alias deletion/metatable fallback;
  recursive target mutation/lowered stack limits; strict scalar argument errors.
  They intentionally assert semantics, not noisy elapsed-time thresholds.
- This is baseline coverage, not an exhaustive proof of future admission.
  Every expanded emitter must add later-iteration guard misses, exact error
  locations/side effects, negative modulo, signed-64 boundaries, closure
  changes, hooks/re-entry, weak tables/finalizers and GC-root tests as applicable.
- Run the complete Python suite on both versions, all 24 required official
  Lua 5.5.1 probes, Penlight 23/23, luatest 5/5, LuaCov scanner 24/24 and AWFY
  11/11 before the 0.36 runtime release. Keep established exclusions unchanged.

## Reproduce

From this planning checkout, using each Python executable:

```bash
PYTHONPATH=src:benchmarks PYTHONHASHSEED=0 python benchmarks/speed_036_ab.py \
  --revision d44ab6e493870f0341c37add6821fef0127649c3 \
  --warmups 7 --repeats 31 --json /tmp/probes036.json

PYTHONPATH=src:benchmarks PYTHONHASHSEED=0 python benchmarks/inspect_codegen.py \
  --suite 036 --include-source \
  --revision d44ab6e493870f0341c37add6821fef0127649c3 \
  --json /tmp/codegen036.json

PYTHONPATH=src python -m pytest tests/test_performance_036_probes.py
```

Run the timing command in three processes, adding `--python-first` for the
middle one. Use repeated `--case` options to isolate a shape. `--revision`
labels the tested runtime; it does not select a checkout. For later A/B runs,
keep the harness fixed and select source with `PYTHONPATH=<checkout>/src`.
The original `python_headroom.py` and `profile_hotpaths.py` remain reusable
for the nine-case gap and broader helper-count baselines.
