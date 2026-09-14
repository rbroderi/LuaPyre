# LuaPyre 0.35: remove remaining generated-Python overhead

Status: proposed implementation plan. The 0.34 runtime is committed and
published as `f9399f8a239b15603741918f8c6effaff51c5183`, tree
`8a74a8fdd3ae6aeacc47c21f6da72278951cea24`. This commit adds measurements,
a reusable code-generation audit, and the plan; it leaves the package at
`0.34.0a1`.

There is still substantial removable work before LuaPyre approaches direct
CPython execution on table, recursive, and boundary-heavy programs. 0.34 reduced
several important costs, but it did not complete the broader CFG, dataflow,
table-proof, or virtual-recursion work proposed in the preceding roadmap.

Keep generating Python source/AST and inspect its warmed specialization with
`dis(..., adaptive=True)`. Direct bytecode generation is outside this plan.
Keep Python objects and the standard library; preserve ordinary Lua semantics
and the existing explicit `integer` versus `integer_lua` contracts.

## Current distance from direct Python

The existing `python_headroom.py` compares the same algorithms with ordinary
Python functions. These references omit Lua fuel, debugging, metamethods,
deoptimization, and Lua GC, and use Python containers. Their ratios describe
the combined runtime and representation gap, not promised speedups or a
mathematical lower bound. In particular, Python's dictionary tree stores
`None` fields while Lua deletes nil-valued fields. The string comparison keeps
repeated byte-string concatenation on both sides.

| Workload | 3.13.15 LuaPyre / Python | Ratio | 3.14.7 LuaPyre / Python | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Typed arithmetic | 1.151 / 0.899 ms | 1.28× | 0.968 / 0.610 ms | 1.59× |
| Typed branch | 4.150 / 1.549 ms | 2.68× | 3.704 / 1.174 ms | 3.15× |
| Recursive Fibonacci | 33.484 / 0.811 ms | 41.29× | 29.944 / 0.806 ms | 37.14× |
| Binary trees | 118.351 / 2.520 ms | 46.96× | 110.184 / 2.533 ms | 43.49× |
| Sieve | 27.355 / 0.682 ms | 40.10× | 24.519 / 0.571 ms | 42.91× |
| Table mix | 34.599 / 3.647 ms | 9.49× | 25.120 / 2.622 ms | 9.58× |
| String building | 2.533 / 0.397 ms | 6.38× | 2.224 / 0.391 ms | 5.68× |
| Spectral norm | 31.487 / 2.214 ms | 14.22× | 27.918 / 2.649 ms | 10.54× |
| 1,000 Python calls | 0.917 / 0.057 ms | 16.10× | 1.030 / 0.044 ms | 23.30× |

Each Python version has three independent process runs with Lua/Python,
Python/Lua, Lua/Python execution order, seven warmups and 31 checked samples.
CPU affinity and `PYTHONHASHSEED=0` are fixed. Python GC is disabled only during
timing; Lua GC remains active. The table uses medians of process medians and
ratios of those aggregates, computed from unrounded values. All samples are in
[`python_headroom_034.json`](../benchmarks/results/python_headroom_034.json).
These are fresh 0.34/reference comparisons, not an A/B measurement of any
proposed 0.35 change.

## What remains in the hot paths

Separate profiles use seven warmups followed by three checked executions.
Counts below are per workload execution, averaged where necessary. The two
Python builds generally show the same helper counts. Profiling perturbs
relative costs, so its timing shares are diagnostic only. Seven warmups do not
guarantee that every rare compilation path has stabilized: the spectral profile
still records one compile failure across three executions.

| Workload | Evidence in 0.34 | Actionable implication |
| --- | --- | --- |
| Typed branch | Both arms charge 11 instructions per iteration; generated code still loads constants, checks a literal modulo divisor against zero, copies registers, and repeats Lua truth conversion | Simplify the proven body and charge equal-cost, non-exiting iterations as one batch |
| String building | About 6,000 `isinstance` calls and a generated `_state` dispatch loop remain | Carry string facts around the loop and use structured branch lowering |
| Binary trees | 15,300 generic `rawset`, 15,360 `rawset_prehashed`, 15,307 `adopt`, and 30,600 materialized call-wrapper invocations | Preserve constant keys through call continuations and reduce ownership/call work |
| Sieve | 92,559 `Frame.proto` reads, 5,001 CFG branch transitions, 25,018 pending-operation checks, and 8,087 `rawset` calls | Extend compiled coverage across the nested marking loop |
| Table mix | 60,141 `len` and 54,017 `isinstance` calls | Replace repeated access classification with valid region-entry facts |
| Spectral norm | 18,001 `_float_divide` calls; generated inner code retains numeric type dispatch and wrapping after inlining | Preserve argument/result/range facts through inlined calls and simplify the whole-function body |
| Recursive Fibonacci | 21,888 materialized entry-wrapper calls and roughly 43,800 each of list pops and appends | Reduce activation management structurally; another pool binding tweak has little demonstrated headroom |
| 1,000 Python calls | 1,000 adapters, direct runners, hook checks, and safepoints; 3,000 `len` calls | Specialize the complete fixed-arity scalar path while retaining live runtime checks |

The captured direct leaf for `x + 1` still creates `regs = [None] * 4`, reads
four initial `None` values into locals, creates `cells = {}`, and builds a
one-element result tuple. Register promotion has not removed those allocations.
In the binary-tree builder, keys loaded before recursive calls lose their
constant status because `known_constants` starts empty in each function block.
These are specific omissions in the current emitters, not hypothetical costs.

The code audit confirms that the branch loop already specializes to
`FOR_ITER_RANGE` and `BINARY_OP_ADD_INT` on both versions. CPython is specializing
operations that LuaPyre still emits too many times. On 3.14, the branch runner
contains 1,946 bytecode bytes and 22 locals; the recursive Fibonacci runner
contains 5,608 bytes and 43 locals; each major spectral multiply runner contains
11,994 bytes and 67 locals. These static sizes include cold recovery paths and
are not dynamic instruction counts or transitive memory measurements.

Evidence: [`profile_034.json`](../benchmarks/results/profile_034.json),
[`codegen_034.json`](../benchmarks/results/codegen_034.json), and the reusable
[`inspect_codegen.py`](../benchmarks/inspect_codegen.py). Same-named generated
functions can be separate code objects; do not sum only the top profile row as
though it included every compiled function.

## Ordered implementation candidates

### 1. Finish scalar AST cleanup and uniform-cost fuel batching

Start with the existing structured branch path. Substitute immutable constants
into expressions, fold a known-nonzero modulo divisor check, propagate copies,
remove dead initialization, and simplify boolean values already proved boolean.
Keep those proofs and recovery mappings in the compiler/IR; do not optimize by
matching benchmark names or register numbers.

After proving that a region cannot call, allocate Lua objects, observe hooks,
or take a semantic side exit, move uniform per-iteration fuel charges outside
the loop. The current branch benchmark has equal 11-instruction arms, so its
fully funded path can compute the exact charge from the iteration count. For
unequal paths, retain exact path accounting. For partial fuel, use the existing
precise runner or a proven complete-iteration prefix; never charge unexecuted
iterations. Preserve temporary-register values needed at every observable exit.

Targets: typed branch and typed arithmetic as a control. Files:
`structured_jit.py`, `ast_backend.py`, `jit_codegen.py`, `typed_ir_passes.py`.
Diagnostic success: the fully funded pure branch loop has no constant-list
loads, redundant truth conversions, or `used += ...` in its hot body.

### 2. Preserve facts through CFG joins and compiled-call continuations

Use fixed-point dataflow for constants, type facts, and integer ranges. Join
facts conservatively across predecessors and use widening for loops. The
current range pass clears facts at merge points; the function emitter resets
its constant map per block. Reuse the existing IR where possible instead of
adding unrelated emitter-specific trackers.

Retain immutable scalar locals across a call when they are not overwritten or
captured. Invalidate aliased heap facts and mutable upvalue facts according to
effects. First use this to keep record-key tokens across recursive calls, then
propagate guarded, validated callee return types and inlined argument ranges.
The goal is to extend prehashed writes to binary-tree child fields and remove
unnecessary numeric dispatch after spectral's inlined helper.

For string loops, establish the accumulator's type at entry and prove every
backedge preserves it before removing repeated byte-string guards. Only remove
wrapping where the existing fast-integer contract or sound range facts permit
it. A float-division experiment needs a proved finite, nonzero denominator and
the same operand conversions/evaluation order. Do not repeat the rejected
per-operation dynamic division branch.

Targets: binary trees, spectral norm, string building. Files: `typed_ir.py`,
`range_analysis.py`, `typed_ir_function_jit.py`, `function_jit.py`,
`typed_ir_jit.py`. Diagnostic success: surviving constant record writes avoid
`rawset`; proven string sites avoid repeated `isinstance`; inlined float results
do not re-enter general integer/float dispatch.

### 3. Lower longer reducible regions to Python control flow

Carry steps 1–2 into reducible `if`/`for`/`while` regions in both loop and
whole-function compilation. Start with string diamonds, then sieve's nested
marking loop, then spectral's nested multiplication loops. Sieve is sparse;
its compiled coverage must not depend on a dense-array proof.

Preserve the current fallback for irreducible CFGs and unsupported effects.
Region entry guards must dominate their uses. At every guard miss, allocation,
callback, hook boundary, and quota stop, recover the exact Lua PC, consumed
instruction prefix, registers, and pending operation. Preserve small-callee
inlining when a function promotes to its fully compiled tier. Avoid obtaining
an apparent gain by simply disabling whole-function promotion.

Targets: string building, sieve, spectral norm. Files: `cfg_value_ir.py`,
`cfg_loop_opt_jit.py`, `structured_jit.py`, `typed_ir_jit.py`, `function_jit.py`,
`super_jit.py`, `jitvm.py`. Share the lowering through the existing CFG/AST
modules where possible.
Diagnostic success: fewer executed state dispatches and interpreter returns;
substantial reductions in sieve's `Frame.proto` and branch-transition counts.
Measure first-hot execution and warmed whole functions separately.

### 4. Specialize table access from region-level representation proofs

Use a bounded first experiment: metatable-free dense-table reads in a loop with
proved in-bounds integer indices and no effect that can replace the array,
change its representation, or mutate it through an alias. Bind the array and
its length once. Next admit existing-slot primitive writes while preserving
version changes. Keep growth, holes, deletion, weak modes, and collectable writes
on paths with full bookkeeping until each has its own proof.

Reject unsupported representations before entering the optimized region.
Do not use changing table contents, a sampled index, or a previous iteration
as a permanent proof. A table's ordinary version changes on writes; checking
that same version every iteration would defeat or continually exit a writer.
Either prove stability for the bounded region or introduce a narrowly defined
representation invariant with complete invalidation coverage.

Separately measure an early return for an already-owned table in `LuaGC.adopt`.
The current traversal already stops at a same-owner root, so this can eliminate
the initial traversal list without skipping child work that used to occur.
Preserve primitive, foreign-owner, unowned graph, and remembered-set behavior.
This is a smaller binary-tree candidate, not permission to reduce GC accounting.

Targets: table mix and spectral norm; binary trees for the independent adoption
experiment. Files: `table.py`, `typed_ir.py`, `typed_ir_jit.py`,
`function_jit.py`, `gc.py`. Depends on steps 2–3 for region/alias facts.
Diagnostic success: fewer per-access `len`/`isinstance` checks and no repeated
deoptimization. Reject promptly if either Python version repeats the previous
table regressions.

### 5. Remove leftover allocations in bound scalar leaf calls

Generate direct leaves from parameters and live Python locals. Omit `regs`,
`cells`, and initial `None` assignments when neither the admitted body nor its
recovery paths need them. Reuse one direct-leaf code descriptor per Proto and
bind closure-specific values only where their identity/lifetime is guarded.

For certified fixed-arity scalar returns, test fusing the leaf expression/body
into the cached Python adapter, avoiding a second Python call and the
intermediate result tuple. A scalar ABI must distinguish Lua nil, a real return
value, and deoptimization without sentinel collisions. Keep a tuple-based ABI
for multiple results and generic calls. Extend arities/types only after the
one-integer case meets the gate; include float and mixed-arity controls before
claiming a broader improvement.

Fuel, JIT enabled state, active hooks, re-entry, max frames, argument limits,
return conversion, and GC roots/safepoints stay live on every call. Hoist only
immutable contract decisions. Validate missing/extra arguments, bool/int/float
distinctions, optional return annotations, hook callbacks, errors after a JIT
toggle, low/default fuel, and closure/target changes. Track cache retention
when bindings are discarded.

Targets: 1,000 independent Python calls plus the existing interop corpus.
Files: `dense_jit.py`, `jit_codegen.py`, `runtime.py`, `interop.py`,
`optimizing_jitvm.py`. This candidate can be developed independently of step 4.
Diagnostic success: no scratch register list or empty cell dictionary on the
admitted leaf path, then fewer Python transitions and result-tuple allocations.

### 6. Prototype lazy activations for pure scalar recursion

This is the largest remaining structural call opportunity and the highest-risk
candidate. Existing virtual frames cover effect-free leaves; recursive calls
still use materialized frames and entry wrappers. Start with fully typed,
fixed-arity scalar recursion whose only capture is a guarded call target, with
no table access, host callbacks, yields, finalizers, or mutable data captures.

Use compact logical activations and reconstruction maps so normal execution
can call a compiled scalar entry directly. Define exact stack/fuel accounting
and error unwinding before broadening admission. Reconstruct parents and the
child in correct order at deoptimization or quota failure, with original Lua
locations and no repeated call. Check Lua's configured frame limit separately
from CPython's recursion limit and fall back safely before either is exceeded.

Targets: recursive Fibonacci plus additional recursive shapes, call depths,
base-case distributions, target mutation, and failure paths. Files:
`escape_analysis.py`, `virtual_frame.py`, `function_jit.py`,
`optimizing_jitvm.py`, `gc.py`, `diagnostics.py`.
Diagnostic success: elimination of most materialized entry and pool operations
on admitted recursion. This is a separately gated prototype: if correctness or
cross-version performance fails, ship accepted steps 1–5 and carry this work
forward without widening production admission.

## Boundaries and lower-priority work

The 0.34 record documents failed structured-table/modulo and dense-table
prototypes, plus a division change that did not improve spectral norm. Another
opcode allowlist expansion or equivalent guard sequence is not a new candidate.
The recursive binding cleanup only moved Fibonacci by about 0–1% in the 0.34
A/B comparison; do not spend another tranche on that same pool adjustment.

Temporary-table scalar replacement remains a later experiment with explicit
escape, identity, GC, and materialization proofs. Binary-tree children escape
through their parents, so eliminating their allocation is not the starting
case. Broader coroutine compilation also remains useful, but needs its own
same-contract reference and resume/yield profile before inclusion. Batch-call
APIs and different algorithms such as joining accumulated strings are separate
API/algorithm changes and must not replace the current comparison corpus.

Pure arithmetic has the least measured headroom here. Once the generated body
matches the direct Python operations closely, Python object arithmetic and the
remaining Lua obligations dominate. There is no single attainable ratio for
all workloads, and an entire-runtime claim of "native Python speed" would be
premature. These references define engineering targets, not an absolute ceiling.

## Delivery and acceptance gates

Implement each candidate as an isolated change or switch against the committed
0.34 baseline. Use three paired processes in A/B, B/A, A/B order on Python 3.13
and 3.14, seven warmups, 31 checked samples, fixed hash seed and CPU affinity.
Require at least 5% elapsed-time reduction on the named target on both versions
for a performance candidate; investigate mixed/noisy results with more pairs.
Reject a repeatable regression above 5% on any important control. Record each
accepted and rejected experiment, not just the combined favorable result.

Use `speed_034_ab.py` unchanged as the initial nine-case regression corpus and
add a `speed_035_ab.py` for new admission shapes. Run `python_headroom.py` again
after accepted changes to quantify the new gap. Assert results on every run;
profile separately. Verify tier counters and code shape before and after warmup
so a promotion or compilation retry cannot masquerade as steady-state speed.
`loop_iterations` is derived from charged instructions and a nominal loop cost;
do not treat it as an exact trip count for variable-cost CFG paths.

Record cold execution and compilation latency, first-hot behavior, generated
AST/bytecode size, generic and adaptive disassembly, and retained/transient heap
size for changes that add specializations. Inspect whether expected CPython
specializations occurred, but do not require a particular opcode on unrelated
Python implementations. The audit's frame/slot counts are only an initial
retention baseline: Fibonacci currently retains 19 pooled frames/304 slots,
binary trees 18/396, and spectral norm 3/81 in these runs. Measure transitive
collectable graphs, cache lifetime, and behavior after lowering max frames.

Before release, run the existing 549-test baseline and new differential tests
on both Python versions, all 24 required unchanged official Lua 5.5.1 probes,
Penlight 23/23, luatest 5/5, LuaCov scanner 24/24, and Are We Fast Yet 11/11.
For every changed emitter, cover cold, first-hot, and warmed execution; all
fuel budgets through completion of small examples; later-iteration exits;
signed-64-bit boundaries; negative modulo; float zero/signed zero/NaN/infinity;
changing arguments and targets; aliases, metatables, weak tables, finalizers,
GC during re-entry, stack limits, and exact error locations/side effects.

Reproduce the baseline measurements from repository root with each Python
executable, using separate output paths and `--python-first` for the middle
headroom process:

```bash
PYTHONPATH=src PYTHONHASHSEED=0 python benchmarks/python_headroom.py \
  --revision f9399f8a239b15603741918f8c6effaff51c5183 --warmups 7 --repeats 31 \
  --json /tmp/headroom-034.json

PYTHONPATH=src PYTHONHASHSEED=0 python benchmarks/profile_hotpaths.py \
  --revision f9399f8a239b15603741918f8c6effaff51c5183 --warmups 7 --executions 3 \
  --json /tmp/profile-034.json

PYTHONPATH=src PYTHONHASHSEED=0 python benchmarks/inspect_codegen.py \
  --revision f9399f8a239b15603741918f8c6effaff51c5183 --include-source \
  --json /tmp/codegen-034.json
```

The commands label the source; they do not select it. Check out the stated
revision or set `PYTHONPATH` to its source tree, retaining the new audit script
from the planning checkout when inspecting 0.34.
