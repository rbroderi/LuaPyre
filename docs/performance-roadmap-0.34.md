# Toward the Python performance limit: 0.34 and beyond

Status: proposed; no 0.34 runtime changes are implemented by this document.
Baseline: published 0.33 commit `a32af2e59200381c8cd80cebbda5ff0fd1f1344e`.
The measurements used local commit `8728eab`; both commits have tree
`2628ebcbc29b50bd35c1dc9b7838b435bfcde70d`. Publication through the authenticated
GitHub API preserved the complete file tree but assigned a new commit identity.

There is substantial remaining work within Python. Simple typed arithmetic is
already relatively close to its direct-Python reference. Branches, tables,
recursive calls, and the Python/Lua boundary still execute much more machinery.
The next pass should remove that machinery where the compiler can prove it
redundant, while preserving the current Lua contract.

The implementation constraint remains Python source/AST compiled by CPython,
Python objects, and the standard library. A native backend, Cython, Numba,
NumPy kernels, and a replacement Lua engine are outside this roadmap.

## What the measurements establish

[`python_headroom.py`](../benchmarks/python_headroom.py) compares nine workloads
with direct Python implementations of the same algorithms and checks every
result. These references deliberately omit Lua fuel, debug state, metamethods,
deoptimization, and Lua GC. Tables use ordinary Python lists or dictionaries.
They do not use closed forms, memoization, vectorized kernels, or a different
string-building algorithm. The spectral reference retains its Python helper
calls; it is not an optimized lower bound either.

The following are **cost ratios against reduced-contract references**, not
available speedups or a mathematical speed limit. The two language contracts
and table representations differ. These measurements identify where to
investigate; only production A/B measurements can establish an improvement.

| Workload | CPython 3.14.7: LuaPyre / Python | Ratio | CPython 3.13.15 ratio |
| --- | ---: | ---: | ---: |
| Typed arithmetic | 1.048 / 0.608 ms | 1.72× | 1.29× |
| Typed branch | 6.863 / 1.174 ms | 5.85× | 4.48× |
| Recursive Fibonacci | 31.175 / 0.806 ms | 38.67× | 41.65× |
| Binary trees | 134.464 / 2.537 ms | 53.00× | 48.94× |
| Sieve | 27.871 / 0.569 ms | 48.99× | 43.50× |
| Table mix | 26.187 / 2.729 ms | 9.60× | 8.90× |
| String building | 2.977 / 0.380 ms | 7.83× | 7.94× |
| Spectral norm | 29.124 / 2.698 ms | 10.79× | 14.51× |
| 1,000 Python calls | 1.572 / 0.046 ms | 34.11× | 26.70× |

Each variant has three independent process runs per Python version, seven
warmups, and 31 unprofiled samples per process. Execution order is Lua/Python,
Python/Lua, then Lua/Python. CPU affinity and `PYTHONHASHSEED=0` are fixed;
Python GC is disabled only during timing, while Lua GC remains active.
Reported times are medians of the three process medians; ratios divide those
aggregates. Raw samples, process medians, and both Python versions are retained
in [`python_headroom_033.json`](../benchmarks/results/python_headroom_033.json).

There is visible timing variation. For example, the three 3.13 spectral process
ratios are 14.03×, 28.47×, and 14.07×; the middle run is retained, and its cause
was not established. Reproduce candidate improvements with paired baseline and
candidate processes before accepting them. Do not compare these absolute times
with an earlier release run to infer a release speedup.

[`profile_033.json`](../benchmarks/results/profile_033.json) separately records
seven warmups followed by three profiled executions on CPython 3.14.7. These
instrumented self-time shares locate work, rather than forecast savings:

| Workload | Remaining cost after 0.33 |
| --- | --- |
| Typed branch | 97.9% inside generated structured control flow |
| Table mix | 80.0% inside generated code; 9.0% in `len`, 8.2% in `isinstance` |
| Spectral norm | 68.6% generated code; 9.7% in 18,001 `_float_divide` calls per run |
| Fibonacci | 39.6% generated function; 39.5% compiled call-entry wrapper; further pool/stack operations |
| Binary trees | 10.0% `rawset`, 5.2% GC threshold calculation, 5.1% key hashing, 4.5% write barriers |
| Sieve | 26.6% VM runner, 11.8% `rawset`, 6.2% `Frame.proto` access |
| String building | 6,000 `_to_lua_string` calls per run, accounting for 13.1% |
| Python calls | 32.5% `call_compiled_leaf`, 25.9% `_call_bound` |

0.33 already removed table mix's generic `type_matches` calls and spectral
norm's 18,000 virtual helper entries. Repeating those optimizations is not the
next opportunity. The substantial generated-code share means further profiles
must also inspect generated AST and warmed Python bytecode.

## Ordered 0.34 work

All five entries are hypotheses supported by code inspection and profiles.
None has a new measured optimization prototype yet. Land independently useful
changes separately; defer any candidate that fails its timing or semantic gate.

### 1. Specialize proven byte-string concatenation

Emit direct byte concatenation when both operands are proven Lua strings.
Propagate the result type through moves and branch joins so the next iteration
does not repeat generic conversion. Keep conversion and metamethod handling for
numbers, unknown values, and failed guards. This first experiment preserves the
same sequence of concatenations; replacing repeated concatenation with a builder
is a separate, more demanding transformation.

Files: `typed_ir.py`, `typed_ir_jit.py`, `ast_backend.py`, `opdispatch.py`.
Target: string building. Diagnostic success: eliminate the 6,000 conversion
helper calls on this proven-string workload without moving equivalent checks
into each operation. Check embedded NUL/non-UTF-8 bytes, numeric conversion,
`__concat`, invalid operands, exception location, and exact fuel exhaustion.

### 2. Precompute GC pacing invariants

Binary trees computes `_threshold` about 30,635 times per execution. Separate
the invariant step-size calculation from changing heap baselines. Prototype a
cached threshold first for generational mode, where the baseline depends on the
last major collection, then assess incremental mode separately.

Audit every mutation of mode, parameters, last-major size, and approximate
heap bytes. Update cached values at those mutations or retain the dynamic path
where a reliable invalidation contract is unavailable. Merely reducing the
number of collection checks would change observable behavior and is not this
optimization. Measure a guarded early return for adoption of already-owned
objects separately, after establishing its ownership invariant.

Files: `gc.py`, `gcvm.py`, `table.py`.
Targets: binary trees and sieve. Diagnostic success: fewer threshold helper
calls and less repeated arithmetic, with identical debt, pending flags,
collection boundaries, and allocation accounting. Cover parameter/mode changes,
weak tables, finalizers, cross-runtime adoption, and collection during callbacks.

### 3. Reduce work inside generated Python

Extend type and range facts through control-flow joins and integer modulo.
Use them to remove redundant type checks, proven-unnecessary 64-bit wrapping,
temporary copies, and repeated loads. Preserve wrapping whenever overflow is
possible. On proven finite, positive denominators, test direct float division
with the same operand conversions and evaluation order. Spectral norm's index
arithmetic is a candidate for that proof; the benchmark name is not a rule.

Lower reducible nested loops and branches to Python `for`/`while`/`if` inside
whole-function compilation. Carry the existing small-callee inlining through
that lowering. Sieve needs broader compiled coverage; spectral norm needs a
smaller already-compiled body. Neither should lose whole-function promotion.

Use a leaner-AST strategy throughout 0.34. Generate the smallest clear Python
AST for each admitted shape, warm the resulting function, and inspect its
specialized bytecode with `dis.dis(function, adaptive=True, show_caches=True)`.
Design emitted Python
around the specialization CPython actually selects. This captures most of the
potential benefit while leaving cache entries, jumps, exception tables, line
tables, and version-specific bytecode encoding to CPython.

Record unspecialized and warmed disassembly on both supported Python versions.
When an expected specialization does not occur, reduce the generated Python
shape and measure again. Direct bytecode generation is outside the 0.34 plan.
Reconsider it in a later roadmap only if measured evidence shows that a required
optimization cannot be expressed through AST-generated Python and the remaining
gap justifies the version-specific maintenance cost. Reducing JIT compilation
latency alone should be measured and reported separately from steady-state
execution speed.

Keep precise maps from optimized values to Lua registers and instruction PCs.
Charge a block's fuel together only when the complete path can execute safely;
on short budgets or observable exits, resume with the exact partially executed
state. Calls, allocations, callbacks, and hooks delimit what may be moved.
Dead assignments may disappear while their Lua instruction fuel still counts.

Files: `range_analysis.py`, `typed_ir.py`, `structured_jit.py`,
`function_jit.py`, `ast_backend.py`.
Targets: typed branch, spectral norm, sieve; arithmetic is a regression control.
Record generated code size, dynamic helper counts, and warmed disassembly as
well as elapsed time. Test both initial loop compilation and fully warmed
function compilation. Cover wrapping boundaries, negative modulo, zero and
signed-zero divisors, NaN/infinity, changing values, and every side exit.

### 4. Add stronger table access proofs

For a guarded dense numeric table, prove that the hot integer index stays in
the existing array and that the loop cannot mutate the representation through
an alias or callback. Bind the array once and use direct Python indexing in
the admitted region. Separate existing-slot writes from array growth; the
latter must still update table/GC bookkeeping. Preserve cache invalidation
and write barriers; omit collectable-value work only when primitive-value
facts prove it unnecessary.

For fixed record fields, investigate eliminating repeated key classification
and hashing on writes as well as reads. Binary trees still makes 30,660
`rawset` calls and 46,020 `_hash_key` calls per run. This path must preserve
`nil` deletion, key canonicalization, iteration behavior, and barriers for
the child tables stored in the records.

Files: `table.py`, `typed_ir_jit.py`, `function_jit.py`, `inline_cache.py`,
`gc.py`. Depends on the region/type facts from step 3 where appropriate.
Targets: table mix, spectral norm, binary trees, and sieve. Test holes,
array/hash migration, bool versus integer keys, metatable changes, aliases,
weak modes, re-entry, and deoptimization after a completed write.

The 0.33 structured table/modulo prototype was rejected: it did not improve
3.13 and regressed table mix by 6.2% on 3.14. Expanding an opcode allowlist and
retaining the same expensive operations is insufficient. Revisit this area
only with a materially cheaper generated access sequence.

### 5. Reduce repeated call adaptation

First, extend the current compiled-call descriptor so fixed-arity internal
call sites can bind registers and enter a runner with fewer Python wrapper
transitions. 0.33 already fused the common one-argument materialized entry;
the next experiment must remove additional work rather than rename that path.
Keep materialized Lua frames for this bounded experiment and measure changes
to wrapper, pool, and stack work independently.

Second, generate a reusable typed Python boundary adapter from the bound
closure's argument/result contract. Fuse `_call_bound` and leaf-entry work
where possible. Validate arguments once in the specialized path, while
refreshing fuel, hooks, JIT enabled state, re-entry status, stack limits, and
GC safepoints every call. Cache lifetime and invalidation must be explicit.

Files: `function_jit.py`, `optimizing_jitvm.py`, `runtime.py`, `interop.py`.
Targets: Fibonacci, binary trees, and 1,000 Python calls. Check missing/extra
arguments, strict bool/int/float distinctions, integer limits, union/container
conversion, multiple results, callback re-entry, and error ordering.
Track retained collectable objects after pooled frames return; fewer register
writes must not retain arbitrarily large dead object graphs.

## Structural work after 0.34

**Lazy materialization for recursive compiled frames.** Once the call adapters
are measured, extend virtual frames to more recursive and branching functions.
Keep a compact logical Lua stack, root descriptors, and recovery maps; create
full Frames when errors, hooks, GC, suspension, or deoptimization require them.
Distinguish Lua stack limits from Python recursion limits and preserve a safe
fallback near either boundary. This has higher potential and much higher
implementation cost than another frame-pool tweak.

**Scalar replacement for temporary tables.** Extend escape and alias analysis
to keep a small non-escaping table's fields in Python locals. Materialize at
escapes and every observable boundary. Begin with allocation-focused kernels
that prove the opportunity; binary-tree child tables generally escape through
their parents and cannot simply have allocation removed. Lua GC accounting,
weak references, finalization, and identity constrain what can be elided.

**More compiled coroutine execution.** The warmed coroutine workload still
performs about 17,329 `Frame.proto` reads and 806 `_jit_gettable` calls per run.
Investigate cached resume descriptors and longer compiled regions between
yields. Preserve close/error semantics, hook visibility, and suspended roots.
Establish a separate same-contract timing baseline before prioritizing this
over the measured table/call work.

**Application-level boundary amortization.** An explicit batch-call API or a
Lua-side loop may help applications making many tiny Python calls. Treat this
as a separate API/workload change with declared per-call fuel, errors, and
conversion semantics. It does not establish that the existing scalar API got
faster, so retain the 1,000 independent-call benchmark.

## Approaching the practical limit

There is no single Python-only ceiling across these workloads. For a fixed
algorithm and CPython build, the useful destination is generated Python close
to the direct reference, plus the smallest unavoidable Lua semantic work.
Pure arithmetic is already nearer that destination; the current table and
call results do not justify declaring the whole runtime near its limit.

Python local promotion does not turn general integer/float operations into
unboxed machine arithmetic. Object operations, container behavior, calls,
and necessary Lua guards/accounting remain costs. Algorithm transformations
and proven builtin substitutions can move the reference itself, so a ratio
against these particular Python functions is never an absolute lower bound.

Favor stable operand types and simple generated Python operations that CPython
can specialize. Use `dis` with `adaptive=True` to inspect warmed code rather
than assuming specialization occurred. Keep bytecode checks version-aware;
CPython explicitly treats bytecode as an implementation detail. See the
[Python 3.14 dis documentation](https://docs.python.org/3.14/library/dis.html)
and [PEP 659](https://peps.python.org/pep-0659/) for the underlying mechanism.

As a separate runtime experiment, benchmark CPython's own experimental JIT
when an appropriate build is available. This keeps LuaPyre's implementation in
Python, but is not a measured gain here or a prerequisite for the roadmap.
Record availability and enabled state outside hot loops and compare both
LuaPyre and the Python references. The API and availability are build/version
dependent; see [sys._jit](https://docs.python.org/3.14/library/sys.html#sys._jit).

Stop optimizing a particular kernel when several distinct, plausible reductions
fail the acceptance gate, generated code is already compact and specializes
as intended, and remaining costs are required by its contract or Python's
representation. Move effort to a workload with a demonstrated removable cost.
Record the rejected experiments so later passes do not repeat them.

## Delivery and acceptance

For each candidate, keep a separate baseline/candidate switch or checkout,
then run three paired processes in A/B, B/A, A/B order on both supported Python
versions, with seven warmups and 31 checked samples. Start with a **5% elapsed
time reduction** on the target on both versions as the acceptance target;
this is a gate, not a predicted gain. Investigate noisy or mixed results with
additional pairs. Reject a repeatable regression above 5% on any control and
assess smaller consistent regressions against the overall workload benefit.

Run differential checks against the interpreter before release. For fuel,
test all budgets through the first successful completion of small examples,
plus boundaries after later iterations and inlined operations. Exercise cold,
first-hot, and fully warmed tiers; check side effects, error locations, and
logical stack state, not just the returned value. Broader inlining/frame
admission also requires child-error recovery and target/upvalue mutation cases.

Retain the existing full test suite, all 24 required official Lua 5.5.1 probes,
and the pinned Penlight, luatest, LuaCov scanner, and Are We Fast Yet gates on
both Python 3.13 and 3.14. Record compilation latency, generated code size,
steady-state memory, and pool/cache retention alongside throughput for changes
that increase specialization. No version bump is part of this planning commit.

Reproduce a comparison from the repository root with each Python executable:

```bash
PYTHONPATH=src PYTHONHASHSEED=0 python benchmarks/python_headroom.py \
  --revision "$(git rev-parse HEAD)" --warmups 7 --repeats 31 \
  --json /tmp/python-headroom.json
```

Use a fresh process for every run, distinct output paths, and `--python-first`
for the middle run. `--case` can select an individual workload. Profile
separately with `profile_hotpaths.py`; do not time under the profiler.
