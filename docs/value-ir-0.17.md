# LuaPyre 0.17 value/expression IR

LuaPyre 0.17 establishes the optimizer architecture as a **small statically typed IR compiler targeting optimized Python AST**. The Python representation is a backend; optimization semantics live in reusable IR.

The certified-source pipeline is now:

```text
-- luapyre: typed source
        ↓
LuaPyre bytecode / static type facts
        ↓
0.16 TypedIRPlan
  CFG facts / aliases / table-global specialization / deopt state
        ↓
0.17 ValueIRPlan + CALL IR
  value numbering / folds / liveness / static call identity
        ↓
constant + copy propagation / exact folding / CSE / DSE
        ↓
inline tiny pure calls OR direct real-frame static calls
        ↓
optimized Python AST backend
        ↓
CPython
```

The exact Lua 5.5.1 interpreter remains Tier 0 and the universal fallback/deoptimization oracle. The IR is intentionally independent of Python syntax so a later native backend can consume the same facts and correctness contract.

## Value graph

`ValueIRPlan` assigns stable value numbers to typed arguments, constant-pool scalar values, folded scalar literals, and pure typed expressions.

`MOVE` and `LOCAL` become aliases instead of generated assignments. Typed integer, float, boolean, and scalar operations become expression nodes. Identical expressions share a value number. The Python backend emits only nodes reachable from observable results.

This makes several optimizations consequences of the IR rather than Python peepholes:

- copy propagation is value aliasing
- constant propagation follows immutable value identities
- common-subexpression elimination is expression interning
- dead-store elimination is graph liveness
- computed-value rematerialization is the expression DAG
- constant folding replaces a pure expression with an immutable literal value

### Exact Lua integers

Integer folding and generated integer expressions preserve Lua's signed 64-bit wraparound. Python's unbounded integer arithmetic is never treated as equivalent without the explicit mask/sign correction.

## CALL IR

0.17 makes statically resolved lexical calls backend-neutral compiler facts instead of Python-AST guesses. There are two lowering policies.

### Pure inline calls

A lexical call can disappear into the value graph only when all of the following are proven:

- the function register traces to an exact child `CLOSURE`
- the closure cannot escape through a return, dynamic use, capture, table, comparison, or other observable path
- the child is `jit_fully_typed`
- the child is not vararg
- the child has no upvalues or nested children
- argument count is exact
- every argument's static type is accepted by the corresponding parameter type
- the child's reachable body itself lowers to pure value IR

The child's value DAG is cloned into the caller DAG with child arguments mapped to caller values. Folding, CSE and DSE can then cross the former function boundary.

### Direct non-inlined calls

A larger lexical child that cannot enter the pure value graph may still become a `DirectCallSite` when its identity is statically exact. The first 0.17 direct-call backend admits straight-line typed callers and fully typed fixed-argument child closures with no captures, nested children, or varargs.

Unlike pure inlining, direct lowering **does not erase Lua call semantics**:

- the caller still executes `CLOSURE` and allocates a real `Closure`
- the child receives a real Lua `Frame`
- `max_frames` is enforced by the normal frame machinery
- caller and child share the same exact fuel meter
- the caller PC advances before child execution exactly as in the proven function backend
- child suspension leaves the real child frame on the VM stack and resumes through Tier 0
- child errors/unwinds retain normal frame/trace state

The Python backend reuses `run_compiled_child`; CALL IR supplies static identity and lowering policy rather than implementing a second call stack.

Captured, recursive, dynamic, vararg, escaping, or otherwise unsupported calls fail closed to the proven 0.15/0.16 compiler/interpreter paths. Recursive direct compiled calls therefore retain the existing real-frame implementation until recursive identity itself is represented safely in CALL IR.

## Closure identity and allocation

A lexical child closure is omitted only in the pure static-inline tier where the optimizer proves that closure identity/allocation cannot be observed and the child captures no cells.

Direct CALL IR deliberately keeps real closure allocation. Any use that could make identity, capture, or lifetime observable outside the pure-inline proof either stays materialized or rejects the new tier.

## Fuel and quota accounting

Optimization never discounts Lua bytecode.

For pure value graphs, `ValueIRPlan.instruction_count` contains every reachable caller instruction plus every reachable instruction of each inlined callee. A compiled pure function runs only when the complete fixed instruction cost fits in the remaining budget. If it does not fit, the optimized function suspends before pc 0 without consuming fuel or performing side effects; Tier 0 then reaches the same quota boundary instruction-by-instruction.

Direct CALL IR uses incremental accounting. Caller work is committed to the shared meter before entering the real child frame, and the child accounts for its own instructions through the same meter. Differential tests cover nearby quota boundaries and frame-limit behavior for both inlined and direct calls.

## Fail-closed boundary

The first 0.17 value tier remains intentionally narrow. It does not value-compile branches/loops inside the pure whole-function graph, mutable table operations, generic/metamethod-sensitive arithmetic, division/modulo or other extra-error-path operations, upvalue/cell access, dynamic calls, recursive calls, varargs, dynamic multi-results, or escaping function values.

Direct CALL IR widens call classification/execution, not those value-graph assumptions. Its first caller backend is straight-line and helper/metamethod-free; complex callers continue through older exact tiers.

## Performance

Focused same-runner CPython 3.13 comparison against merged 0.16 showed:

- pure nested lexical call: **124.08 ms -> 53.10 ms**, about **57.2% faster** / **2.34x speedup**
- branchy non-inline lexical child through direct CALL IR: **113.07 ms -> 82.52 ms**, about **27.0% faster** / **1.37x speedup**

Simple leaf CSE/constant-fold microbenchmarks remain approximately flat because the older structured-loop compiler already handles those hot loop shapes efficiently. The value IR is retained because it centralizes compiler facts and enables cross-function transformations rather than because every isolated fold must beat a highly specialized older tier.

The standard 3-warmup / 7-sample four-way suite remains green. Representative 0.17 medians on CPython 3.13.15 include typed calls at **1.575 ms vs 34.535 ms interpreter**, recursive Fibonacci at **45.875 vs 110.419 ms**, Sieve at **12.733 vs 31.473 ms**, binary trees at **89.847 vs 221.302 ms**, string build at **2.479 vs 9.830 ms**, and spectral norm at **62.642 vs 223.217 ms**. Native Lua 5.5/LuaJIT remain substantially faster on most workloads, reinforcing the decision to keep the optimizer backend-neutral.

## Next compiler work

The representation can now grow without changing Lua semantics or tying optimization to Python syntax. Natural next steps are:

1. CFG-aware value graphs with explicit merge/phi values where they pay for themselves
2. guarded side-exit expression rematerialization
3. table-value and shape facts in the value graph
4. captured/static closure materialization in CALL IR
5. recursive/self-call identity in CALL IR lowered through the existing real-frame/trampoline machinery
6. broader cross-function inlining budgets and hotness weighting
7. native x86-64/AArch64 or LLVM lowering from the same typed/value/CALL IR

Exact Lua 5.5.1 behavior, quota accounting, frame limits, closure identity, and fail-closed fallback remain release gates for every extension.
