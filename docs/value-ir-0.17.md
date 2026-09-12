# LuaPyre 0.17 value/expression IR

LuaPyre 0.17 adds an SSA-like value layer above the backend-neutral typed IR introduced in 0.16. The optimization pipeline for certified source is now:

```text
-- luapyre: typed source
        ↓
LuaPyre bytecode / typed facts
        ↓
0.16 TypedIRPlan (CFG, aliases, deopt facts, table/global specialization)
        ↓
0.17 ValueIRPlan + CALL IR
        ↓
constant folding / copy elimination / CSE / DSE / call classification
        ↓
inline tiny pure calls OR direct real-frame calls
        ↓
optimized Python AST backend
```

The exact Lua 5.5.1 interpreter remains Tier 0 and the universal fallback. Python AST is a backend, not the optimizer's semantic representation.

## Value graph

`ValueIRPlan` assigns stable value numbers to:

- typed function arguments
- constant-pool scalar values
- folded scalar literals
- pure typed expressions

`MOVE` and `LOCAL` become aliases rather than generated assignments. Typed integer, float, boolean, and scalar-comparison operations become expression nodes. Identical expressions share a value number, and only expressions reachable from a function result are emitted by the Python backend.

This makes several optimizations consequences of the representation rather than separate Python peepholes:

- copy propagation is value aliasing
- common-subexpression elimination is expression interning
- dead-store elimination is graph liveness
- computed-value rematerialization is the expression DAG itself
- constant folding replaces an expression with an immutable literal node

### Exact Lua integers

Integer folding and generated integer expressions preserve Lua's signed 64-bit wraparound. Python's unbounded integer arithmetic is masked and sign-corrected rather than trusted directly.

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

The child's value DAG is cloned into the caller DAG, with child arguments mapped to caller values. Folding, CSE and DSE can then cross the former function boundary.

### Direct non-inlined calls

A larger lexical child that cannot enter the pure value graph may still become a `DirectCallSite` when its identity is statically exact. The first 0.17 direct-call backend admits straight-line typed callers and fully typed fixed-argument child closures with no captures, nested children, or varargs.

Unlike pure inlining, direct lowering **does not erase Lua call semantics**:

- the caller still executes `CLOSURE` and allocates a real `Closure`
- the child receives a real Lua `Frame`
- `max_frames` is enforced by the normal frame machinery
- caller and child share the same exact fuel meter
- the caller PC is advanced before child execution exactly as in the proven 0.15 function backend
- child suspension leaves the real child frame on the VM stack and resumes through Tier 0
- child errors/unwinds retain the normal frame/trace path

The Python backend reuses `run_compiled_child`; CALL IR provides the static identity/classification, not a second call stack.

Captured, recursive, dynamic, vararg, escaping, or otherwise unsupported calls still fail closed to the proven 0.15/0.16 compiler/interpreter paths. Recursive direct compiled calls therefore retain their existing real-frame/shared-meter implementation until recursive identity itself is represented safely in CALL IR.

## Closure identity and allocation

A lexical child closure is omitted only in the pure static-inline tier where the optimizer proves that the closure value cannot escape and the child captures no cells. In that admitted domain, closure identity and allocation are not observable by Lua code.

Direct CALL IR deliberately keeps real closure allocation. Any use that could make identity, capture, or lifetime observable outside the pure-inline proof either stays materialized or rejects the new tier.

## Fuel and quota accounting

Optimization never discounts Lua bytecode.

For pure value graphs, `ValueIRPlan.instruction_count` contains every reachable caller instruction plus every reachable instruction of each inlined callee. A compiled pure function runs only when the complete fixed instruction cost fits in the remaining budget. If it does not fit, the optimized function suspends before pc 0 without consuming fuel or performing side effects, and Tier 0 reaches the same quota boundary instruction-by-instruction.

Direct CALL IR uses incremental accounting instead: caller work is committed to the shared meter before entering the real child frame, and the child accounts for its own instructions through the same meter. Differential tests cover every nearby quota boundary for both inlined and direct calls.

## Fail-closed boundary

The first 0.17 value tier remains intentionally narrow. It does not value-compile:

- branches or loops inside the pure whole-function value graph
- mutable table operations
- generic/metamethod-sensitive arithmetic
- division/modulo or other operations that introduce additional error paths
- upvalue/cell access
- dynamic calls
- recursive calls
- varargs or multi-result dynamic calls
- escaping function values

Direct CALL IR widens only call classification/execution, not those value-graph assumptions. Its first caller backend is straight-line and helper/metamethod-free; complex callers continue through older exact tiers.

## Performance

Focused same-runner CPython 3.13 comparison against merged 0.16 on the final implementation direction showed:

- pure nested lexical call: about **124.08 ms -> 53.10 ms**, roughly **57.2% faster** / **2.34x speedup**
- branchy non-inline lexical child through direct CALL IR: about **113.07 ms -> 82.52 ms**, roughly **27.0% faster** / **1.37x speedup**

Simple leaf CSE/constant-fold microbenchmarks remain approximately flat because the older structured-loop compiler already handles those hot loop shapes efficiently. The value IR is retained because it centralizes optimization facts and enables cross-function transformations rather than because every isolated fold is faster by itself.

The standard 3-warmup / 7-sample four-way suite remains green. Native Lua 5.5/LuaJIT are still substantially faster on most workloads, which is why the optimizer remains backend-neutral instead of treating generated Python as the final architecture.

## Next value-IR work

The representation can now grow without changing Lua semantics or tying optimization to Python syntax. Natural next steps are:

1. CFG-aware value graphs with explicit merge/phi values where profitable
2. side-exit expression rematerialization for guarded regions
3. table-value and shape facts in the value graph
4. broaden direct CALL IR to captured/static closures with exact cell materialization
5. represent recursive/self-call identity in CALL IR and lower it through the existing frame/trampoline machinery
6. broader cross-function inlining budgets and hotness weighting
7. native x86-64/AArch64 or LLVM lowering from the same typed/value/CALL IR

Exact Lua 5.5.1 behavior, quota accounting, frame limits, closure identity, and fail-closed fallback remain release gates for every extension.
