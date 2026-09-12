# LuaPyre 0.17 value/expression IR

LuaPyre 0.17 adds an SSA-like value layer above the backend-neutral typed IR introduced in 0.16. The optimization pipeline for certified source is now:

```text
-- luapyre: typed source
        ↓
LuaPyre bytecode / typed facts
        ↓
0.16 TypedIRPlan (CFG, aliases, deopt facts, table/global specialization)
        ↓
0.17 ValueIRPlan (immutable values, expression DAG, static CALL IR)
        ↓
constant folding / copy elimination / CSE / DSE / static inlining
        ↓
Python AST backend
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

## Static CALL IR

0.17 introduces backend-neutral `StaticCallSite` records. A lexical call can be inlined only when all of the following are proven:

- the function register traces to an exact child `CLOSURE`
- the closure cannot escape through a return, dynamic use, capture, table, comparison, or other observable path
- the child is `jit_fully_typed`
- the child is not vararg
- the child has no upvalues or nested children
- argument count is exact
- every argument's static type is accepted by the corresponding parameter type
- the child's reachable body itself lowers to the pure value IR

The child's value DAG is cloned into the caller DAG, with child arguments mapped to caller values. This allows CSE and folding across the former function boundary.

Captured, recursive, dynamic, vararg, effectful, or otherwise unsupported calls fail closed to the proven 0.16/0.15 function compiler. Recursive direct compiled calls therefore keep their existing real-frame/shared-meter implementation in 0.17.

## Closure identity and allocation

A lexical child closure is omitted only in the pure static-call tier where the optimizer has proved that the closure value cannot escape and the child captures no cells. In that admitted domain, closure identity and allocation are not observable by Lua code.

Any use that could make identity, capture, or lifetime observable rejects the value tier. The normal VM then creates the closure and cells exactly as before.

## Fuel and quota accounting

Optimization never discounts Lua bytecode.

`ValueIRPlan.instruction_count` contains:

- every reachable caller instruction, including `CLOSURE` and `CALL`
- every reachable instruction of each inlined callee, including its `RETURN`

A compiled pure function runs only when the complete fixed instruction cost fits in the remaining budget. If it does not fit, the optimized function suspends before pc 0 without consuming fuel or performing side effects. Tier 0 then executes instruction-by-instruction and reaches the same `LuaQuotaError` boundary as interpreter-only execution.

The test suite differentially checks JIT and interpreter outcomes across budgets surrounding an inlined static call.

## Fail-closed boundary

The first 0.17 value tier is intentionally narrow. It does not value-compile:

- branches or loops inside the whole-function value tier
- mutable table operations
- generic/metamethod-sensitive arithmetic
- division/modulo or other operations that may introduce additional error paths
- upvalue/cell access
- dynamic calls
- recursive calls
- varargs or multi-result dynamic calls
- escaping function values

Those shapes retain the earlier structured-loop, typed-IR, direct-call/recursion, or interpreter paths.

## Performance

Focused same-runner CPython 3.13 comparison against merged 0.16 showed the new whole-function static-call path reducing a nested lexical-call workload from about **184.19 ms to 68.00 ms**, approximately **2.71× faster**.

Simple leaf-loop workloads remain roughly unchanged because the 0.15 structured-loop compiler already inlines that shape efficiently. 0.17 therefore does not replace that tier merely to claim ownership; the JIT keeps whichever exact specialization is already stronger.

## Next value-IR work

The representation is designed to grow without changing Lua semantics or tying optimization to Python syntax. Natural next steps are:

1. CFG-aware value graphs with phi/merge values where profitable
2. side-exit expression rematerialization for guarded regions
3. table-value and shape facts in the value graph
4. direct non-inlined CALL nodes for larger typed callees
5. recursive-call/trampoline lowering from CALL IR
6. broader cross-function inlining budgets and hotness weighting
7. a native backend consuming the same typed/value IR and deoptimization metadata

Exact Lua 5.5.1 behavior, quota accounting, and fail-closed fallback remain release gates for every extension.
