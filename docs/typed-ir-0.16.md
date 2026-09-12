# LuaPyre 0.16 typed IR optimizer

LuaPyre 0.16 makes the optimizer architecture explicit:

```text
typed Lua source
    -> LuaPyre bytecode
    -> compact statically typed IR
    -> backend-neutral optimization passes
    -> optimized Python AST
    -> CPython code object
```

The IR is the optimization contract. Python AST is a backend, not the optimizer's semantic model. This keeps the current pure-Python JIT useful while leaving the same IR available to a future native or AOT backend.

## Scope

The aggressive path remains restricted to source accepted under the fully typed contract:

```lua
-- luapyre: typed
```

Plain Lua still targets exact Lua 5.5.1 behavior. PUC/binary chunks do not inherit source type trust. Unsupported shapes fail closed to the existing exact interpreter or an earlier proven JIT tier.

0.16 does not replace the 0.15 call/recursion backend. Call-heavy and recursive typed functions continue to use that proven compiler until CALL is represented explicitly in the typed IR with equivalent frame, fuel, suspension, and deoptimization semantics.

## IR value model

The initial IR tracks four value classes:

- materialized Lua VM registers
- immutable Proto constants
- direct lexical upvalue values that can be rematerialized
- a statically stable root `_ENV`

Each lowered instruction can carry source-value facts, a proven result type, a specialization, and whether a definition can be omitted from generated code while still being reconstructed exactly on a side exit.

The IR also records the logical register state before instructions. Python AST generation uses that state to rebuild the exact Lua frame and PC when a guard fails.

## CFG propagation

0.16 performs fixed-point dataflow across basic blocks. A fact crosses a control-flow join only when every predecessor carries the same semantic IR value.

This intentionally avoids phi nodes in the first IR. Conflicting paths simply lose the fact and continue with a materialized register. The first block also has an implicit external predecessor, so a loop backedge cannot invent a value that was not valid on the first entry.

The result is SSA-like propagation where it is unambiguous without making the optimizer substantially larger.

## Table and global specialization

Lua source globals are already lowered through lexical `_ENV` plus ordinary table bytecode. The IR therefore recognizes `_ENV` plus a constant string key as a global access without adding a new Lua opcode.

Recognized constant-key operations include:

- `global_get`
- `global_set`
- `table_get_const`
- `table_set_const`

The Python backend specializes only raw `LuaTable` instances without a metatable. Anything metatable-sensitive deoptimizes to Tier 0 before the Lua operation is executed.

For stable constant-key reads, the backend uses `LuaTable.version` to validate an inline cache. Constant hash keys avoid repeated Lua key hashing. Array writes retain the proven `rawset` path because array/hash migration and trailing-nil behavior are semantically subtle.

## Loop-invariant global reads

A root `_ENV` constant-key read can be hoisted out of a compiled loop or function when the complete compiled region contains no operation that may invalidate the value.

Current barriers conservatively include table writes, global/upvalue writes, and calls/re-entry. Without alias analysis, any table write is treated as potentially aliasing `_ENV`.

The original Lua instruction still consumes its original fuel on every logical execution. Hoisting changes Python work, not Lua quota semantics.

## Dead definitions and liveness

The optimizer can virtualize constant and direct-upvalue temporaries when every backend use consumes the corresponding IR value. Their physical Lua register does not need to be written on the fast path because the value is available for exact deoptimization rematerialization.

Whole-function IR also runs backward CFG liveness for these rematerializable definitions. This fixes the common register-reuse case where an earlier constant key is dead even though the same physical register is reused later for an unrelated live value.

The first liveness pass is intentionally narrow:

- `LOADK`
- `GETUPVAL`

`LOCAL` is not generally dead-store-safe because it may update a captured cell. Arithmetic is not broadly removed yet because the initial IR does not carry an expression/value-number form capable of reconstructing an eliminated computed result at every deopt point. Those optimizations belong in a later IR expansion rather than in Python-AST peepholes.

Partial loop regions do not yet use the new backward-liveness pass because their normal side exits spill promoted locals directly. Whole functions have a closed CFG and therefore a well-defined live-out set. Loop-region DSE will widen only after explicit live-out/rematerialization metadata exists.

## Whole-function IR

0.16 adds a whole-function Python-AST backend driven by the same `TypedIRCompiler` as loop specialization. It preserves the existing real Lua frame, exact PC, and shared fuel meter.

The first whole-function IR tranche accepts non-call functions with IR-interesting table/global operations. Functions containing dynamic/direct Lua calls, recursive call edges, or unsupported child-closure shapes fall through to the proven 0.15 function compiler.

That split is deliberate: one optimizer IR now owns value/table/global decisions while the existing call machinery remains the exact implementation until call IR is ready.

## Exactness rules

The following remain release gates:

- exact Lua 5.5.1 semantics
- exact instruction fuel/quota accounting
- exact metatable behavior
- exact Lua frame/PC reconstruction on deopt
- no source-type trust for binary chunks
- fail-closed behavior for unsupported operations

An optimization that cannot reconstruct an exact Lua continuation is not admitted to the fast path.

## Performance

Same-runner CPython 3.13 measurements against merged 0.15 showed the initial 0.16 IR paths materially reducing Python-level table/global overhead:

| workload | 0.15 main | 0.16 typed IR | improvement |
|---|---:|---:|---:|
| typed global read loop | 23.44 ms | 15.97 ms | ~31.9% |
| typed constant-field loop | 33.54 ms | 24.27 ms | ~27.6% |
| typed function global read | 27.52 ms | 16.43 ms | ~40.3% |
| typed function constant field | 33.18 ms | 23.86 ms | ~28.1% |

The permanent four-way benchmark still shows native Lua and LuaJIT far ahead. The purpose of the 0.16 architecture is therefore not merely to add two table peepholes; it is to move optimization decisions into a compact typed IR where register traffic, calls, arithmetic value propagation, and eventually native lowering can be attacked coherently.

## Next IR work

Natural follow-ons are:

1. value-number/expression IR for typed integer and float operations
2. exact constant folding and copy propagation
3. broader deopt-aware DSE once computed values are rematerializable
4. explicit CALL IR with static callee identity and signatures
5. IR-level inlining of small nonrecursive typed functions
6. direct compiled call/trampoline lowering for larger or recursive typed functions
7. explicit loop live-out metadata so the same liveness/DSE machinery can safely cover partial regions

The long-term target remains a small statically typed IR compiler with optimized Python AST as the current backend, not a collection of unrelated Python-code-generation special cases.
