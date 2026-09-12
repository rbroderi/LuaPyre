# LuaPyre 0.15 typed AST JIT

LuaPyre 0.15 keeps the exact Lua 5.5 interpreter as Tier 0 and makes certified
`-- luapyre: typed` source the primary optimization target. The Python backend is
now organized as a set of progressively more specialized compilation paths.

## Fast-path hierarchy

1. **Structured typed loops** compile reducible straight-line hot loops to native
   Python control flow with promoted register locals. Small static typed callees
   can be inlined into the loop body.
2. **Dense base emitters** replace compiler-side opcode `if`/`elif` cascades with
   dense tuples indexed by `Op.value`. Each opcode has its own emitter, so ADD,
   SUB, MUL, comparisons, and other specializations no longer perform secondary
   operator lookup while a hot region is compiled.
3. **Typed super-regions** represent complex/nested control flow as local jump
   lists/basic blocks, then AST-inline the block functions into one generated
   dispatcher. The jump-list representation is therefore a lowering tool, not a
   permanent Python-call cost in the executable hot path.
4. **Whole typed functions** compile larger and recursive typed functions with
   direct compiled calls while retaining real Lua frames for exact unwind,
   diagnostics, fuel accounting, and deoptimization.
5. Unsupported or dynamic shapes fail closed to the exact interpreter or the
   proven 0.13/0.14 JIT paths.

## Compiler-side dispatch

Opcode selection during Python code generation uses dense tables:

```python
emitter = LOOP_EMITTERS[item.ins.op.value]
if emitter is None or not emitter(lines, item, offset, trusted):
    # fail closed / use another tier
    ...
```

This replaces long compiler-side opcode cascades. The lookup affects JIT compile
latency; it is not executed once per Lua operation after the region has compiled.
The table is dense because `Op` is an `IntEnum` with compact values, avoiding a
hash-table lookup as well as the cascade.

Operator choice is bound into each emitter. For example ADD_I, SUB_I, and MUL_I
have distinct handlers rather than doing another dictionary lookup for `+`, `-`,
or `*` inside the compilation loop.

## AST helper inlining

Generated code is parsed to Python AST and optimized before `compile()`. The first
shared helper transform inlines direct `_i64(expression)` assignments into the
64-bit mask/sign sequence.

The transform is statement based rather than naive expression substitution. The
argument expression is assigned to a temporary exactly once before wrapping, so
side effects and Lua evaluation order cannot be duplicated.

This pattern is intended for additional tiny runtime helpers where inlining is
semantically safe and benchmark-positive. Complex semantic helpers stay as calls
unless the JIT has enough static information to replace them exactly.

## Function inlining and guards

Small non-recursive typed callees are inlined at the LuaPyre IR/bytecode level,
not by calling `inspect.getsource()` on generated Python functions. Arguments are
bound exactly once, registers are alpha-renamed/promoted, and return values are
mapped back into the caller.

No-upvalue static callees are guarded by stable `Proto` identity rather than by a
particular `Closure` object. A fresh closure can be constructed for the same
immutable child Proto on each VM run; guarding the transient object would cause a
false deoptimization on every later run.

Recursive and larger functions are compiled as direct generated functions rather
than recursively inlined without bound.

## Exactness rules

The optimization tiers must continue to preserve:

- Lua 5.5 integer wrapping and floating behavior;
- left-to-right, once-only expression evaluation;
- metatable-sensitive table behavior;
- exact fuel/quota accounting;
- precise interpreter PC/register state on deoptimization;
- Lua frame/traceback behavior for compiled calls;
- no fully-typed trust for translated binary chunks;
- fail-closed fallback for unsupported semantics.

The official Lua 5.5.1 differential/conformance baseline remains a release gate.
