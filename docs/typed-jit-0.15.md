# LuaPyre 0.15 typed AST JIT

LuaPyre 0.15 keeps the exact Lua 5.5 interpreter as Tier 0 and makes certified
`-- luapyre: typed` source the primary optimization target. The Python backend is
organized as progressively more specialized compilation paths.

## Fast-path hierarchy

1. **Structured typed loops** compile reducible hot loops to native Python
   control flow with promoted register locals. Small static typed callees can be
   inlined into the loop body.
2. **Dense base emitters** replace compiler-side opcode `if`/`elif` cascades with
   dense tuples indexed by `Op.value`. Each opcode has its own emitter, so ADD,
   SUB, MUL, comparisons, and other specializations do not perform secondary
   operator lookup while a hot region is compiled.
3. **Typed super-regions** represent complex/nested control flow as local jump
   lists/basic blocks, then AST-inline the block functions into one generated
   dispatcher. The jump-list form is a lowering representation, not a permanent
   Python-call cost in the executable hot path.
4. **Semantic AST fast paths** remove small Python helper/method calls when the
   generated AST has enough information to reproduce Lua semantics exactly.
5. **Whole typed functions** compile larger and recursive typed functions with
   direct compiled calls while retaining real Lua frames for exact unwind,
   diagnostics, fuel accounting, and deoptimization.
6. Unsupported or dynamic shapes fail closed to the exact interpreter or the
   proven earlier JIT tiers.

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
Operator choice is bound into each emitter, so ADD_I, SUB_I, MUL_I and similar
operations have distinct handlers instead of another operator dictionary lookup.

## AST semantic lowering

Generated code is parsed to Python AST and optimized before `compile()`.

### Integer wrapping

Direct `_i64(expression)` assignments become an inline 64-bit mask/sign sequence.
The expression is first assigned to a temporary, so it is evaluated exactly once
and side effects cannot be duplicated.

### Equality and type guards

For local/constant operands, `_lua_equal(a, b)` is lowered to exact Lua equality
logic in Python AST: booleans stay distinct from numbers, integer/float numeric
values compare numerically, strings/nil compare by same-type value, and reference
values use identity. Literal `_type_matches()` guards are similarly lowered to
direct exact Python type tests when the requested type is statically understood.
Unknown type expressions keep the proven helper path.

### Dense tables

Hot typed array-like tables can bypass `LuaTable.rawget/rawset` for positive
integer keys in the dense array part. The fast path updates `LuaTable.version`
exactly and retains the original methods for nil deletion, gaps, float/NaN keys,
hash migration, and every unsupported shape.

A hash-part shape guard is deliberately checked first. Sparse or mixed tables
therefore go immediately to the exact `rawget/rawset` implementation rather than
paying dense-array checks before the fallback. This matters for algorithms such
as Sieve, while dense workloads such as table mixing retain direct array access.
`rawlen()` lowers directly to `len(table.array)`.

### Numeric FORLOOP

The dense straight-line loop tier no longer calls `vm._forloop` at each backedge.
It emits the same mechanics directly: signed-64-bit overflow termination for
integer loops, directional limit checks, and the exact float stepping path.
Fuel cost and exit PC behavior are unchanged. Structured/super-region loops
already used direct backedge mechanics; this makes the base dense JIT consistent.

## Function inlining and guards

Small non-recursive typed callees are inlined at the LuaPyre IR/bytecode level,
not by calling `inspect.getsource()` on generated Python functions. Arguments are
bound exactly once, registers are alpha-renamed/promoted, and return values are
mapped back into the caller.

No-upvalue static callees are guarded by stable `Proto` identity rather than by a
particular `Closure` object. Recursive and larger functions are compiled as
direct generated functions rather than recursively inlined without bound.

## Benchmark impact

On the standard CPython 3.13.15 / Ubuntu 24.04 four-way benchmark with 3 warmups
and 7 timed samples, the semantic-fast-path tranche retained exact results and
improved several already-JIT-compiled workloads relative to the preceding 0.15
head:

| workload | previous 0.15 | semantic fast paths | change |
|---|---:|---:|---:|
| tables | 15.742 ms | 15.016 ms | ~4.6% faster |
| typed branch | 36.351 ms | 33.754 ms | ~7.1% faster |
| table mix | 74.113 ms | 64.152 ms | ~13.4% faster |
| string build | 4.580 ms | 4.398 ms | ~4.0% faster |
| spectral norm | 121.037 ms | 115.932 ms | ~4.2% faster |

Sieve remains effectively at the previous level (23.864 ms vs 24.107 ms) after
the sparse-table bypass; the first unconditional dense-table version had
regressed it to 25.828 ms and was not kept.

## Exactness rules

The optimization tiers must continue to preserve:

- Lua 5.5 integer wrapping and floating behavior;
- left-to-right, once-only expression evaluation;
- metatable-sensitive table behavior;
- exact table versioning and raw table semantics;
- exact fuel/quota accounting;
- precise interpreter PC/register state on deoptimization;
- Lua frame/traceback behavior for compiled calls;
- no fully-typed trust for translated binary chunks;
- fail-closed fallback for unsupported semantics.

The official Lua 5.5.1 differential/conformance baseline remains a release gate.
