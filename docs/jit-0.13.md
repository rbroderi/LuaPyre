# LuaPyre 0.13 tiered JIT

LuaPyre 0.13 introduces the first executable tier above the exact register interpreter. The design goal is not to create a second Lua implementation: the interpreter remains the semantic oracle, error path, and deoptimization target.

## Execution tiers

- **Tier 0: exact interpreter.** The existing table-dispatched VM handles all LuaPyre opcodes and every fallback path.
- **Tier 1: hotness tracking.** Natural numeric-loop entries and Lua closure calls accumulate counters. Cold code stays entirely in the interpreter.
- **Tier 2: generated Python.** Supported hot regions are lowered to a small backend-neutral IR and emitted as Python functions with register access, arithmetic, table access, and loop control localized inside one Python frame.

The first 0.13 backend intentionally targets Python rather than native machine code. That lets LuaPyre prove hotness, IR, guarding, deoptimization, quota accounting, and type-specialization rules while remaining portable. A future native backend can consume the same IR instead of redefining Lua semantics.

## What is compiled in 0.13.0a1

The initial tier-2 compiler supports two high-value region types:

1. **Natural numeric `for` loops.** Straight-line loop bodies containing loads/moves, locals, numeric arithmetic, comparisons, boolean conversion, and raw table access can execute without per-opcode dispatch. `FORLOOP` itself reuses the interpreter's exact loop helper.
2. **Straight-line leaf Lua functions.** Hot non-tail closure calls with supported register operations and a fixed `RETURN` can execute directly without creating an interpreted callee loop.

Unsupported control flow, metamethod-sensitive operations, yielding calls, close/unwind behavior, or other unhandled opcodes stay in Tier 0. This is deliberate: 0.13 specializes only regions whose fallback boundary is unambiguous.

## Guards and deoptimization

Speculation is allowed only when failure can return to the interpreter before the speculative operation becomes observably incorrect.

Examples:

- generic arithmetic observed as integer arithmetic guards that both operands are still exact Python `int` values;
- raw `GETTABLE` / `SETTABLE` fast paths require a `LuaTable` with no metatable, so `__index` / `__newindex` semantics cannot be bypassed;
- unsupported numeric coercions, table shapes, or comparison categories deopt to the original opcode;
- PUC-Lua translated chunks are never given source-typing trust automatically.

Loop deoptimization records the exact LuaPyre PC at which interpretation must resume. Register mutations from earlier successfully executed operations in the same iteration are retained, and the failing operation itself is left to the interpreter.

## Optional typing is an optimization contract

LuaPyre's optional typing now has a direct runtime performance use.

The source compiler already emits specialized opcodes such as `ADD_I`, `SUB_I`, `MUL_I`, `ADD_F`, `SUB_F`, and `MUL_F` only when its type analysis has proved the operand class. Any dynamic `Any -> typed` boundary is protected by the existing `GUARD` bytecode.

Source-compiled `Proto` objects therefore carry `jit_trust_types=True`. Tier 2 may omit redundant type guards for those proven specialized opcodes. Ordinary dynamic Lua still works normally: generic arithmetic remains guarded/speculative and deoptimizes on type changes.

Binary chunks and manually constructed prototypes default to `jit_trust_types=False`, so specialized-looking bytecode from an untrusted origin does not inherit source-compiler assumptions.

This preserves one language/runtime model:

```lua
local function typed_add(a: integer, b: integer): integer
    return a + b     -- compiler emits ADD_I; JIT can trust it
end

local function dynamic_add(a, b)
    return a + b     -- generic ADD; JIT guards the observed numeric case
end
```

## Fuel and quotas

JIT execution consumes the same logical instruction budget as interpretation.

- A compiled loop knows the number of LuaPyre instructions represented by one successful iteration.
- It executes only while the remaining budget can pay for a complete iteration.
- A compiled leaf function is charged its exact fixed instruction count only after its guards succeed.
- If there is insufficient budget, execution stays in the interpreter, preserving the existing `LuaQuotaError` boundary.

`tests/test_jit.py` explicitly checks interpreter/JIT quota equivalence.

## Runtime controls

The JIT is enabled by default in 0.13:

```python
from luapyre import LuaRuntime

lua = LuaRuntime()
```

Use the exact interpreter-only path with:

```python
lua = LuaRuntime(jit=False)
```

The default hotness threshold is 32 entries/calls and can be changed for profiling or short-lived workloads:

```python
lua = LuaRuntime(jit_threshold=8)
```

Live counters are exposed as `lua.jit_stats`, including loop/leaf compilations, executions, loop iterations, deoptimizations, and rejected compile attempts.

## Benchmarking against Lua 5.5 and LuaJIT

`benchmarks/compare_runtimes.py` runs the same plain-Lua workloads through four execution modes:

- LuaPyre 0.13 JIT
- LuaPyre interpreter-only
- PUC Lua 5.5 through `lupa.lua55`
- LuaJIT through `lupa.luajit21` (or `lupa.luajit20` as fallback)

Run:

```bash
python benchmarks/compare_runtimes.py
```

For CI or a development machine where both external comparison runtimes are required:

```bash
python benchmarks/compare_runtimes.py --require-all
```

The harness compiles each workload once, performs warm-up runs, reports median/best execution times, and prints LuaPyre-JIT ratios against the interpreter, Lua 5.5, and LuaJIT. Missing optional LuaJIT modules are reported explicitly rather than silently substituted.

## Next JIT work after 0.13.0a1

The current IR intentionally leaves room for the next stages:

- internal-branch/basic-block region compilation rather than only straight-line natural loops;
- inline caches for globals, tables, and calls;
- side traces / additional versions after repeated guard failures;
- more complete typed-register facts instead of relying only on specialized opcodes;
- compiled coroutine-safe regions that cross more loop shapes without crossing yield boundaries;
- a native x86-64/AArch64 or LLVM backend once the IR/deopt contract is proven.

The rule remains: closed, proven fast paths compile; everything else keeps using the exact interpreter.