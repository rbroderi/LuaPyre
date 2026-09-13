# Fast integer and structured execution (0.29)

LuaPyre 0.29 keeps ordinary Lua exact while giving fully typed programs an
explicit way to opt out of repeated signed-64 wraparound work.

## Integer contracts

`integer_lua` is the inferred and exact Lua integer type. Addition,
subtraction, and multiplication wrap into the signed 64-bit domain exactly as
they do in ordinary Lua. Untyped source is unchanged.

An explicit `integer` annotation in `-- luapyre: typed` source means the program
guarantees that every intermediate value produced through that binding remains
within the signed 64-bit range. The compiler records those arithmetic sites as
no-overflow facts, and every Python JIT backend consumes the same proof. There
is no per-operation overflow check on this path; violating the contract is a
program error. Use `integer_lua` whenever wraparound is possible or intentional.

```lua
-- luapyre: typed
local exact: integer_lua = 0x7fffffffffffffff
local fast: integer = 0

for i = 1, 10000 do
    fast = fast + i
end
```

Numeric-for variables retain inferred `integer_lua` semantics. Their
representable induction values and integer literals may participate in fast
arithmetic when another operand supplies an explicit `integer` contract;
neither establishes a contract on its own. Python exposes
`LuaInt = NewType("LuaInt", int)` for annotations and checks its signed-64 range
when converting a Lua result or entering an explicitly typed integer parameter.

## Execution changes

This tranche also removes overhead around the generated code:

1. Python-to-Lua calls reuse a bounded stable trampoline, allowing call-site
   feedback and whole-function compilation to warm across calls.
2. Proven integer numeric-for loops use CPython's `range` iterator and batch
   fuel accounting instead of performing a Python budget and limit branch for
   every Lua iteration.
3. Primitive argument validation uses direct Python representation checks.
   Compiled and virtual child frames keep entry validation unless a separate
   static inlining proof establishes the callee's argument contract.
4. Structured typed loops admit table reads/writes. Stable plain Lua tables are
   guarded once, and positive dense integer reads access the Python list array
   part directly; other keys retain the exact `rawget` fallback.
5. The structured compiler now gets first refusal for supported typed table
   loops and accepts division and guarded dynamic-to-typed table reads, so the
   same fast-local lowering applies to a broader set of numeric kernels.

All optimizations retain bytecode-level fuel accounting and fail closed to an
earlier exact tier or the interpreter when a structural or table-shape guard is
not satisfied.

The pre-merge audit added regression coverage for literal/induction overflow,
changing dynamic table types, float-to-integer modulo transitions, compiled and
inlined argument checks, and reentrant Python calls. Runtime profile observations
remain guarded; only static IR facts can remove representation checks.
