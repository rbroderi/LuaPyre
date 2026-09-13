# LuaPyre 0.14: fully typed optimization mode

LuaPyre 0.14 keeps ordinary Lua 5.5 source as the semantic compatibility target and adds an explicit source contract for code that wants to be the primary optimization target.

## Source cookie

Fully typed mode is enabled by this comment on the first or second physical source line:

```lua
-- luapyre: typed
```

The placement rule deliberately mirrors Python's encoding-cookie convention. A cookie after the second line is just an ordinary comment. An unknown `-- luapyre: ...` mode in the first two lines is a compile-time error so a typo cannot silently select different semantics.

The cookie is not a promise that LuaPyre will ignore type errors. It is the opposite: it asks the compiler to certify a stronger invariant before the chunk may execute.

## Fully typed contract

For a source chunk accepted in fully typed mode:

- lexical locals must have a concrete non-`Any` type after inference or annotation;
- local initializers with statically known types are inferred; since 0.29,
  `local total = 0` becomes exact-wraparound `integer_lua`, while an explicit
  `integer` annotation opts into the no-overflow contract;
- function parameters and return values must be explicitly typed;
- typed varargs must have an explicit type;
- implicit globals are disabled;
- ambient globals must be declared with a concrete type, for example `global math: table` or `global print: function`;
- `global *` is rejected;
- dynamic values such as table reads may enter a typed binding only through an explicit annotation, where LuaPyre emits the existing runtime `GUARD` at that boundary;
- integral numeric `for` variables use `integer_lua`; induction values may
  participate in an explicit operand's `integer` contract, and numeric float
  loops use `float`;
- typed generic-`for` iterator contracts are not defined yet, so generic `for` is rejected in fully typed mode for now.

Example:

```lua
-- luapyre: typed
global input: table

local total: integer = 0
for i = 1, 10000 do
    local value: integer = input[i] -- one checked dynamic -> typed boundary
    total = total + value           -- integer specialization from here on
end
return total
```

This is intentionally stricter than LuaPyre's ordinary optional annotations. Plain Lua and gradually typed Lua continue to work without the cookie.

## Why this can be faster

Dynamic Lua requires an optimizer to prove or speculate about types, metatables, and control-flow behavior before removing checks. Fully typed mode moves part of that proof to compile time.

Each accepted source `Proto` is marked `jit_fully_typed=True`, in addition to the existing `jit_trust_types` marker used for source-proven specialized arithmetic. This gives later tiers a durable optimization contract rather than having to rediscover the same facts from profiling every run.

The expected optimization path is:

1. infer and validate type facts once while compiling source;
2. emit specialized bytecode for statically proven operations;
3. keep guards only at genuinely dynamic boundaries;
4. compile larger hot regions using those facts;
5. eventually lower the same typed IR to a native backend without changing the language contract.

The fully typed marker does **not** make arbitrary table contents or host values magically typed. Those remain dynamic until checked at a boundary. This is important for exact semantics and safe embedding.

## 0.14 internal-branch regions

0.13 compiled only straight-line natural numeric loop bodies. 0.14 adds a generated-Python region backend for acyclic internal `if`/`else` control flow.

The compiler may now quicken a supported branchy numeric loop to `JFORLOOP`. At the hot threshold the region JIT:

- lowers the body into basic blocks;
- accepts forward internal `JMP`, `JMPIF`, and `JMPIFNOT` edges;
- emits one generated-Python function for the whole loop region;
- dispatches at basic-block boundaries rather than returning to the VM for every branch;
- preserves raw-table fast paths only when no metatable can affect the operation;
- keeps dynamic type/table guards fail-closed and deoptimizes to the exact interpreter on a miss.

Nested internal loops are deliberately not region-compiled yet because their backward edges would turn the current acyclic region into a second interpreter. They remain valid Lua and execute in Tier 0/Tier 1.

## Exact fuel accounting

Branch paths can have different instruction counts. A compiled region therefore does not charge a fixed cost per iteration.

Before entering a compiled iteration, 0.14 requires enough remaining fuel to cover the longest possible path through the acyclic region. If the budget is too close to the quota boundary, execution returns to the interpreter, which consumes instructions one at a time. While compiled, the runner returns the actual dynamic number of instructions executed on the path taken.

This preserves the quota contract while still allowing large batches of hot iterations to run as generated Python.

## Benchmark target

The permanent four-way benchmark now has three workload groups:

- `micro` for dispatch/JIT kernels;
- `typed` for direct fully-typed optimization probes;
- `algorithm` for larger recursive, table, string, numeric, and allocation-heavy programs.

Algorithm workloads may carry two source spellings: fully typed LuaPyre source and equivalent standard Lua for Lua 5.5/LuaJIT. Both are required to return the same result. This lets us make fully typed LuaPyre the performance target without weakening the native reference comparison.

Run all workloads with:

```bash
python benchmarks/compare_runtimes.py --require-all
```

or isolate the optimization target with:

```bash
python benchmarks/compare_runtimes.py --group typed --require-all
python benchmarks/compare_runtimes.py --group algorithm --require-all
```

## Optimization policy going forward

LuaPyre should optimize in this order:

1. preserve exact Lua 5.5 behavior for ordinary source and the interpreter/deoptimization path;
2. make fully typed source as fast as the architecture allows;
3. let ordinary dynamic Lua opportunistically reach the same optimized paths when profiling proves equivalent facts;
4. never require speculative dynamic-Lua machinery on a fact that fully typed source already proved statically.

This means new native/region/inlining work should be designed against the fully typed contract first, then generalized to guarded dynamic Lua where useful.
