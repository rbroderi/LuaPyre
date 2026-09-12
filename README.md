# LuaPyre

LuaPyre is a clean-slate Lua runtime written in Python. It targets **Lua 5.5.1** semantics, a sandbox-first embedding model, and optional gradual type annotations that feed runtime optimization without creating a second language/runtime.

**Python 3.13+** · **current pre-alpha: 0.22.0a1**

LuaPyre is not yet a complete Lua 5.5.1 implementation. Correct semantics come first; the runtime is built around a register VM and explicit Lua frames, with a guarded tiered JIT that specializes proven hot paths and deoptimizes back to the same interpreter.

## Language model

Plain Lua remains dynamic. Optional annotations add static guarantees and optimization facts:

```lua
local dynamic = get_value()       -- Any
local count: integer = 10

local function add(a: integer, b: integer): integer
    return a + b
end
```

Missing annotations mean `Any` in ordinary LuaPyre source. Typed and untyped code use the same compiler, bytecode, VM, tables, closures, standard library, and JIT.

Lua 5.5 declarations and LuaPyre annotations can be combined:

```lua
local limit: integer <const> = 10
global answer: integer
answer = 42
```

### Fully typed optimization mode

LuaPyre 0.14 added an explicit source contract for code that wants maximum optimization. Put this cookie on the first or second physical line:

```lua
-- luapyre: typed
```

In fully typed mode, locals with statically known initializers are inferred, function parameters/returns must be typed, implicit globals are disabled, and a lexical binding may not silently remain `Any`. Dynamic table/host values can still enter typed code through an explicit typed binding, where the compiler emits a runtime guard at that boundary.

```lua
-- luapyre: typed
global input: table

local total = 0
for i = 1, 10000 do
    local value: integer = input[i]
    total = total + value
end
return total
```

Accepted chunks carry a stronger `jit_fully_typed` optimization contract. **Fully typed LuaPyre source is the primary performance target going forward.** Ordinary Lua remains the Lua 5.5 semantic/compatibility target and can opportunistically use the same optimized paths when runtime guards prove equivalent facts. See [`docs/typed-jit-0.14.md`](docs/typed-jit-0.14.md).

### Typing and JIT specialization

The source compiler emits specialized integer/float bytecode only when it has proved those operand classes, and existing `GUARD` instructions protect dynamic `Any -> typed` boundaries. The tiered JIT can therefore omit redundant type guards for source-proven specialized operations.

Ordinary untyped Lua remains speculative: generic arithmetic is specialized from observed values and deoptimizes to the interpreter if a guard stops matching. PUC-Lua translated chunks do not inherit source-compiler type trust.

## Python embedding

```python
from luapyre import LuaRuntime

lua = LuaRuntime()
lua.expose("double_from_python", lambda n: n * 2)

result = lua.execute("""
local function add(a: integer, b: integer): integer
    return a + b
end

local x = double_from_python(10)
return add(x, 22)
""")

assert result == 42
```

Host capabilities are explicit. A default runtime has no ambient filesystem, networking, process execution, Python import/eval, `io`, `os`, `debug`, native-library loading, or Python object introspection.

### JIT controls

The guarded tiered JIT is enabled by default:

```python
lua = LuaRuntime()
```

Use the exact interpreter-only path with `LuaRuntime(jit=False)`. The default hotness threshold is 32 loop entries/calls and can be changed with `jit_threshold=`. Live counters are available through `lua.jit_stats`; `lua.jit_feedback` snapshots adaptive call/table cache states and deoptimization reasons.

0.13 introduced generated-Python straight-line numeric-loop and leaf-function compilation. 0.14–0.20 built the typed/value/CALL/CFG pipeline, exact deopt rematerialization, dominance, cyclic SSA, LICM, guard hoisting, and induction recognition. 0.21 added adaptive call/table PICs and deoptimization feedback. **0.22 adds profile-guided trace-shaped CFG compilation, hot side exits, and exact on-stack replacement from live Frame/PC/register state.** Typed branch targets become OSR entries after their edge reaches the hot threshold; trace guards return to the exact alternate PC, and source-bytecode fuel remains authoritative. Python AST remains a backend rather than the optimizer's semantic representation. See [`docs/trace-osr-0.22.md`](docs/trace-osr-0.22.md) and the earlier design notes in [`docs/`](docs/).

### Output and warnings

Lua `print` is safe and available by default. It applies Lua `tostring` semantics, including `__tostring`, uses tab separators, appends a newline, and sends the already-formatted bytes to an output sink. The default sink delegates to Python `print`.

```python
captured = []
lua = LuaRuntime(output=captured.append)
lua.execute('print("hello", 42)')
assert captured == [b"hello\t42\n"]
```

`warn` has a separate byte-oriented sink and preserves Lua 5.5 `@off` / `@on` controls:

```python
messages = []
lua = LuaRuntime(warning=messages.append)
lua.execute('warn("hello")')
assert messages == [b"hello"]
```

Sinks can be changed later with `set_output_sink()` and `set_warning_sink()`.

### Safe modules

The default runtime exposes `package` and `require`, but only the in-memory preload searcher is enabled. There is no ambient filesystem search and no C/native loader.

```python
lua = LuaRuntime()
lua.preload("answer", "return { value = 42 }")
assert lua.execute("return require('answer').value") == 42
```

`LuaRuntime.preload()` accepts Lua source text/bytes, an existing LuaPyre Lua/host function, or a Python callable. `package.loaded`, `package.preload`, `package.searchers`, and `package.searchpath` are available. `package.cpath` is empty and `package.loadlib` is intentionally absent.

### Explicit file loading

`loadfile`, `dofile`, and the Lua-file `require` searcher appear only when the embedder provides a file-loader capability:

```python
files = {
    "answer.lua": b"return 42",
    "game/vector.lua": b"return { x = 1, y = 2 }",
}

lua = LuaRuntime(file_loader=files.get)
assert lua.execute("return dofile('answer.lua')") == 42
assert lua.execute("return require('game.vector').x") == 1
```

The loader receives a logical UTF-8 name and may return `bytes`, `str`, or `None`. LuaPyre never turns that name into a Python filesystem operation itself. Applications can back the capability with a real filesystem, virtual filesystem, zip/archive, database, package resources, or anything else they choose.

The capability can be changed dynamically with `set_file_loader()`. Removing it removes `loadfile`, `dofile`, and the file searcher again while leaving preload-only `require` available.

See [`docs/embedding-0.11.md`](docs/embedding-0.11.md) for the embedding contract introduced in 0.11; 0.17 does not broaden the default host-capability boundary.

## Implemented runtime semantics

LuaPyre currently includes substantial Lua 5.5 behavior, including:

- lexical scopes, closures, recursive locals, and shared mutable upvalues
- multiple returns and last-expression expansion
- Lua 5.5 named varargs
- lexical `_ENV`
- signed 64-bit integer behavior and Lua numeric/table-key semantics
- metatables and the core indexing/call/arithmetic/bitwise/comparison metamethod families
- `do`, `repeat`, `break`, numeric/generic `for`, `goto`, and labels
- proper Lua tail calls by explicit frame replacement
- local `<const>` and `<close>` plus reverse-order `__close` unwinding
- Lua 5.5 global declarations and strict-global behavior after explicit declaration
- persistent Lua threads/coroutines, including nested yields, wrapping, status, close, and pending `<close>` cleanup
- Lua-aware weak tables, ephemerons, finalizers, resurrection, and GC-observable reachability
- text loading and LuaPyre-native binary `string.dump` / `load`
- validated PUC-Lua 5.5 binary chunk input translated into LuaPyre bytecode
- source/chunk names, line information, structured host tracebacks, and PUC debug-line reconstruction
- selected Lua-style field/global/upvalue attribution in runtime errors
- guarded generated-Python JIT execution for supported hot loops, nested/branch regions, typed calls, recursive functions, typed-IR table/global regions, pure value-IR whole functions, and direct static CALL-IR functions
- SSA-like typed value numbering with exact scalar folding, CSE/DSE, static non-escaping lexical-call inlining, and real-frame direct lexical calls
- opt-in fully typed source certification for aggressive optimization

The VM uses explicit Lua frames rather than Python recursion for ordinary Lua calls. ASTs are compile-time only and are never interpreted directly.

## Safe standard library

The default safe environment includes the deterministic/in-memory portions of the Lua 5.5 libraries:

- base functions such as `assert`, `error`, `type`, `tostring`, `tonumber`, `select`, `pcall`, `xpcall`, `pairs`, `ipairs`, `next`, `rawget`, `rawset`, `rawlen`, `getmetatable`, `setmetatable`, `load`, `print`, `warn`, and `collectgarbage`
- `coroutine`
- `string`, including native Lua patterns, formatting, `%q`, packing/unpacking, and `string.dump`
- `table`, including Lua 5.5 `table.create`
- `math`, including Lua-compatible explicit-seed xoshiro256** random sequences
- `utf8`
- safe `package` / `require` with preload-only loading by default

`io`, `os`, `debug`, C/native module loading, and ambient file loading are intentionally excluded from the default sandbox.

Standard-library callbacks are currently synchronous continuation boundaries. Lua callbacks from facilities such as `__pairs`, `__tostring`, `table.sort`, protected calls, and pattern replacements work, but yielding through those native-library callback boundaries is still rejected. Full C/API-style yieldable continuations are future work.

## Binary chunks

LuaPyre keeps its own VM serialization separate from PUC bytecode:

- `string.dump` emits a bounded, non-pickle LuaPyre-native serialization of LuaPyre prototypes
- `load` can reload those chunks through normal binary/text mode checks
- PUC-Lua 5.5 binary chunks are a separate interoperability input path
- PUC chunks are validated and translated to LuaPyre register bytecode before execution
- malformed, truncated, version/format-mismatched, or unsupported chunks fail as Lua load errors instead of being trusted as host data

This separation preserves the custom optimizer/JIT architecture while allowing compiled Lua 5.5 code to enter through `load`.

## Garbage collection semantics

Python remains responsible for physical memory reclamation, but LuaPyre maintains a separate Lua-level reachability model for observable GC behavior. It traces Lua roots, uses bytecode liveness rather than stale physical register contents, implements weak values/keys/all-weak tables and ephemerons, and models finalization/resurrection ordering.

Collection is currently explicit/deterministic at `collectgarbage` or `LuaRuntime.collect()` safe points. Automatic byte-debt pacing for Lua's incremental/generational collectors is not yet modeled; `collectgarbage("count")` is an approximate Lua-reachable-memory estimate.

## Lua 5.5.1 conformance

Normal CI runs on Python 3.13 and 3.14 and installs Lupa 2.8+, explicitly selecting `lupa.lua55` as the fast PUC-Lua 5.5 differential oracle.

For exact micro-release work, `tools/official_551.py` pins the official `lua-5.5.1-tests.tar.gz` archive to SHA-256:

```text
da07b543872dc0bb2ff12aabd0c248578d78df3eb6b67efdc537a46d455c7f31
```

The harness bounds archive/download/extraction sizes, rejects traversal paths and links/devices, classifies the upstream suite by dependency type, and runs selected files in fresh LuaPyre runtimes. Its read-only suite access is supplied through the same production `file_loader` capability used by embedders; it no longer replaces `package`, `require`, `loadfile`, `dofile`, or `print` with test-only Lua implementations.

The committed release gate contains seven unchanged upstream files: `bwcoercion.lua`, `pm.lua`, `tpack.lua`, `vararg.lua`, `bitwise.lua`, `math.lua`, and `utf8.lua`. Additional official files remain explicit probes until their semantic gaps are fixed; the baseline is never weakened by copying or patching upstream tests, skipping assertions, or marking failures as expected passes.

See [`docs/conformance-0.12.md`](docs/conformance-0.12.md) for the exact conformance tranche. The 0.17 value/CALL-IR work must preserve that same exact gate for ordinary Lua.

The complete official suite does **not** pass yet. Some upstream tests depend on Lua's internal C test API, debug/io/os/native-module facilities, allocator details, or stress behavior that is outside the default sandbox. Other failures identify genuine remaining Lua semantics and are tracked through explicit suite runs.

## Benchmarking

`benchmarks/compare_runtimes.py` is the permanent cross-runtime harness:

```bash
python benchmarks/compare_runtimes.py --require-all
```

It compares:

- LuaPyre JIT
- LuaPyre interpreter-only
- PUC Lua 5.5 through `lupa.lua55`
- LuaJIT through `lupa.luajit21` (falling back to `lupa.luajit20` when appropriate)

The corpus contains `micro`, `typed`, and `algorithm` groups. Algorithm workloads include recursive Fibonacci, Sieve, binary trees, table mixing, string construction, and spectral norm. Where LuaPyre uses typed annotations, the native engines receive an equivalent standard-Lua spelling and all backends must produce the same result.

```bash
python benchmarks/compare_runtimes.py --group typed --require-all
python benchmarks/compare_runtimes.py --group algorithm --require-all
```

Use `--json PATH` for a versioned machine-readable report. The permanent **Four-way runtime benchmark** Actions workflow records text and JSON artifacts. See [`benchmarks/README.md`](benchmarks/README.md) for methodology and comparison guidance.

## Current compatibility gaps

Major remaining work includes:

- broader official Lua 5.5.1 suite coverage
- additional exact runtime-error / `getobjname` categories
- automatic incremental/generational GC pacing
- yieldable native-library / C-API continuation semantics
- fuller userdata/C-API behavior beyond the current sandbox value model
- host-facing `io`, `os`, and `debug` capabilities where an embedding explicitly wants them
- typed generic-`for` iterator contracts

Unsafe host-facing libraries are not treated as default-sandbox requirements.

## Performance roadmap

0.17 establishes the intended optimizer architecture: **a small statically typed IR compiler targeting optimized Python AST today, with the IR remaining reusable by future backends.**

1. exact table-dispatched interpreter as Tier 0 and universal deoptimization target
2. source certification and type inference for `-- luapyre: typed`
3. hotness counters and quickening for loops/functions
4. compact backend-neutral typed IR with CFG fixed-point value/type propagation
5. SSA-like scalar value graph where copies are aliases and expressions have stable value numbers
6. exact signed-64 constant folding, CSE, graph-liveness DSE, and expression rematerialization
7. backend-neutral CALL IR with two policies: value-DAG inlining for tiny pure lexical children and direct real-frame lowering for larger statically resolved lexical callees
8. promoted-local Python AST lowering with exact aggregate fuel for pure graphs and shared incremental fuel for real-frame direct calls
9. proven 0.15/0.16 structured loops, recursion, captured closures, dynamic calls, and table/global tiers retained whenever the new typed/value/CALL IR cannot prove a stronger exact path
10. bounded call/table polymorphic inline caches and deoptimization site/reason feedback with unstable-region retirement
11. hot side-exit traces and CFG OSR through the exact live Frame/PC/register contract
12. later: native x86-64/AArch64 or LLVM backend consuming the same typed/value/CALL IR and deoptimization contract

CPython's own optimizer/JIT can accelerate generated Python when available, but it is never a LuaPyre correctness dependency.
