# LuaPyre

LuaPyre is a clean-slate Lua runtime written in Python. It targets **Lua 5.5.1** semantics, a sandbox-first embedding model, and optional gradual type annotations that can feed later optimization without creating a second runtime.

**Python 3.13+** · **current pre-alpha: 0.11.0a1**

LuaPyre is not yet a complete Lua 5.5.1 implementation. Correct semantics come first; the runtime is deliberately built around a register VM and explicit Lua frames so later quickening and JIT work can specialize stable behavior instead of replacing an AST interpreter.

## Language model

Plain Lua remains dynamic. Optional annotations add static guarantees and optimization facts:

```lua
local dynamic = get_value()       -- Any
local count: integer = 10

local function add(a: integer, b: integer): integer
    return a + b
end
```

Missing annotations mean `Any`. Typed and untyped code use the same parser, compiler, bytecode, VM, tables, closures, and standard library.

Lua 5.5 declarations and LuaPyre annotations can be combined:

```lua
local limit: integer <const> = 10
global answer: integer
answer = 42
```

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

See [`docs/embedding-0.11.md`](docs/embedding-0.11.md) for the complete 0.11 embedding contract.

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

The committed 0.11 release gate contains only unchanged upstream test files that are currently proven to pass. Additional official files remain explicit probes until their semantic gaps are fixed; the baseline is never weakened by marking failures as expected passes.

The complete official suite does **not** pass yet. Some upstream tests depend on Lua's internal C test API, debug/io/os/native-module facilities, allocator details, or stress behavior that is outside the default sandbox. Other failures identify genuine remaining Lua semantics and are tracked through explicit suite runs.

## Current compatibility gaps

Major remaining work includes:

- broader official Lua 5.5.1 suite coverage
- additional exact runtime-error / `getobjname` categories
- automatic incremental/generational GC pacing
- yieldable native-library / C-API continuation semantics
- fuller userdata/C-API behavior beyond the current sandbox value model
- host-facing `io`, `os`, and `debug` capabilities where an embedding explicitly wants them

Unsafe host-facing libraries are not treated as default-sandbox requirements.

## Performance roadmap

Correct semantics come first. Once compatibility is broad enough, the intended performance path is:

1. adaptive quickening and inline caches
2. table/global/call specialization
3. generated superinstructions for common opcode sequences
4. hot-function compilation to Python code using Python locals for Lua registers
5. deoptimization back to the VM when specialization assumptions fail

Typed code can skip many dynamic guards because annotations survive into compiler optimization metadata. CPython's optimizer/JIT is an optional accelerator, never a correctness dependency.
