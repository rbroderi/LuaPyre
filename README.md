# LuaPyre

LuaPyre is a clean-slate Lua runtime written in Python. It targets Lua 5.5.1 semantics, a sandbox-first embedding model, and optional gradual type annotations that feed the optimizer without creating a second runtime.

**Python 3.13+**.

LuaPyre is still pre-alpha and is not yet a complete Lua 5.5.1 implementation. The runtime is deliberately built around a register VM and explicit Lua frames so later quickening and JIT work can specialize stable semantics rather than replace an AST interpreter.

## Language model

Plain Lua is dynamic. Optional annotations add static guarantees and optimization facts:

```lua
local dynamic = get_value()       -- Any
local count: integer = 10

local function add(a: integer, b: integer): integer
    return a + b
end
```

Missing annotations mean `Any`. Typed and untyped code use the same parser, compiler, register bytecode, and VM.

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

Host capabilities are explicit. The default runtime does not expose filesystem, networking, process execution, Python import/eval, package loading, `io`, `os`, `debug`, or Python object introspection.

## 0.5 coroutine core

The 0.5 tranche adds persistent Lua threads on top of the explicit-frame VM:

- `LuaThread` values with Lua `thread` type identity
- `coroutine.create`
- `coroutine.resume`
- `coroutine.yield`, including yields from nested Lua calls
- resume arguments becoming the return values of the suspended `coroutine.yield(...)` expression
- correct handling of `return coroutine.yield(...)` in tail position
- `coroutine.status` with `running`, `suspended`, `normal`, and `dead`
- `coroutine.running`, including main-thread identification
- `coroutine.isyieldable`
- `coroutine.wrap`
- `coroutine.close` for suspended and errored threads
- self-closing running coroutines, where `coroutine.close()` does not return to the coroutine body
- preservation of an errored coroutine's Lua stack until it is explicitly closed
- integration with 0.4 `<close>` state so pending `__close` methods run when a suspended or errored coroutine is closed
- the original Lua error object is supplied to pending `__close` methods during coroutine error cleanup

Coroutine stacks own the same Lua `Frame` objects used by the register VM, so registers, closures, upvalues, program counters, pending close state, and nested Lua calls survive suspension directly. Ordinary main-chunk execution continues through the established VM path; coroutine execution reuses the same opcode and semantic helpers with a persistent per-thread frame list.

The 0.4 tranche already provides `goto`/labels, local `<const>`/`<close>`, Lua 5.5 global declarations, strict-global behavior, and unified scope/error unwinding. The 0.3 tranche provides metatable/metamethod dispatch, additional control flow, anonymous functions, method syntax, generic iteration, and proper tail calls. Earlier semantic-core work includes lexical closures/upvalues, multiple returns, Lua 5.5 named varargs, lexical `_ENV`, split Lua tables, byte strings, signed 64-bit integer behavior, optional type guards, explicit host capabilities, and fuel/frame limits.

## Compatibility testing

CI runs on Python 3.13 and 3.14 and installs Lupa 2.8+, then explicitly imports `lupa.lua55`. This gives the differential suite an in-process PUC-Lua 5.5 oracle without compiling or launching a separate Lua executable. CI verifies that the selected backend reports `_VERSION == "Lua 5.5"` before running tests.

For exact micro-release conformance, LuaPyre targets Lua 5.5.1 and the official 5.5.1 tests/reference implementation remain the final authority. Lupa is the fast per-commit differential oracle; release-level conformance will additionally be checked against the exact 5.5.1 distribution.

The full official Lua test suite is **not** expected to pass yet. Major remaining work includes weak tables and ephemeron behavior, `__gc` finalization and other GC-observable semantics, the remaining safe standard libraries and library-level metamethod details such as `__pairs`, binary chunks, detailed debug/error compatibility, and substantially broader coverage of the official Lua 5.5.1 suite. Coroutine behavior is now implemented for normal Lua code, including close-aware suspension/error paths, but C-API/debug-hook yield compatibility remains outside the current sandbox-oriented scope.

## Performance roadmap

Correct semantics come first. Once the runtime is broad enough for the Lua 5.5.1 test suite, the performance path is:

1. adaptive quickening and inline caches
2. table/global/call specialization
3. generated superinstructions for common opcode sequences
4. hot-function compilation to Python code using Python locals for Lua registers
5. deoptimization back to the VM when specialization assumptions fail

Typed code can skip many dynamic guards because its annotations survive into compiler optimization metadata.
