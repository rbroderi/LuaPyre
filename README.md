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

## 0.3 semantic core

The 0.3 tranche adds the semantic dispatch layer that future optimization tiers will specialize:

- metatables plus protected `getmetatable` / `setmetatable`
- `__index` and `__newindex`, including table delegation chains and loop protection
- `__call` for callable tables
- arithmetic and bitwise metamethods (`__add`, `__sub`, `__mul`, `__div`, `__idiv`, `__mod`, `__pow`, `__band`, `__bor`, `__bxor`, `__shl`, `__shr`)
- unary `__unm` / `__bnot`, `__len`, `__concat`, and comparison metamethods
- `do ... end`, `repeat ... until`, and `break`
- numeric `for`, including read-only loop variables
- generic `for` plus safe in-memory `next`, `pairs`, and `ipairs`
- anonymous function expressions
- dotted function definitions and `:` method definition/call syntax
- proper tail calls implemented by replacing VM frames instead of growing the frame stack
- Python 3.13 and 3.14 CI coverage

Earlier semantic-core work already includes register-bytecode execution, lexical scopes, closures/shared mutable upvalues, recursive local functions, multiple returns, Lua 5.5 named varargs, lexical `_ENV`, split Lua tables, byte strings, signed 64-bit integer behavior, optional type guards, explicit host capabilities, and fuel/frame limits.

## Compatibility testing

CI runs on Python 3.13 and 3.14 and installs Lupa 2.8+, then explicitly imports `lupa.lua55`. This gives the differential suite an in-process PUC-Lua 5.5 oracle without compiling or launching a separate Lua executable. CI verifies that the selected backend reports `_VERSION == "Lua 5.5"` before running tests.

For exact micro-release conformance, LuaPyre targets Lua 5.5.1 and the official 5.5.1 tests/reference implementation remain the final authority. Lupa is the fast per-commit differential oracle; release-level conformance will additionally be checked against the exact 5.5.1 distribution.

The full official Lua test suite is **not** expected to pass yet. Major remaining work includes `goto`/labels and full scope-entry validation, Lua 5.5 global declarations, local attributes and to-be-closed variables, coroutine semantics, weak tables/finalization/GC-visible behavior, `__pairs` and the rest of library-level metamethod details, the remaining safe standard libraries, binary chunks, and detailed error/debug compatibility.

## Performance roadmap

Correct semantics come first. Once the runtime is broad enough for the Lua 5.5.1 test suite, the performance path is:

1. adaptive quickening and inline caches
2. table/global/call specialization
3. generated superinstructions for common opcode sequences
4. hot-function compilation to Python code using Python locals for Lua registers
5. deoptimization back to the VM when specialization assumptions fail

Typed code can skip many dynamic guards because its annotations survive into compiler optimization metadata.
