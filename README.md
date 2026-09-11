# LuaPyre

LuaPyre is a clean-slate Lua runtime written in Python. It targets Lua 5.5.1 semantics, a sandbox-first embedding model, and optional gradual type annotations that feed the optimizer without creating a second runtime.

**Python 3.14+ only.**

LuaPyre is still pre-alpha and is not yet a complete Lua 5.5.1 implementation. The current core is deliberately built around the execution architecture intended for later quickening and JIT work rather than an AST interpreter.

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

## Current semantic core

Implemented in the current tranche:

- register-bytecode execution; the AST is compile-time only
- explicit VM call frames rather than Python recursion
- lexical block scopes and local shadowing
- closures and shared mutable upvalues
- fresh captured local cells when block declarations execute again in loops
- recursive local functions
- Lua-style multiple returns and last-expression expansion
- Lua 5.5 named vararg tables (`...args`) plus optional annotations (`...args: integer`)
- lexical `_ENV`, including reassignment and capture by nested functions
- Lua tables with separate dense-array/hash storage
- Lua-correct table key distinctions (`true` and `1` are different keys; `1` and `1.0` alias)
- byte strings internally
- signed 64-bit integer wrapping for integer arithmetic
- arithmetic, comparison, bitwise, concatenation, and length operators in the current parser subset
- gradual type guards at dynamic-to-typed boundaries
- safe in-memory base helpers (`type`, `tostring`, `tonumber`, `assert`, `rawequal`, `rawget`, `rawset`, `rawlen`)
- capability-only Python host functions
- fuel and frame limits
- selected differential tests against the official Lua 5.5.1 reference interpreter

## Compatibility testing

CI targets Python 3.14, downloads the official Lua 5.5.1 source archive from lua.org, verifies its published SHA-256 checksum, builds it, and runs selected programs through both PUC-Lua and LuaPyre. The repository also has direct unit tests for the extended typing syntax.

The full official Lua test suite is **not** expected to pass yet. Remaining language/runtime work includes the rest of the grammar (`for`, `repeat`, `goto`, labels, anonymous functions, method syntax, global declarations and attributes), complete metamethod semantics, coroutines, weak tables/finalization/GC-visible behavior, to-be-closed variables, the remaining safe standard libraries, binary chunks, and detailed error/debug compatibility.

## Performance roadmap

Correct semantics come first. Once the core is broad enough for the Lua 5.5.1 test suite, the performance path is:

1. adaptive quickening and inline caches
2. table/global/call specialization
3. generated superinstructions for common opcode sequences
4. hot-function compilation to Python code using Python locals for Lua registers
5. deoptimization back to the VM when specialization assumptions fail

Typed code can skip many dynamic guards because its annotations survive into compiler optimization metadata.
