# LuaPyre

LuaPyre is a clean-slate Lua runtime written in Python. The project targets Lua 5.5 semantics, a sandbox-first embedding model, and optional gradual type annotations that feed the optimizer without creating a second runtime.

This repository is at the first executable milestone. It is **not yet a complete Lua 5.5 implementation**. The current core establishes the architecture that later conformance and performance work will extend.

## Design goals

- Plain Lua 5.5 source remains valid input.
- Optional annotations use familiar syntax: `local x: integer = 1` and `function add(a: integer, b: integer): integer`.
- Missing annotations mean `Any`.
- The AST is compile-time only; execution uses register bytecode.
- Lua calls use an explicit VM frame stack rather than the Python call stack.
- Host access is capability-based. Lua cannot import Python, open files, or access the network unless the embedding application explicitly exposes a function.
- The same typed IR/bytecode path is used for typed and untyped programs.

## Example

```python
from luapyre import LuaRuntime

lua = LuaRuntime()

lua.expose("double_from_python", lambda n: n * 2)

result = lua.execute("""
local function add(a: integer, b: integer): integer
    return a + b
end

local x = double_from_python(10) -- x is Any statically
return add(x, 22)
""")

assert result == 42
```

## Current milestone

Implemented now:

- lexer and parser for a useful Lua subset
- optional `: type` annotations on locals, parameters, and returns
- `Any`, `nil`, `boolean`, `integer`, `float`, `number`, `string`, and unions
- compile-time type checking for explicit annotations
- register bytecode compiler
- explicit VM call frames
- integer/float-specialized arithmetic opcodes when types are proven
- sandboxed Python host functions
- fuel-based execution quota
- differential-friendly bytecode disassembly

Next major targets are full Lua 5.5 grammar/semantics, tables and metamethods, closures/upvalues, multiple returns, coroutines, `_ENV`, `goto`, `<close>`, the complete safe standard library, official 5.5.1 conformance tests, quickening/inline caches, and the Python-code JIT.
