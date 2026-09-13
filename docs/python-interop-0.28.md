# Python interoperability (0.28)

LuaPyre 0.28 adds an explicit, Python-friendly boundary without changing the
existing low-level `execute`, `get`, or `LuaTable` APIs.

## Values

`LuaRuntime.set` and exposed Python return values recursively convert:

- `int`, `float`, `str`, `bytes`, `bool`, and `None`;
- `list`/`tuple` to a 1-indexed Lua sequence table;
- `set`/`frozenset` to a Lua membership table (`value -> true`);
- dictionaries with scalar keys to Lua tables; and
- dataclass instances to tables keyed by field name.

Python callables explicitly supplied through `set`, `expose`, or a containing
value become sandboxed host functions. Lua receives no ambient ability to
import or inspect Python.

`execute_python`, `get_python`, `call`, and `LuaFunction` convert results back.
Without a requested type, UTF-8 Lua strings become `str`, dense sequence tables
become `list`, other tables become `dict`, and functions become `LuaFunction`.
Pass `return_type=` when a table must become a `set`, dataclass, or specifically
typed container.

```python
from dataclasses import dataclass
from luapyre import LuaRuntime

@dataclass
class Point:
    x: int
    y: int

lua = LuaRuntime()
lua.set("point", Point(20, 22))
updated = lua.execute_python(
    "return {x = point.x + 1, y = point.y + 2}",
    return_type=Point,
)
assert updated == Point(21, 24)
```

Lua functions are callable from Python:

```python
add = lua.execute_python("return function(a, b) return a + b end")
assert add(20, 22, return_type=int) == 42
```

Python callbacks receive converted arguments. An annotation selects an exact
table conversion, including dataclasses:

```python
def magnitude(point: Point) -> float:
    return (point.x * point.x + point.y * point.y) ** 0.5

lua.expose("magnitude", magnitude)
```

## Boundary rules

- Ordinary Lua integers and typed `integer_lua` retain signed 64-bit Lua
  semantics. From 0.29, explicit typed `integer`/Python `LuaInt` boundaries are
  a no-overflow optimization contract.
- Lua treats integral float and integer table keys as the same key, so Python
  dictionary key identity cannot distinguish `1` from `1.0` after a round trip.
- A typed `set[T]` accepts either the membership-table representation or a
  dense Lua sequence.
- Cyclic Python containers and cyclic Lua tables are rejected at this value
  boundary. Opaque explicitly supplied host userdata retains its existing
  identity-preserving behavior.
- Raw APIs remain available for embedders that need `bytes`, `LuaTable`, or
  underlying closure objects directly.
