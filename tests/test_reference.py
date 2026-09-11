"""Selected differential tests against PUC-Lua 5.5 through Lupa.

Lupa 2.7+ ships a dedicated ``lupa.lua55`` backend. Using it keeps the
reference interpreter in the same Python process and makes differential tests
fast enough to run in normal CI rather than requiring a separate Lua build.
"""

from __future__ import annotations

import pytest
from lupa.lua55 import LuaRuntime as ReferenceLuaRuntime

from luapyre import LuaRuntime


CASES = [
    "return 20 + 22",
    "return true == 1, false == 0, 1 == 1.0",
    'local t = {}; t[true] = "bool"; t[1] = "num"; return t[true], t[1]',
    "local x = 9223372036854775807; return x + 1",
    '''
local x = 10
local function add(n)
    x = x + n
    return x
end
add(5)
return add(7), x
''',
    '''
local function pair()
    return 20, 22
end
local a, b = pair()
return a, b
''',
    '''
local function f(first, ...args)
    return first, args.n, args[1], args[2], ...
end
return f(10, 20, 30)
''',
    '''
local newenv = {x = 40}
_ENV = newenv
x = x + 2
return x
''',
    'return "hello" .. " " .. "world"',
]


def _encode_python(value):
    if value is None:
        return "nil"
    if type(value) is bool:
        return f"boolean:{str(value).lower()}"
    if type(value) is int:
        return f"integer:{value}"
    if type(value) is float:
        return f"float:{value.hex()}"
    if isinstance(value, bytes):
        return "string:" + value.hex()
    return type(value).__name__


def _normalise(result):
    values = result if isinstance(result, tuple) else (result,)
    return [str(len(values)), *(_encode_python(v) for v in values)]


def _run_luapyre(source):
    return _normalise(LuaRuntime().execute(source))


def _run_reference(source):
    lua = ReferenceLuaRuntime(encoding=None)
    assert lua.eval("_VERSION") == b"Lua 5.5"
    return _normalise(lua.execute(source))


@pytest.mark.parametrize("source", CASES)
def test_selected_semantics_match_lua_55(source):
    assert _run_luapyre(source) == _run_reference(source)
