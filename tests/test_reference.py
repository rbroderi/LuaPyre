"""Selected differential tests against the official Lua 5.5.1 interpreter.

Set LUA55_BIN to the reference interpreter path. CI builds the exact 5.5.1
release from lua.org. The test is skipped for local developers who do not have
that binary installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from luapyre import LuaRuntime


LUA55 = os.environ.get("LUA55_BIN") or shutil.which("lua5.5")
pytestmark = pytest.mark.skipif(not LUA55, reason="official Lua 5.5 interpreter not available")

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


def _run_luapyre(source):
    result = LuaRuntime().execute(source)
    values = result if isinstance(result, tuple) else (result,)
    return [str(len(values)), *(_encode_python(v) for v in values)]


def _run_reference(source):
    wrapper = r'''
local function __hex(s)
    return (s:gsub(".", function(c) return string.format("%02x", string.byte(c)) end))
end
local function __enc(v)
    local t = type(v)
    if t == "nil" then return "nil" end
    if t == "boolean" then return "boolean:" .. tostring(v) end
    if t == "number" then
        if math.type(v) == "integer" then return "integer:" .. string.format("%d", v) end
        return "float:" .. string.format("%a", v)
    end
    if t == "string" then return "string:" .. __hex(v) end
    return t
end
local function __case()
    local _ENV = _ENV
%s
end
local __r = table.pack(__case())
io.write(tostring(__r.n), "\n")
for i = 1, __r.n do
    io.write(__enc(__r[i]), "\n")
end
''' % source
    proc = subprocess.run(
        [LUA55, "-"],
        input=wrapper,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.splitlines()


@pytest.mark.parametrize("source", CASES)
def test_selected_semantics_match_lua_551(source):
    assert _run_luapyre(source) == _run_reference(source)
