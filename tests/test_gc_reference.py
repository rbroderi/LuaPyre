"""GC-observable differential cases against PUC-Lua 5.5 via Lupa."""

from __future__ import annotations

import pytest
from lupa.lua55 import LuaRuntime as ReferenceLuaRuntime

from luapyre import LuaRuntime


CASES = [
    '''
local weak = setmetatable({}, {__mode = "v"})
local function seed()
  local value = {n = 42}
  weak[1] = value
end
seed()
collectgarbage()
return weak[1] == nil
''',
    '''
local weak = setmetatable({}, {__mode = "k"})
local function seed()
  local key = {}
  weak[key] = 42
  weak["string-key"] = 7
end
seed()
collectgarbage()
local n = 0
for k, v in pairs(weak) do n = n + 1 end
return n, weak["string-key"]
''',
    '''
local eph = setmetatable({}, {__mode = "k"})
local function seed()
  local key = {}
  local value = {key = key}
  eph[key] = value
end
seed()
collectgarbage()
return next(eph) == nil
''',
    '''
local log = ""
local mt = {__gc = function(self) log = log .. self.name end}
local function seed()
  local a = setmetatable({name = "a"}, mt)
  local b = setmetatable({name = "b"}, mt)
end
seed()
collectgarbage()
return log
''',
    '''
local weak = setmetatable({}, {__mode = "k"})
local during = false
local mt = {__gc = function(self) during = weak[self] == 42 end}
local function seed()
  local value = setmetatable({}, mt)
  weak[value] = 42
end
seed()
collectgarbage()
local first = next(weak) ~= nil
collectgarbage()
local second = next(weak) ~= nil
return during, first, second
''',
]


def _encode(value):
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
    return [str(len(values)), *(_encode(v) for v in values)]


def _reference(source):
    lua = ReferenceLuaRuntime(encoding=None)
    assert lua.eval("_VERSION") == b"Lua 5.5"
    return _normalise(lua.execute(source))


@pytest.mark.parametrize("source", CASES)
def test_gc_observable_semantics_match_lua_55(source):
    assert _normalise(LuaRuntime().execute(source)) == _reference(source)
