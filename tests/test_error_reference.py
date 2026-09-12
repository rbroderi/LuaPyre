"""Differential coverage for the 0.9 source/error-fidelity tranche."""

from __future__ import annotations

import pytest
from lupa.lua55 import LuaRuntime as ReferenceLuaRuntime

from luapyre import LuaRuntime


CASES = [
    '''
local f=assert(load("error('boom')","=named"))
local ok,e=pcall(f)
return ok,e
''',
    '''
local f=assert(load("assert(false)","=assertion"))
local ok,e=pcall(f)
return ok,e
''',
    '''
local f=assert(load([[local function outer()
  local function inner()
    error("boom",2)
  end
  inner()
end
outer()]],"=levels"))
local ok,e=pcall(f)
return ok,e
''',
    '''
local f=assert(load("error('x')"))
local ok,e=pcall(f)
return ok,e
''',
    '''
local done=false
local function reader()
  if done then return nil end
  done=true
  return "error('x')"
end
local f=assert(load(reader))
local ok,e=pcall(f)
return ok,e
''',
    '''
local ok,e=pcall(function() error(nil) end)
return ok,e
''',
    '''
local f,e=load("local = 1","=bad")
return f,type(e),string.sub(e,1,6)
''',
    '''
local f=assert(load("local x=nil\\nreturn x+1","=calc"))
local ok,e=pcall(f)
return ok,string.sub(e,1,8),string.find(e,"arithmetic") ~= nil
''',
    '''
local maker=assert(load("return function()\\n  error('boom')\\nend","=origin"))
local f=maker()
local full=string.dump(f,false)
local g=assert(load(full,"=replacement","b"))
local ok,e=pcall(g)
return ok,e
''',
    '''
local maker=assert(load("return function()\\n  error('boom')\\nend","=origin"))
local f=maker()
local stripped=string.dump(f,true)
local g=assert(load(stripped,"=replacement","b"))
local ok,e=pcall(g)
return ok,e
''',
]


def _encode(value):
    if value is None:
        return "nil"
    if type(value) is bool:
        return "boolean:" + str(value).lower()
    if type(value) is int:
        return f"integer:{value}"
    if type(value) is float:
        return "float:" + value.hex()
    if isinstance(value, bytes):
        return "string:" + value.hex()
    return type(value).__name__


def _normalise(result):
    values = result if isinstance(result, tuple) else (result,)
    return [str(len(values)), *(_encode(value) for value in values)]


def _reference(source):
    lua = ReferenceLuaRuntime(encoding=None)
    return _normalise(lua.execute(source))


@pytest.mark.parametrize("source", CASES)
def test_error_diagnostics_match_lua_55(source):
    assert _normalise(LuaRuntime().execute(source)) == _reference(source)
