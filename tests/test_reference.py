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
    '''
local backing = {x = 40}
local proxy = {}
setmetatable(proxy, {
  __index = function(t, k) return backing[k] end,
  __newindex = function(t, k, v) backing[k] = v end
})
proxy.x = proxy.x + 2
return proxy.x, rawget(proxy, "x"), backing.x
''',
    '''
local mt = {__add = function(a,b) return a.n + b.n end}
local a = setmetatable({n=20}, mt)
local b = setmetatable({n=22}, mt)
return a + b
''',
    '''
local x = 0
repeat
  x = x + 1
  if x == 5 then break end
until false
return x
''',
    'local s=0; for i=1,10 do s=s+i end; return s',
    'local s=0; for k,v in pairs({a=20,b=22}) do s=s+v end; return s',
    'local f=function(x) return x+2 end; return f(40)',
    'local t={n=40}; function t:add(x) return self.n+x end; return t:add(2)',
    '''
local function loop(n, acc)
  if n == 0 then return acc end
  return loop(n-1, acc+1)
end
return loop(500,0)
''',
    '''
local i = 0
::again::
i = i + 1
if i < 3 then goto again end
return i
''',
    'local x <const> = 42; return x',
    '''
local log = ""
local mt = {__close = function(self) log = log .. self.name end}
do
  local a <close> = setmetatable({name="a"}, mt)
  local b <close> = setmetatable({name="b"}, mt)
end
return log
''',
    '''
local log = ""
local mt = {__close = function(self) log = log .. self.name end}
do
  local x <close> = setmetatable({name="x"}, mt)
  goto done
end
::done::
return log
''',
    '''
global x
global<const> *
x = 42
return x, type(x)
''',
    '''
global function fact(n)
  if n <= 1 then return 1 end
  return n * fact(n - 1)
end
return fact(6)
''',
    '''
local co = coroutine.create(function(a)
  local x, y = coroutine.yield(a + 1, a + 2)
  return x + y
end)
local ok1, a, b = coroutine.resume(co, 10)
local s1 = coroutine.status(co)
local ok2, c = coroutine.resume(co, 20, 22)
return ok1, a, b, s1, ok2, c, coroutine.status(co), type(co)
''',
    '''
local function inner()
  return coroutine.yield("inner")
end
local co = coroutine.create(function()
  return inner()
end)
local ok1, value = coroutine.resume(co)
local ok2, final = coroutine.resume(co, 42)
return ok1, value, ok2, final
''',
    '''
local parent
local child = coroutine.create(function()
  return coroutine.status(parent)
end)
parent = coroutine.create(function()
  local ok, status = coroutine.resume(child)
  return ok, status
end)
local ok, childok, status = coroutine.resume(parent)
return ok, childok, status
''',
    '''
local f = coroutine.wrap(function(x)
  local y = coroutine.yield(x + 1)
  return y + 1
end)
local first = f(10)
local second = f(41)
return first, second
''',
    '''
local closed = 0
local co = coroutine.create(function()
  local resource <close> = setmetatable({}, {
    __close = function(self, err) closed = closed + 1 end
  })
  coroutine.yield("paused")
end)
local ok1, value = coroutine.resume(co)
local ok2 = coroutine.close(co)
return ok1, value, ok2, closed, coroutine.status(co)
''',
    '''
local closed = 0
local co = coroutine.create(function()
  local resource <close> = setmetatable({}, {
    __close = function(self, err) closed = closed + 1 end
  })
  coroutine.yield("ready")
  error("boom")
end)
local first = coroutine.resume(co)
local second = coroutine.resume(co)
local closeok = coroutine.close(co)
return first, second, closeok, closed, coroutine.status(co)
''',
    '''
local closed = 0
local co = coroutine.create(function()
  local resource <close> = setmetatable({}, {
    __close = function(self, err) closed = closed + 1 end
  })
  coroutine.close()
  closed = 100
  return 99
end)
local ok, value = coroutine.resume(co)
return ok, value, closed, coroutine.status(co)
''',
]


ERROR_CASES = [
    '''
goto target
local x = 1
::target::
return x
''',
    "local x <const> = 1; x = 2",
    "global x; return y",
    "x = 1; global x = 2",
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


@pytest.mark.parametrize("source", ERROR_CASES)
def test_selected_compile_and_runtime_errors_match_lua_55(source):
    with pytest.raises(Exception):
        LuaRuntime().execute(source)
    lua = ReferenceLuaRuntime(encoding=None)
    assert lua.eval("_VERSION") == b"Lua 5.5"
    with pytest.raises(Exception):
        lua.execute(source)
