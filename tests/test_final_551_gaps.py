from __future__ import annotations

import pytest

from luapyre import LuaRuntime


@pytest.mark.parametrize("jit", [False, True])
def test_nested_protected_stack_overflow_recovers(jit):
    lua = LuaRuntime(jit=jit, max_frames=40, fuel=500_000)
    ok, message = lua.execute(
        """
local function loop()
  assert(pcall(loop))
end
return xpcall(loop, loop)
"""
    )
    assert ok is False
    assert b"error" in message


@pytest.mark.parametrize("jit", [False, True])
def test_protected_coroutine_call_can_yield_then_fail(jit):
    lua = LuaRuntime(jit=jit)
    assert lua.execute(
        """
local function work()
  coroutine.yield(11)
  error(23)
end
local co = coroutine.create(pcall)
local r1, yielded = coroutine.resume(co, work)
local r2, ok, err = coroutine.resume(co)
return r1, yielded, r2, ok, err
"""
    ) == (True, 11, True, False, 23)


def test_direct_yield_is_resumable_through_pcall():
    lua = LuaRuntime(jit=False)
    assert lua.execute(
        """
local co = coroutine.create(function ()
  return pcall(coroutine.yield, 7)
end)
local r1, yielded = coroutine.resume(co)
local r2, ok, value = coroutine.resume(co, 9)
return r1, yielded, r2, ok, value
"""
    ) == (True, 7, True, True, 9)


def test_close_error_traceback_names_close_metamethod():
    lua = LuaRuntime(jit=False)
    ok, message = lua.execute(
        """
local value = setmetatable({}, {
  __close = function() error("closing") end
})
local function work()
  local item <close> = value
end
return xpcall(work, debug.traceback)
"""
    )
    assert ok is False
    assert b"in metamethod 'close'" in message


def test_weak_table_does_not_keep_coroutine_wrapper_alive():
    lua = LuaRuntime(jit=False)
    assert lua.execute(
        """
local weak = setmetatable({}, {__mode = "v"})
local wrapper = coroutine.wrap(function ()
  local value = 10
  while true do
    coroutine.yield(function () value = value + 1; return value end)
  end
end)
weak[1] = wrapper
local retained = wrapper()
wrapper = nil
collectgarbage()
return weak[1], retained()
"""
    ) == (None, 11)


def test_wrapped_coroutine_reports_replacement_close_error():
    lua = LuaRuntime(jit=False)
    ok, message, closes = lua.execute(
        """
local closes = 0
local function closer(callback)
  return setmetatable({}, {__close = callback})
end
local wrapper = coroutine.wrap(function ()
  local outer <close> = closer(function (_, prior)
    closes = closes + 1
    assert(string.find(prior, "first"))
    error("replacement")
  end)
  local inner <close> = closer(function ()
    closes = closes + 1
    error("first")
  end)
  coroutine.yield()
  error("body")
end)
wrapper()
local ok, message = pcall(wrapper)
return ok, message, closes
"""
    )
    assert ok is False
    assert b"replacement" in message
    assert closes == 2


def test_debug_function_metadata_and_call_site_names():
    lua = LuaRuntime(jit=False)
    assert lua.execute(
        """
local function sample()
  local info = debug.getinfo(1)
  return info.name, info.namewhat
end
local alias = sample
local metadata = debug.getinfo(sample, "SL")
local name, kind = alias()
return name, kind, metadata.linedefined, metadata.lastlinedefined,
       metadata.activelines[metadata.lastlinedefined]
"""
    ) == (b"alias", b"local", 2, 5, True)


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        (b"local a = {4\n", b"'}' expected (to close '{' at line 1) near <eof>"),
        (b"syntax error", b"syntax error near 'error'"),
        (b"for >> do end", b"<name> expected near '>>'"),
        (b"a\x01a = 1", b"syntax error near '<\\1>'"),
        (b"\xffa = 1", b"unexpected symbol near '<\\255>'"),
    ],
)
def test_parser_diagnostics_use_lua_tokens(source, fragment):
    lua = LuaRuntime(jit=False)
    lua.set("source", source)
    message = lua.execute("local fn, err = load(source); return err")
    assert fragment in message


def test_nonclosable_local_names_the_variable():
    lua = LuaRuntime(jit=False)
    ok, message = lua.execute(
        "local function f() local item <close> = {} end; return pcall(f)"
    )
    assert ok is False
    assert b"variable 'item' got a non-closable value" in message
