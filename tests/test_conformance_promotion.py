from __future__ import annotations

from luapyre import LuaRuntime
from luapyre.native_debug_chunks import PUC55_HEADER


def test_dump_has_canonical_lua_55_header_and_rejects_every_truncation():
    lua = LuaRuntime(fuel=5_000_000)
    dumped = lua.execute("return string.dump(function() return 42 end)")
    assert dumped.startswith(PUC55_HEADER)
    lua.set("dumped", dumped)
    assert lua.execute(
        """for i = 1, #dumped - 1 do
  local f, message = load(string.sub(dumped, 1, i))
  assert(f == nil and string.find(message, "truncated"))
end
return assert(load(dumped))()
"""
    ) == 42


def test_recursive_xpcall_handler_terminates_or_hits_lua_stack_limit():
    assert LuaRuntime(fuel=5_000_000).execute(
        """local function handler(n)
  if type(n) ~= "number" then return n end
  if n == 0 then return "END" end
  error(n - 1)
end
local ok1, end_message = xpcall(error, handler, 170)
local ok2, overflow = xpcall(error, handler, 300)
return ok1, end_message, ok2, overflow
"""
    ) == (False, b"END", False, b"C stack overflow")


def test_bounded_parser_and_function_limits_match_lua_failures():
    lua = LuaRuntime()
    assert lua.execute(
        """local names = {}
for i = 1, 201 do names[i] = "v" .. i end
local f1, e1 = load("local " .. table.concat(names, ","))
local f2, e2 = load("return " .. string.rep("(", 500) .. "1" .. string.rep(")", 500))
return f1, string.find(e1, "too many local variables") ~= nil,
       f2, string.find(e2, "too many") ~= nil
"""
    ) == (None, True, None, True)


def test_gc_parameter_extremes_are_bounded_without_host_allocation_control():
    lua = LuaRuntime()
    assert lua.execute(
        """local old = collectgarbage("param", "stepmul", 0x7ffffffe)
local current = collectgarbage("param", "stepmul")
collectgarbage("param", "stepmul", old)
return current
"""
    ) == 0x7FFFFFFE


def test_nested_collection_from_finalizer_is_deferred_without_an_error():
    warnings = []
    lua = LuaRuntime(warning=warnings.append)
    assert lua.execute(
        """local result
do
  local value = setmetatable({}, {__gc = function()
    result = collectgarbage("step")
  end})
end
collectgarbage()
return result
"""
    ) is False
    assert warnings == []


def test_plain_lua_environment_binding_remains_dynamic():
    ok, message = LuaRuntime().execute(
        """local f = assert(load("_ENV = 1; global function foo() end", "=env"))
return pcall(f)
"""
    )
    assert ok is False
    assert message.startswith(b"env:1: attempt to index a number value")
