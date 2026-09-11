import pytest

from luapyre import LuaRuntime, LuaTypeError, LuaRuntimeError, LuaQuotaError


def test_plain_lua_defaults_to_any_and_runs():
    lua = LuaRuntime()
    assert lua.execute("local x = 20; local y = 22; return x + y") == 42


def test_typed_arithmetic_specializes():
    lua = LuaRuntime()
    src = "local x: integer = 20; local y: integer = 22; return x + y"
    assert lua.execute(src) == 42
    assert "ADD_I" in lua.disassemble(src)


def test_static_type_mismatch_rejected():
    lua = LuaRuntime()
    with pytest.raises(LuaTypeError):
        lua.compile('local x: integer = "nope"')


def test_host_capability_boundary():
    lua = LuaRuntime()
    lua.expose("double", lambda n: n * 2)
    assert lua.execute("return double(21)") == 42
    assert "open" not in lua.globals and "__import__" not in lua.globals


def test_typed_function_and_any_boundary_guard():
    lua = LuaRuntime()
    lua.expose("value", lambda: 21)
    src = '''
local function twice(x: integer): integer
    return x * 2
end
local x = value()
return twice(x)
'''
    assert lua.execute(src) == 42


def test_bad_any_to_typed_function_fails_runtime_guard():
    lua = LuaRuntime()
    lua.expose("value", lambda: "oops")
    src = '''
local function twice(x: integer): integer
    return x * 2
end
return twice(value())
'''
    with pytest.raises(LuaRuntimeError, match="expected integer"):
        lua.execute(src)


def test_while_and_assignment():
    lua = LuaRuntime()
    src = '''
local i: integer = 0
local total: integer = 0
while i < 10 do
    i = i + 1
    total = total + i
end
return total
'''
    assert lua.execute(src) == 55


def test_fuel_quota():
    lua = LuaRuntime(fuel=100)
    with pytest.raises(LuaQuotaError):
        lua.execute("while true do end")
