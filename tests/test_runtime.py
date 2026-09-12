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


def test_strings_are_bytes_and_concat():
    lua = LuaRuntime()
    assert lua.execute('return "hello" .. " " .. "world"') == b"hello world"


def test_table_array_and_fields():
    lua = LuaRuntime()
    assert lua.execute('local t = {10, 20, name = "Judy"}; return t[2], t.name') == (20, b"Judy")


def test_table_boolean_and_integer_keys_are_distinct():
    lua = LuaRuntime()
    assert lua.execute('local t = {}; t[true] = "bool"; t[1] = "num"; return t[true], t[1]') == (b"bool", b"num")


def test_integral_float_and_integer_keys_alias():
    assert LuaRuntime().execute('local t = {}; t[1.0] = 42; return t[1]') == 42


def test_lexical_shadowing():
    src = '''
local x = 1
if true then
    local x = 2
    x = x + 1
end
return x
'''
    assert LuaRuntime().execute(src) == 1


def test_closure_reads_and_writes_upvalue():
    src = '''
local x = 10
local function add(n)
    x = x + n
    return x
end
add(5)
return add(7), x
'''
    assert LuaRuntime().execute(src) == (22, 22)


def test_recursive_local_function_uses_shared_cell():
    src = '''
local function fact(n: integer): integer
    if n <= 1 then return 1 end
    return n * fact(n - 1)
end
return fact(6)
'''
    assert LuaRuntime().execute(src) == 720


def test_multiple_returns_expand_in_assignment_and_return():
    src = '''
local function pair() return 20, 22 end
local a, b = pair()
return a, b
'''
    assert LuaRuntime().execute(src) == (20, 22)


def test_only_last_expression_expands():
    src = '''
local function pair() return 1, 2 end
return pair(), pair()
'''
    assert LuaRuntime().execute(src) == (1, 1, 2)


def test_multiple_results_expand_in_call_arguments():
    src = '''
local function pair() return 20, 22 end
local function add(a, b) return a + b end
return add(pair())
'''
    assert LuaRuntime().execute(src) == 42


def test_varargs_and_named_vararg_table():
    src = '''
local function f(first, ...args)
    return first, args.n, args[1], args[2], ...
end
return f(10, 20, 30)
'''
    assert LuaRuntime().execute(src) == (10, 2, 20, 30, 20, 30)


def test_typed_varargs_are_guarded():
    lua = LuaRuntime()
    assert lua.execute('''
local function sum(...args: integer): integer
    return args[1] + args[2]
end
return sum(20, 22)
''') == 42
    with pytest.raises(LuaRuntimeError, match="vararg 2"):
        lua.execute('''
local function sum(...args: integer): integer
    return args[1] + args[2]
end
return sum(20, "bad")
''')


def test_local_env_redirects_global_access_lexically():
    lua = LuaRuntime()
    assert lua.execute('''
local env = {x = 41}
local _ENV = env
x = x + 1
return x
''') == 42
    assert lua.get("x") is None


def test_nested_function_captures_local_env():
    src = '''
local env = {x = 40}
local _ENV = env
local function f()
    x = x + 2
    return x
end
return f(), env.x
'''
    assert LuaRuntime().execute(src) == (42, 42)


def test_safe_stdlib_has_no_io_or_os():
    lua = LuaRuntime()
    assert lua.execute('return type(_G), _VERSION') == (b"table", b"Lua 5.5")
    assert lua.get("io") is None and lua.get("os") is None
    assert lua.get("debug") is None
    assert lua.get("package") is not None and lua.get("require") is not None
    assert lua.get("loadfile") is None and lua.get("dofile") is None


def test_rawlen_table_and_string():
    assert LuaRuntime().execute('return rawlen({1,2,3}), rawlen("abc")') == (3, 3)


def test_integer_overflow_wraps():
    assert LuaRuntime().execute("local x: integer = 9223372036854775807; return x + 1") == -(1 << 63)


def test_lua_boolean_numeric_equality_is_false():
    assert LuaRuntime().execute("return true == 1, false == 0, 1 == 1.0") == (False, False, True)


def test_captured_loop_local_gets_fresh_cell_each_iteration():
    src = '''
local f1
local f2
local i = 0
while i < 2 do
    local x = i
    local function f() return x end
    if i == 0 then f1 = f else f2 = f end
    i = i + 1
end
return f1(), f2()
'''
    assert LuaRuntime().execute(src) == (0, 1)


def test_implicit_env_is_assignable_lexical_variable():
    lua = LuaRuntime()
    assert lua.execute('''
local newenv = {x = 40}
_ENV = newenv
x = x + 2
return x
''') == 42
    assert lua.get("x") is None


def test_source_cache_reuses_proto_and_reports_hits():
    lua = LuaRuntime()
    first = lua.compile("return 42")
    second = lua.compile("return 42")
    assert second is first
    assert lua.source_cache_info == {
        "hits": 1,
        "misses": 1,
        "size": 1,
        "maxsize": 128,
    }


def test_source_cache_is_bounded_lru():
    lua = LuaRuntime(source_cache_size=2)
    first = lua.compile("return 1")
    evicted = lua.compile("return 2")
    assert lua.compile("return 1") is first
    lua.compile("return 3")
    assert lua.compile("return 2") is not evicted
    assert lua.source_cache_info["size"] == 2


def test_source_cache_can_be_disabled_and_cleared():
    disabled = LuaRuntime(source_cache_size=0)
    assert disabled.compile("return 42") is not disabled.compile("return 42")
    assert disabled.source_cache_info == {
        "hits": 0,
        "misses": 2,
        "size": 0,
        "maxsize": 0,
    }

    lua = LuaRuntime(source_cache_size=2)
    lua.compile("return 42")
    lua.clear_source_cache()
    assert lua.source_cache_info == {
        "hits": 0,
        "misses": 0,
        "size": 0,
        "maxsize": 2,
    }
