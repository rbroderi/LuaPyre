import pytest

from luapyre import LuaInt, LuaRuntime
from luapyre.bytecode import Op
from luapyre.errors import LuaSyntaxError, LuaTypeError


def test_typed_cookie_marks_proto_and_infers_integer_bindings():
    source = """-- luapyre: typed
local total = 0
for i = 1, 100 do
    total = total + i
end
return total
"""
    runtime = LuaRuntime()
    proto = runtime.compile(source)

    assert proto.jit_fully_typed is True
    assert Op.ADD_I in [ins.op for ins in proto.code]
    assert runtime.vm.run(proto) == 5050


def test_typed_cookie_may_be_on_second_line_only():
    source = """-- generated file
-- luapyre: typed
local x = 40
local y = 2
return x + y
"""
    proto = LuaRuntime().compile(source)
    assert proto.jit_fully_typed is True
    assert Op.ADD_I in [ins.op for ins in proto.code]


def test_explicit_integer_is_no_overflow_contract_and_inferred_integer_wraps():
    runtime = LuaRuntime(jit=False)
    explicit = runtime.compile(
        "-- luapyre: typed\n"
        "local x: integer = 40\n"
        "local y: integer = 2\n"
        "return x + y\n"
    )
    inferred = runtime.compile(
        "-- luapyre: typed\n"
        "local x = 0x7fffffffffffffff\n"
        "return x + 1\n"
    )
    lua_integer = runtime.compile(
        "-- luapyre: typed\n"
        "local x: integer_lua = 0x7fffffffffffffff\n"
        "return x + 1\n"
    )

    explicit_add = next(i for i, ins in enumerate(explicit.code) if ins.op is Op.ADD_I)
    inferred_add = next(i for i, ins in enumerate(inferred.code) if ins.op is Op.ADD_I)
    lua_add = next(i for i, ins in enumerate(lua_integer.code) if ins.op is Op.ADD_I)
    assert explicit_add in explicit.jit_no_overflow_pcs
    assert inferred_add not in inferred.jit_no_overflow_pcs
    assert lua_add not in lua_integer.jit_no_overflow_pcs
    assert runtime.vm.run(explicit) == 42
    assert runtime.vm.run(inferred) == -(1 << 63)
    assert runtime.vm.run(lua_integer) == -(1 << 63)
    assert runtime.execute(
        "-- luapyre: typed\n"
        "local x = ~(-9223372036854775808)\n"
        "return x + 1\n"
    ) == -(1 << 63)
    assert runtime.execute("return 0x7fffffffffffffff + 1") == -(1 << 63)


def test_lua_int_python_hint_checks_signed_64_bit_result():
    runtime = LuaRuntime()
    assert runtime.execute_python("return 42", return_type=LuaInt) == 42
    with pytest.raises(OverflowError, match="signed 64-bit"):
        runtime._from_lua(1 << 63, LuaInt)


def test_typed_cookie_after_second_line_is_ordinary_comment():
    source = """-- first
-- second
-- luapyre: typed
return missing_global
"""
    proto = LuaRuntime().compile(source)
    assert proto.jit_fully_typed is False


def test_unknown_luapyre_cookie_is_rejected():
    with pytest.raises(LuaSyntaxError, match="unknown LuaPyre source mode"):
        LuaRuntime().compile("-- luapyre: tpyed\nreturn 1")


def test_typed_function_requires_parameter_and_return_types():
    runtime = LuaRuntime()
    with pytest.raises(LuaTypeError, match="parameter 'x'"):
        runtime.compile(
            "-- luapyre: typed\n"
            "local function bump(x): integer\n"
            "  return x + 1\n"
            "end\n"
            "return bump(1)\n"
        )

    with pytest.raises(LuaTypeError, match="return type"):
        runtime.compile(
            "-- luapyre: typed\n"
            "local function bump(x: integer)\n"
            "  return x + 1\n"
            "end\n"
            "return bump(1)\n"
        )


def test_typed_function_contract_runs_and_propagates_to_child_proto():
    source = """-- luapyre: typed
local function bump(x: integer): integer
    return x + 1
end
local value = bump(41)
return value
"""
    runtime = LuaRuntime()
    proto = runtime.compile(source)
    assert proto.children[0].jit_fully_typed is True
    assert runtime.vm.run(proto) == 42


def test_typed_keyword_type_atoms_are_available_without_changing_plain_parser():
    output = []
    runtime = LuaRuntime(output=output.append)
    source = """-- luapyre: typed
global print: function
global nothing: nil
print("typed")
return nothing
"""
    proto = runtime.compile(source)
    assert proto.jit_fully_typed is True
    assert runtime.vm.run(proto) is None
    assert output == [b"typed\n"]


def test_typed_mode_rejects_untyped_implicit_globals():
    with pytest.raises(LuaSyntaxError, match="global 'missing' is not declared"):
        LuaRuntime().compile("-- luapyre: typed\nreturn missing")


def test_typed_mode_allows_explicit_dynamic_boundary_guard():
    source = """-- luapyre: typed
global input: table
local value: integer = input.answer
return value
"""
    runtime = LuaRuntime()
    runtime.set("input", {"answer": 42})
    proto = runtime.compile(source)
    assert Op.GUARD in [ins.op for ins in proto.code]
    assert runtime.vm.run(proto) == 42


def test_typed_mode_rejects_uninferable_local_without_annotation():
    source = """-- luapyre: typed
global input: table
local value = input.answer
return value
"""
    with pytest.raises(LuaTypeError, match="cannot infer type of local 'value'"):
        LuaRuntime().compile(source)


def test_typed_mode_rejects_global_wildcard_and_generic_for_for_now():
    with pytest.raises(LuaTypeError, match="does not allow 'global \\*'"):
        LuaRuntime().compile("-- luapyre: typed\nglobal *\nreturn 1")

    source = """-- luapyre: typed
global input: table
local total = 0
for k, v in input do
    total = total + 1
end
return total
"""
    with pytest.raises(LuaTypeError, match="typed iterator contract"):
        LuaRuntime().compile(source)
