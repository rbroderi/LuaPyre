from __future__ import annotations

import pytest

from luapyre import LuaInt, LuaQuotaError, LuaRuntime, LuaRuntimeError


def outcome(runtime, proto, fuel):
    try:
        return ("return", runtime.vm.run(proto, fuel=fuel))
    except (LuaQuotaError, LuaRuntimeError) as error:
        return (type(error).__name__, str(error))


@pytest.mark.parametrize("loop", ["1, 6", "6, 1, -1", "1.0, 3.0, 0.5"])
@pytest.mark.parametrize("condition", ["i % 2 == 0", "i < 4"])
def test_diamond_asymmetric_paths_keep_every_fuel_boundary(loop, condition):
    source = f"""-- luapyre: typed
local total: integer = 0
for i = {loop} do
    if {condition} then
        total = total + 3
        total = total * 2
    else
        total = total - 1
    end
end
return total
"""
    runtimes = [LuaRuntime(jit=jit, jit_threshold=1) for jit in (False, True)]
    protos = [runtime.compile(source) for runtime in runtimes]
    assert outcome(runtimes[0], protos[0], 1000) == outcome(runtimes[1], protos[1], 1000)
    for fuel in range(150):
        assert outcome(runtimes[0], protos[0], fuel) == outcome(runtimes[1], protos[1], fuel), fuel


@pytest.mark.parametrize("arm", ["prefix", "then", "else"])
def test_diamond_side_exits_preserve_partial_path_fuel_and_error(arm):
    condition = "i % (4-i) == 0" if arm == "prefix" else "i % 2 == 0"
    divisor = 5 if arm == "else" else 4
    operation = f"local x: integer = i % ({divisor}-i); total = total + x"
    source = f"""-- luapyre: typed
local total: integer = 0
for i = 1, 6 do
    if {condition} then
        {operation if arm == 'then' else 'total = total + 1'}
    else
        {operation if arm == 'else' else 'total = total - 1'}
    end
end
return total
"""
    runtimes = [LuaRuntime(jit=jit, jit_threshold=1) for jit in (False, True)]
    protos = [runtime.compile(source) for runtime in runtimes]
    assert outcome(runtimes[0], protos[0], 1000) == outcome(runtimes[1], protos[1], 1000)
    assert runtimes[1].jit_stats.deopts > 0
    for fuel in (1000, *range(150)):
        assert outcome(runtimes[0], protos[0], fuel) == outcome(runtimes[1], protos[1], fuel), fuel


@pytest.mark.parametrize("condition, expected", [
    ("i % 2 == 0.0", 3),
    ("i % 2 == true", 0),
    ("i % 2 == nil", 0),
    ("(i < 4) == true", 3),
    ("'a' < 'b'", 6),
    ("i % 2 < 1.0", 3),
])
def test_diamond_primitive_comparisons_keep_lua_type_semantics(condition, expected):
    source = f"""-- luapyre: typed
local total: integer = 0
for i = 1, 6 do
    if {condition} then total = total + 1 else total = total + 0 end
end
return total
"""
    assert LuaRuntime(jit_threshold=1).execute(source) == LuaRuntime(jit=False).execute(source) == expected


@pytest.mark.parametrize("key", ["1", "1.0", "100", "'value'"])
def test_invariant_field_refreshes_on_each_runner_entry(key):
    runtime = LuaRuntime(jit_threshold=1)
    runtime.execute("point = {7, [100] = 7, value = 7}")
    proto = runtime.compile(f"""-- luapyre: typed
global point: table
local values = point
local total: integer = 0
for i = 1, 8 do
    local value: integer = values[{key}]
    total = total + value
end
return total
""")
    assert runtime.vm.run(proto) == 56
    runtime.execute(f"point[{key}] = 9")
    assert runtime.vm.run(proto) == 72
    runtime.execute(f"point[{key}] = true")
    with pytest.raises(LuaRuntimeError, match="expected integer"):
        runtime.vm.run(proto)
    runtime.execute(f"point[{key}] = nil")
    with pytest.raises(LuaRuntimeError, match="expected integer"):
        runtime.vm.run(proto)
    runtime.execute("setmetatable(point, {__index = function() return 11 end})")
    assert runtime.vm.run(proto) == 88


def test_constant_field_is_not_invariant_across_aliased_writes():
    source = """-- luapyre: typed
local point = {value = 0}
local alias = point
local total: integer = 0
for i = 1, 8 do
    alias.value = i
    local value: integer = point.value
    total = total + value
end
return total
"""
    assert LuaRuntime(jit_threshold=1).execute(source) == LuaRuntime(jit=False).execute(source) == 36


def test_integral_float_pic_tracks_sparse_to_dense_migration():
    runtime = LuaRuntime()
    fn = runtime.execute_python("point = {[2] = 7}; return function(k) return point[k] end")
    assert fn(2.0) == fn(2.0) == 7
    runtime.execute("point[1] = 3; point[2] = 9")
    assert fn(2.0) == fn(2) == 9
    runtime.execute("point[2] = nil")
    assert fn(2.0) is None


@pytest.mark.parametrize("annotation", [int, LuaInt, int | str])
def test_scalar_conversion_keeps_strict_types_and_unions(annotation):
    runtime = LuaRuntime(jit_threshold=1)
    fn = runtime.execute_python("-- luapyre: typed\nreturn function(x: integer): integer return x end")
    for value in (-(1 << 63), -1, 0, (1 << 63) - 1):
        assert fn(value, return_type=annotation) == value
    for value in (-(1 << 63) - 1, 1 << 63):
        with pytest.raises(OverflowError, match="signed 64-bit"):
            fn(value, return_type=annotation)
    with pytest.raises(LuaRuntimeError, match="expected integer"):
        fn(True, return_type=annotation)
    with pytest.raises(TypeError):
        runtime.execute_python("return true", return_type=annotation)


@pytest.mark.parametrize("value", [-(1 << 100), -(1 << 63) - 1, 1 << 63, 1 << 100])
def test_untyped_python_integers_still_wrap(value):
    runtime = LuaRuntime()
    identity = runtime.execute_python("return function(x) return x end")
    assert identity(value) == ((value + (1 << 63)) % (1 << 64)) - (1 << 63)
