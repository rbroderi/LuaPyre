from __future__ import annotations

import dis

from luapyre import LuaRuntime


def test_structured_string_diamond_retains_accumulator_type():
    runtime = LuaRuntime(jit_threshold=1)
    source = """-- luapyre: typed
local value: string = ""
for i = 1, 20 do
    if i % 2 == 0 then value = value .. "a" else value = value .. "b" end
end
return value
"""
    expected = b"ba" * 10
    assert runtime.execute(source) == expected
    assert runtime.execute(source) == expected
    runner = next(
        state.runner
        for state in runtime.vm._jit_loop_states.values()
        if hasattr(state, "runner")
        and state.runner.__code__.co_filename == "<luapyre-structured-cfg-loop>"
    )
    assert "_to_lua_string" not in runner.__code__.co_names


def test_structured_string_diamond_matches_every_fuel_boundary():
    source = """-- luapyre: typed
local value: string = ""
for i = 1, 4 do
    if i % 2 == 0 then value = value .. "a" else value = value .. "b" end
end
return value
"""
    hot = LuaRuntime(jit_threshold=1)
    proto = hot.compile(source)
    for _ in range(3):
        assert hot.vm.run(proto, fuel=1000) == b"baba"
    for fuel in range(55):
        try:
            actual = ("return", hot.vm.run(proto, fuel=fuel))
        except Exception as error:
            actual = (type(error), str(error))
        cold = LuaRuntime(jit=False)
        try:
            expected = ("return", cold.execute(source, fuel=fuel))
        except Exception as error:
            expected = (type(error), str(error))
        assert actual == expected, fuel


def test_scalar_leaf_abi_omits_scratch_containers_and_result_tuple():
    runtime = LuaRuntime(jit_threshold=1)
    function = runtime.execute_python("""-- luapyre: typed
return function(value: integer): integer return value + 1 end
""")
    assert function(1) == 2
    assert function(2) == 3
    compiled = runtime.vm.jit.get_leaf(function._value.proto)
    assert compiled is not None and compiled.scalar_runner is not None
    opnames = {
        instruction.opname
        for instruction in dis.get_instructions(compiled.scalar_runner)
    }
    assert "BUILD_LIST" not in opnames
    assert "BUILD_MAP" not in opnames
    assert "BUILD_TUPLE" not in opnames
    assert "len" not in compiled.scalar_runner.__code__.co_names

