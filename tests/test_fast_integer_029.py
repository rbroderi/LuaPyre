from __future__ import annotations

import pytest

from luapyre import LuaQuotaError, LuaRuntime, LuaRuntimeError


def test_structured_dense_table_loop_uses_batched_range_lowering():
    runtime = LuaRuntime(jit_threshold=1)
    proto = runtime.compile(
        "-- luapyre: typed\n"
        "local values = {10, 20, 12}\n"
        "local total: integer = 0\n"
        "for i = 1, 3 do\n"
        "  local value: integer = values[i]\n"
        "  total = total + value\n"
        "end\n"
        "return total\n"
    )

    assert runtime.vm.run(proto) == 42
    compiled = next(
        state
        for state in runtime.vm._jit_loop_states.values()
        if hasattr(state, "runner")
    )
    assert compiled.runner.__name__ == "_jit_structured_loop"
    assert runtime.vm.jit.stats.deopts == 0


def test_batched_integer_loop_preserves_exact_fuel_boundary():
    source = (
        "-- luapyre: typed\n"
        "local total: integer = 0\n"
        "for i = 1, 100 do total = total + i end\n"
        "return total\n"
    )
    for jit in (False, True):
        runtime = LuaRuntime(jit=jit, jit_threshold=1)
        with pytest.raises(LuaQuotaError):
            runtime.execute(source, fuel=410)
        assert runtime.execute(source, fuel=411) == 5050


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("expression", [
    "return 0x7fffffffffffffff + 1",
    "local x = 0x7fffffffffffffff + 0; return x + 1",
    "local result = 0; for i = 1, 4 do result = i + 0x7fffffffffffffff end; return result - 3",
])
def test_inferred_expressions_never_establish_an_overflow_contract(jit, expression):
    lua = LuaRuntime(jit=jit, jit_threshold=1)
    assert lua.execute("-- luapyre: typed\n" + expression) == -(1 << 63)


@pytest.mark.parametrize("jit", [False, True])
def test_profiled_table_values_can_change_type_inside_typed_loop(jit):
    lua = LuaRuntime(jit=jit, jit_threshold=1)
    assert lua.execute("""-- luapyre: typed
local values = {1, 1, true, false, "1"}
local total = 0
for i = 1, 5 do
    if values[i] == 1 then total = total + 1 end
end
return total
""") == 2


@pytest.mark.parametrize("jit", [False, True])
def test_modulo_keeps_integer_precision_after_float_warmup(jit):
    lua = LuaRuntime(jit=jit, jit_threshold=1)
    result = lua.execute("""-- luapyre: typed
local values = {7.0, 7.0, 9007199254740995}
local out = {}
for i = 1, 3 do out[i] = values[i] % 3 end
return out[1], out[2], out[3]
""")
    assert result == (1.0, 1.0, 2)
    assert tuple(map(type, result)) == (float, float, int)


@pytest.mark.parametrize("jit", [False, True])
def test_compiled_child_entry_checks_arguments(jit):
    lua = LuaRuntime(jit=jit, jit_threshold=1)
    function = lua.execute_python("""-- luapyre: typed
local function caller(): integer
    local function child(x: integer): integer return x end
    local result: integer = child(true)
    return result
end
return caller
""")
    with pytest.raises(LuaRuntimeError, match="argument 1: expected integer, got boolean"):
        function()


@pytest.mark.parametrize("jit", [False, True])
def test_inline_call_checks_current_iteration_arguments(jit):
    lua = LuaRuntime(jit=jit, jit_threshold=1)
    with pytest.raises(LuaRuntimeError, match="argument 1: expected integer, got boolean"):
        lua.execute("""-- luapyre: typed
local function child(x: integer): integer return x end
local values = {1, 2, true}
local total: integer = 0
for i = 1, 3 do total = child(values[i]) end
return total
""")


@pytest.mark.parametrize("jit", [False, True])
def test_fast_integer_python_input_is_checked_before_wrapping(jit):
    lua = LuaRuntime(jit=jit, jit_threshold=1)
    function = lua.execute_python("""-- luapyre: typed
local function identity(x: integer): integer return x end
return identity
""")
    with pytest.raises(OverflowError, match="signed 64-bit"):
        function(1 << 63)


def test_call_cache_releases_arguments_and_handles_reentrant_entry():
    lua = LuaRuntime(jit_threshold=1)
    inner = []

    def callback(value: int) -> int:
        if value == 1:
            inner.append(function(2))
        return value

    lua.set("callback", callback)
    function = lua.execute_python("return function(x) return callback(x), x end")
    assert function(1) == (1, 1)
    assert inner == [(2, 2)]
    for _, proto in lua._python_call_cache.values():
        assert proto.constants[1:] == [None]
