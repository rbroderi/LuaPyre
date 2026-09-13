from __future__ import annotations

import pytest

from luapyre import LuaQuotaError, LuaRuntime, LuaRuntimeError


DIAMOND = """-- luapyre: typed
local total: integer = 0
for i = 1, 100 do
    if i % 2 == 0 then
        total = total + i
    else
        total = total - 1
    end
end
return total
"""


def test_typed_diamond_loop_lowers_to_direct_python_control_flow():
    runtime = LuaRuntime(jit_threshold=1)
    proto = runtime.compile(DIAMOND)
    assert runtime.vm.run(proto) == 2500
    compiled = next(
        state for state in runtime.vm._jit_loop_states.values()
        if hasattr(state, "runner")
    )
    assert compiled.runner.__name__ == "_jit_structured_cfg_loop"
    assert "_state" not in compiled.runner.__code__.co_varnames
    assert runtime.jit_stats.deopts == 0


def test_typed_diamond_loop_matches_interpreter_fuel_boundary():
    def minimum(jit: bool) -> int:
        for fuel in range(1, 2000):
            try:
                if LuaRuntime(jit=jit, jit_threshold=1).execute(DIAMOND, fuel=fuel) == 2500:
                    return fuel
            except LuaQuotaError:
                pass
        raise AssertionError("fuel boundary not found")

    assert minimum(True) == minimum(False)


def test_structured_inline_guards_are_primitive_checks():
    runtime = LuaRuntime(jit_threshold=1)
    proto = runtime.compile("""-- luapyre: typed
local values = {1, 2, 3}
local total: integer = 0
for i = 1, 3 do
    local value: integer = values[i]
    total = total + value
end
return total
""")
    assert runtime.vm.run(proto) == 6
    compiled = next(
        state for state in runtime.vm._jit_loop_states.values()
        if hasattr(state, "runner")
    )
    assert "_type_matches" not in compiled.runner.__code__.co_names


def test_constant_table_key_is_lowered_to_direct_hash_lookup():
    runtime = LuaRuntime(jit_threshold=1)
    proto = runtime.compile("""-- luapyre: typed
local point = {x = 7}
local total: integer = 0
for i = 1, 100 do
    local x: integer = point.x
    total = total + x
end
return total
""")
    assert runtime.vm.run(proto) == 700
    compiled = next(
        state for state in runtime.vm._jit_loop_states.values()
        if hasattr(state, "runner")
    )
    assert "rawget" not in compiled.runner.__code__.co_names
    assert "get" in compiled.runner.__code__.co_names


def test_direct_python_entry_preserves_type_errors_and_quota():
    runtime = LuaRuntime(jit_threshold=1)
    fn = runtime.execute_python(
        "-- luapyre: typed\nreturn function(x: integer): integer return x + 1 end"
    )
    assert fn(41, return_type=int) == 42
    assert runtime.jit_stats.python_direct_entries == 1
    with pytest.raises(LuaRuntimeError, match="expected integer"):
        fn(True)
    with pytest.raises(LuaQuotaError):
        fn(1, fuel=0)


def test_recursive_compiled_calls_reuse_target_and_frames():
    runtime = LuaRuntime(jit_threshold=1)
    assert runtime.execute("""-- luapyre: typed
local function fib(n: integer): integer
    if n < 2 then return n end
    return fib(n - 1) + fib(n - 2)
end
return fib(12)
""") == 144
    assert runtime.vm.jit.function_executions > 100
    assert runtime.jit_stats.compiled_frame_allocations < 20


def test_diamond_table_equality_deopts_to_exact_metamethod_semantics():
    runtime = LuaRuntime(jit_threshold=1)
    assert runtime.execute("""-- luapyre: typed
global setmetatable: function
local mt = {__eq = function(a: table, b: table): boolean return true end}
local left = setmetatable({}, mt)
local right = setmetatable({}, mt)
local total: integer = 0
for i = 1, 10 do
    if left == right then total = total + 1 else total = total - 1 end
end
return total
""") == 10
    assert runtime.jit_stats.deopts >= 1
