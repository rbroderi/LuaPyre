from __future__ import annotations

import pytest

from luapyre import LuaQuotaError, LuaRuntime, LuaRuntimeError
from luapyre.virtual_frame import compile_virtual_frame


def test_region_guard_uses_literal_type_and_is_inlined():
    runtime = LuaRuntime(jit_threshold=1)
    proto = runtime.compile("""-- luapyre: typed
local values = {1, 2, 3}
local total: integer = 0
for round = 1, 4 do
    for i = 1, 3 do
        local value: integer = values[i]
        total = total + value
    end
end
return total
""")
    assert runtime.vm.run(proto) == 24
    runners = (
        state.runner for state in runtime.vm._jit_loop_states.values()
        if hasattr(state, "runner")
    )
    assert any("_type_matches" not in runner.__code__.co_names for runner in runners)


def test_fresh_table_registration_preserves_gc_accounting():
    runtime = LuaRuntime()
    collector = runtime.vm.gc
    allocations = collector.stats.allocations
    allocated_bytes = collector.stats.allocated_bytes
    table = runtime.vm._new_table()
    assert table._gc_owner is collector
    assert table._gc_age == 0
    assert collector.stats.allocations == allocations + 1
    assert collector.stats.allocated_bytes == allocated_bytes + collector._object_size(table)


def test_virtual_frame_literal_guard_keeps_exact_argument_errors():
    runtime = LuaRuntime(jit_threshold=1)
    fn = runtime.execute_python("""-- luapyre: typed
return function(value: integer): integer return value end
""")
    compiled = compile_virtual_frame(fn.raw.proto, arg_count=1)
    assert compiled is not None
    with pytest.raises(LuaRuntimeError, match="argument 1: expected integer, got boolean"):
        compiled(runtime.vm, [], fn.raw, (True,), -1, 1, 100, [0])


def test_recycled_materialized_frame_adapters_keep_argument_rules():
    runtime = LuaRuntime(jit_threshold=1)
    fn = runtime.execute_python("""-- luapyre: typed
return function(first: integer, second: string): integer
    local scratch = {first}
    return scratch[1]
end
""")
    assert fn(7, "ok", "ignored") == 7
    assert fn(8, "again") == 8
    with pytest.raises(LuaRuntimeError, match="argument 1: expected integer"):
        fn(True, "bad")
    with pytest.raises(LuaRuntimeError, match="argument 2: expected string"):
        fn(1)


def test_bound_python_entry_refreshes_dynamic_fuel_and_argument_types():
    runtime = LuaRuntime(jit_threshold=1)
    fn = runtime.execute_python("""-- luapyre: typed
return function(value: integer): integer return value + 1 end
""")
    assert fn(1, return_type=int) == 2
    assert fn(2, return_type=int) == 3
    assert fn._leaf_cache[0][0] is fn.raw.proto
    with pytest.raises(LuaRuntimeError, match="expected integer"):
        fn(True, return_type=int)
    with pytest.raises(LuaQuotaError):
        fn(3, return_type=int, fuel=0)
    direct_entries = runtime.jit_stats.python_direct_entries
    runtime.vm.jit.enabled = False
    assert fn(4, return_type=int) == 5
    assert runtime.jit_stats.python_direct_entries == direct_entries


def test_warmed_whole_function_inlines_small_upvalue_leaf():
    runtime = LuaRuntime(jit_threshold=2, fuel=2_000_000)
    proto = runtime.compile("""-- luapyre: typed
local function weight(i: integer, j: integer): float
    return 1.0 / (i + j + 1.0)
end
local function total(n: integer): float
    local value: float = 0.0
    for i = 1, n do
        value = value + weight(i, i - 1)
    end
    return value
end
return total(100)
""")
    expected = LuaRuntime(jit=False).vm.run(proto, fuel=2_000_000)
    for _ in range(5):
        assert runtime.vm.run(proto, fuel=2_000_000) == expected
    before = runtime.jit_stats.virtual_frame_elisions
    assert runtime.vm.run(proto, fuel=2_000_000) == expected
    assert runtime.jit_stats.virtual_frame_elisions == before


def test_inlined_leaf_specializes_only_nonzero_constant_divisor():
    runtime = LuaRuntime(jit_threshold=1)
    fn = runtime.execute_python("""-- luapyre: typed
local function half(value: integer): float return value / 2.0 end
local function apply(value: integer): float return half(value) end
return apply
""")
    assert fn(7, return_type=float) == 3.5
    assert fn(-7, return_type=float) == -3.5


def test_inlined_whole_function_call_preserves_small_fuel_outcomes():
    source = """-- luapyre: typed
local function bump(value: integer): integer return value + 1 end
local function total(n: integer): integer
    local value: integer = 0
    for i = 1, n do value = bump(value) end
    return value
end
return total(4)
"""
    hot = LuaRuntime(jit_threshold=1)
    proto = hot.compile(source)
    for _ in range(3):
        assert hot.vm.run(proto, fuel=1000) == 4
    for fuel in range(45):
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
