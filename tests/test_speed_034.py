from __future__ import annotations

import dis

import pytest

from luapyre import LuaQuotaError, LuaRuntime, LuaRuntimeError
from luapyre.table import LuaTable, _hash_key


def test_typed_string_concat_uses_direct_bytes_after_type_proof():
    runtime = LuaRuntime(jit_threshold=1)
    proto = runtime.compile("""-- luapyre: typed
local value: string = ""
for i = 1, 20 do
    value = value .. "a\0b"
end
return value
""")
    expected = b"a\0b" * 20
    assert runtime.vm.run(proto) == expected
    assert runtime.vm.run(proto) == expected
    runners = (
        state.runner for state in runtime.vm._jit_loop_states.values()
        if hasattr(state, "runner")
    )
    assert any("_to_lua_string" not in runner.__code__.co_names for runner in runners)


def test_gc_threshold_cache_tracks_modes_parameters_and_collections():
    runtime = LuaRuntime()
    collector = runtime.vm.gc
    assert collector._threshold() == collector._threshold_value

    collector.command(b"param", b"stepsize", 20)
    assert collector._threshold() == 1 << 20

    assert collector.command(b"incremental") == b"generational"
    collector.command(b"param", b"pause", 250)
    baseline = collector.stats.approximate_bytes or 1 << 20
    expected = max(1 << 20, baseline * 250 // 100)
    assert collector._threshold() == expected

    collector.collect()
    baseline = collector.stats.approximate_bytes or 1 << 20
    expected = max(1 << 20, baseline * 250 // 100)
    assert collector._threshold() == expected


def test_structured_diamond_batches_a_fully_funded_integer_loop():
    runtime = LuaRuntime(jit_threshold=1)
    proto = runtime.compile("""-- luapyre: typed
local total: integer = 0
for i = 1, 100 do
    if i % 2 == 0 then total = total + i else total = total - 1 end
end
return total
""")
    assert runtime.vm.run(proto, fuel=10_000) == 2500
    assert runtime.vm.run(proto, fuel=10_000) == 2500
    runners = [
        state.runner for state in runtime.vm._jit_loop_states.values()
        if hasattr(state, "runner")
    ]
    assert any(
        runner.__code__.co_filename == "<luapyre-structured-cfg-loop>"
        and "range" in runner.__code__.co_names
        for runner in runners
    )


def test_warmed_diamond_uses_cpython_range_and_integer_specializations():
    runtime = LuaRuntime(jit_threshold=1)
    proto = runtime.compile("""-- luapyre: typed
local total: integer = 0
for i = 1, 100 do
    if i % 2 == 0 then total = total + i else total = total - 1 end
end
return total
""")
    for _ in range(2_000):
        assert runtime.vm.run(proto) == 2500
    runner = next(
        state.runner for state in runtime.vm._jit_loop_states.values()
        if hasattr(state, "runner")
        and state.runner.__code__.co_filename == "<luapyre-structured-cfg-loop>"
    )
    opnames = {
        instruction.opname
        for instruction in dis.get_instructions(runner, adaptive=True)
    }
    assert "FOR_ITER_RANGE" in opnames
    assert "BINARY_OP_ADD_INT" in opnames


def test_batched_diamond_matches_interpreter_at_every_fuel_boundary():
    source = """-- luapyre: typed
local total: integer = 0
for i = 1, 4 do
    if i % 2 == 0 then total = total + i else total = total - 1 end
end
return total
"""
    hot = LuaRuntime(jit_threshold=1)
    proto = hot.compile(source)
    for _ in range(3):
        assert hot.vm.run(proto, fuel=1000) == 4
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


def test_bound_integer_adapter_refreshes_runtime_state_and_falls_back():
    runtime = LuaRuntime(jit_threshold=1)
    function = runtime.execute_python("""-- luapyre: typed
return function(value: integer): integer return value + 1 end
""")
    assert function(1, return_type=int) == 2
    adapter = function._adapter_cache[0]
    assert callable(adapter)
    assert function(2, return_type=int) == 3

    # A different argument shape retains generic conversion and Lua errors.
    assert function(4, "ignored", return_type=int) == 5
    with pytest.raises(LuaRuntimeError, match="expected integer"):
        function(True, return_type=int)

    runtime.vm.jit.enabled = False
    assert function(3, return_type=int) == 4
    runtime.vm.jit.enabled = True

    with pytest.raises(LuaQuotaError, match="quota"):
        function(5, return_type=int, fuel=0)


def test_prehashed_constant_table_write_preserves_deletion_successor():
    table = LuaTable()
    first = b"first"
    second = b"second"
    table.rawset_prehashed(first, _hash_key(first), 1)
    table.rawset_prehashed(second, _hash_key(second), 2)
    version = table.version
    table.rawset_prehashed(first, _hash_key(first), None)
    assert table.version == version + 1
    assert table.rawget(first) is None
    assert table.successor_after_deleted(first) == (True, second)
