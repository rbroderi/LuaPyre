"""Correctness and code-shape coverage for the 0.39 sieve paths."""
from __future__ import annotations

import pytest

from luapyre import LuaQuotaError, LuaRuntime
from luapyre.jit import CompiledLoop
from luapyre.table import LuaTable


SMALL_SIEVE = """-- luapyre: typed
local composite: table = {}
local count: integer = 0
for p = 2, 40 do
    if not composite[p] then
        count = count + 1
        local multiple: integer = p * p
        while multiple <= 40 do
            composite[multiple] = true
            multiple = multiple + p
        end
    end
end
return count
"""


def _outcome(runtime, proto, fuel):
    try:
        return ("return", runtime.vm.run(proto, fuel=fuel))
    except LuaQuotaError:
        return ("quota", None)


def test_cached_numeric_loop_enters_from_forprep():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    proto = runtime.compile(SMALL_SIEVE)
    assert runtime.vm.run(proto) == 12
    before = runtime.jit_stats.loop_entry_executions
    assert runtime.vm.run(proto) == 12
    assert runtime.jit_stats.loop_entry_executions == before + 1


def test_forprep_entry_preserves_every_fuel_boundary():
    hot = LuaRuntime(jit_threshold=1)
    hot_proto = hot.compile(SMALL_SIEVE)
    for _ in range(3):
        assert hot.vm.run(hot_proto, fuel=20_000) == 12
    cold = LuaRuntime(jit=False)
    cold_proto = cold.compile(SMALL_SIEVE)
    first_complete = next(
        fuel for fuel in range(2_000)
        if _outcome(cold, cold_proto, fuel)[0] == "return"
    )
    for fuel in range(first_complete + 3):
        assert _outcome(hot, hot_proto, fuel) == _outcome(
            cold, cold_proto, fuel
        ), fuel


def test_forprep_entry_skips_zero_trip_loop():
    runtime = LuaRuntime(jit_threshold=1)
    runtime.globals.rawset(b"limit", 8)
    proto = runtime.compile("""-- luapyre: typed
global limit: integer
local total: integer = 0
for i = 1, limit do total = total + i end
return total
""")
    for _ in range(2):
        assert runtime.vm.run(proto) == 36
    before = runtime.jit_stats.loop_entry_executions
    runtime.globals.rawset(b"limit", 0)
    assert runtime.vm.run(proto) == 0
    assert runtime.jit_stats.loop_entry_executions == before


def test_forprep_entry_preserves_negative_step_loop():
    runtime = LuaRuntime(jit_threshold=1)
    proto = runtime.compile(
        "local total = 0; for i = 9, 1, -2 do total = total + i end; return total"
    )
    assert runtime.vm.run(proto) == 25
    before = runtime.jit_stats.loop_entry_executions
    assert runtime.vm.run(proto) == 25
    assert runtime.jit_stats.loop_entry_executions == before + 1


def test_sparse_integer_boolean_region_materializes_for_generic_semantics():
    runtime = LuaRuntime(jit_threshold=1)
    table = runtime.vm._new_table()
    table._sparse_int = {4: True, 6: True, 8: True}
    assert table.rawget(4) is True
    assert table.rawget(4.0) is True
    assert table.rawget(5) is None
    assert dict(table.items())[4] is True
    assert table.next_item() is not None

    version = table.version
    child = LuaTable()
    table.rawset(100, child)
    assert table._sparse_int is None
    assert table.version == version + 1
    assert table.rawget(4) is True and table.rawget(100) is child
    assert child._gc_owner is runtime.vm.gc

    table.rawset(1, b"dense")
    assert table.rawlen() == 1
    assert table.rawget(1) == b"dense"
    table.rawset(4, None)
    assert table.rawget(4) is None


def test_sparse_codegen_uses_direct_dictionary_and_primitive_barrier_path():
    runtime = LuaRuntime(jit_threshold=1)
    proto = runtime.compile(SMALL_SIEVE)
    for _ in range(3):
        assert runtime.vm.run(proto) == 12
    runner = next(
        state.runner
        for state in runtime.vm._jit_loop_states.values()
        if isinstance(state, CompiledLoop) and "_sparse_int" in state.runner.__code__.co_names
    )
    names = set(runner.__code__.co_names)
    assert "_sparse_int" in names
    assert "get" in names
    assert "table_write_barrier" not in names


def test_sparse_region_requires_table_to_be_dead_after_loop():
    runtime = LuaRuntime(jit_threshold=1)
    proto = runtime.compile("""-- luapyre: typed
local marked: table = {}
for p = 2, 20 do
    local multiple: integer = p * p
    while multiple <= 80 do
        marked[multiple] = true
        multiple = multiple + p
    end
end
return marked[4], marked[6], marked[7]
""")
    for _ in range(3):
        assert runtime.vm.run(proto) == (True, True, None)
    outer = next(
        state
        for state in runtime.vm._jit_loop_states.values()
        if isinstance(state, CompiledLoop) and state.ir.start_pc == 9
    )
    assert "_sparse_int" not in outer.runner.__code__.co_names


def test_active_debug_hook_keeps_forprep_on_interpreter_path():
    runtime = LuaRuntime(jit_threshold=1, debug_hooks=True)
    proto = runtime.compile("""-- luapyre: typed
global debug: table
local hits: integer = 0
debug.sethook(function(): nil hits = hits + 1 end, "l")
local total: integer = 0
for i = 1, 20 do total = total + i end
debug.sethook()
return total, hits > 0
""")
    for _ in range(2):
        assert runtime.vm.run(proto) == (210, True)
    assert runtime.jit_stats.loop_entry_executions == 0
