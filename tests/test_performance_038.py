"""Correctness and code-shape coverage for the 0.38 performance paths."""
from __future__ import annotations

import pytest

from luapyre import LuaQuotaError, LuaRuntime
from luapyre.bytecode import Op
from luapyre.function_jit import _FUNC_RETURN, _pure_return_prefix
from luapyre.table import LuaTable, _hash_key


FIBONACCI = """-- luapyre: typed
local function fib(n: integer): integer
    if n < 2 then return n end
    return fib(n - 1) + fib(n - 2)
end
return fib
"""


def test_pure_base_case_scalar_entry_avoids_a_child_frame():
    runtime = LuaRuntime(jit_threshold=1)
    function = runtime.execute_python(FIBONACCI)
    compiled = runtime.vm.jit.get_compiled_function(function.raw)
    assert compiled is not None
    assert _pure_return_prefix(function.raw.proto) is not None
    entry = runtime.vm.jit.get_call_entry(
        compiled, arg_count=1, trusted_args=True
    )
    assert entry.scalar_runner is not None
    frames = []
    meter = [0]
    status, value = entry.scalar_runner(
        runtime.vm, frames, function.raw, 1, 0, 1, 100, meter
    )
    assert (status, value, frames) == (_FUNC_RETURN, 1, [])
    assert meter[0] == _pure_return_prefix(function.raw.proto).instruction_cost


def test_recursive_scalar_entry_preserves_every_fuel_boundary():
    for fuel in range(1, 70):
        outcomes = []
        for jit in (False, True):
            runtime = LuaRuntime(jit=jit, jit_threshold=1)
            function = runtime.execute_python(FIBONACCI)
            try:
                outcomes.append(("return", function(7, fuel=fuel)))
            except LuaQuotaError:
                outcomes.append(("quota", None))
        assert outcomes[0] == outcomes[1]


def test_two_argument_materialized_entry_has_a_scalar_interface():
    runtime = LuaRuntime(jit_threshold=1)
    function = runtime.execute_python("""-- luapyre: typed
return function(left: integer, right: integer): integer
    local scratch = {left}
    return scratch[1] + right
end
""")
    compiled = runtime.vm.jit.get_compiled_function(function.raw)
    assert compiled is not None
    entry = runtime.vm.jit.get_call_entry(
        compiled, arg_count=2, trusted_args=True
    )
    assert not entry.virtual and entry.scalar_runner is not None
    frames = []
    status, value = entry.scalar_runner(
        runtime.vm, frames, function.raw, 4, 5, 0, 1, 100, [0]
    )
    assert (status, value, frames) == (_FUNC_RETURN, 9, [])


def test_nested_numeric_loop_proves_one_dense_region():
    runtime = LuaRuntime(jit_threshold=1)
    proto = runtime.compile("""-- luapyre: typed
local values: table = {}
for i = 1, 64 do values[i] = i end
local total: integer = 0
for round = 1, 4 do
    for i = 1, 64 do
        local value: integer = values[i]
        values[i] = value + round
        total = total + values[i]
    end
end
return total
""")
    for _ in range(4):
        assert runtime.vm.run(proto) == 9600
    runners = [
        state.runner
        for state in runtime.vm._jit_loop_states.values()
        if hasattr(state, "runner")
    ]
    dense = next(
        runner for runner in runners
        if "array" in runner.__code__.co_names
        and "range" in runner.__code__.co_names
        and "rawget" not in runner.__code__.co_names
        and "rawset" not in runner.__code__.co_names
    )
    assert "len" in dense.__code__.co_names


def test_fresh_constructor_fields_keep_nil_version_gc_and_iteration_rules():
    runtime = LuaRuntime()
    table = runtime.vm._new_table()
    token = _hash_key(b"left")
    before_bytes = runtime.vm.gc.stats.allocated_bytes
    table.rawset_fresh_prehashed(b"left", token, None)
    assert table.version == 1 and table.hash == {}
    assert table._deleted_successors is None
    child = LuaTable()
    table.rawset_fresh_prehashed(b"left", token, child)
    assert table.version == 2 and table.rawget(b"left") is child
    assert child._gc_owner is runtime.vm.gc
    assert runtime.vm.gc.stats.allocated_bytes == (
        before_bytes + 32 + runtime.vm.gc._object_size(child)
    )
    assert table.next_item() == (b"left", table.rawget(b"left"))


def test_source_compiler_marks_only_proven_new_record_keys():
    proto = LuaRuntime().compile(
        "return {left=1, right=2, left=3, [unknown]=4, tail=5}"
    )
    writes = [ins for ins in proto.code if ins.op is Op.SETTABLE]
    assert [ins.d for ins in writes] == [1, 1, 0, 0, 0]
