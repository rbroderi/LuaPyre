from __future__ import annotations

import pytest

from luapyre import LuaQuotaError, LuaRuntime


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
