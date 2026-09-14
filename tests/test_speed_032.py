from __future__ import annotations

import pytest

from luapyre import LuaQuotaError, LuaRuntime, LuaRuntimeError


NUMERIC_UPVALUE = """-- luapyre: typed
local function weight(i: integer, j: integer): float
    return 1.0 / ((i + j) * (i + j + 1) / 2 + i + 1)
end
local function sum(n: integer): float
    local total: float = 0.0
    for i = 1, n do
        total = total + weight(i, i - 1)
    end
    return total
end
return sum(40)
"""


def outcome(source: str, fuel: int, *, jit: bool):
    try:
        return ("return", LuaRuntime(jit=jit, jit_threshold=1).execute(source, fuel=fuel))
    except (LuaQuotaError, LuaRuntimeError) as error:
        return (type(error).__name__, str(error))


def test_final_call_entry_is_cached_with_proven_argument_adapter():
    runtime = LuaRuntime(jit_threshold=1)
    assert runtime.execute("""-- luapyre: typed
local function fib(n: integer): integer
    if n < 2 then return n end
    return fib(n - 1) + fib(n - 2)
end
return fib(10)
""") == 55
    entries = [entry for _proto, entry in runtime.vm.jit._call_entry_cache.values()]
    assert entries
    assert any(entry.trusted_args and entry.arg_count == 1 for entry in entries)
    entry = entries[0]
    compiled = runtime.vm.jit._function_cache[id(entry.proto)][1]
    assert runtime.vm.jit.get_call_entry(
        compiled, arg_count=entry.arg_count, trusted_args=entry.trusted_args
    ) is entry


def test_upvalue_loaded_numeric_leaf_is_inlined_in_structured_loop():
    runtime = LuaRuntime(jit_threshold=1)
    expected = LuaRuntime(jit=False).execute(NUMERIC_UPVALUE)
    assert runtime.execute(NUMERIC_UPVALUE) == expected
    runners = [
        state.runner for state in runtime.vm._jit_loop_states.values()
        if hasattr(state, "runner")
    ]
    assert any(runner.__name__ == "_jit_structured_loop" for runner in runners)
    assert runtime.jit_stats.virtual_frame_elisions == 0


def test_inlined_numeric_leaf_preserves_every_small_fuel_boundary():
    source = NUMERIC_UPVALUE.replace("sum(40)", "sum(4)")
    for fuel in range(90):
        assert outcome(source, fuel, jit=True) == outcome(source, fuel, jit=False), fuel


def test_inlined_numeric_leaf_preserves_logical_stack_limit():
    runtime = LuaRuntime(jit_threshold=1, max_frames=100)
    assert runtime.execute(NUMERIC_UPVALUE) > 0
    runtime.vm.max_frames = 1
    with pytest.raises(LuaRuntimeError, match="stack overflow"):
        runtime.execute(NUMERIC_UPVALUE)


def test_fixed_arity_paths_keep_missing_extra_and_boolean_errors():
    runtime = LuaRuntime(jit_threshold=1)
    fn = runtime.execute_python("""-- luapyre: typed
return function(x: integer): integer
    if x > 0 then return x end
    return 0
end
""")
    assert fn(7, 8, 9) == 7
    with pytest.raises(LuaRuntimeError, match="argument 1: expected integer"):
        fn()
    with pytest.raises(LuaRuntimeError, match="argument 1: expected integer"):
        fn(True)


def test_inlined_adapter_does_not_treat_integer_as_strict_float():
    source = """-- luapyre: typed
local function identity(x: float): float return x end
local total: float = 0.0
for i = 1, 3 do total = total + identity(i) end
return total
"""
    for jit in (False, True):
        with pytest.raises(LuaRuntimeError, match="expected float, got integer"):
            LuaRuntime(jit=jit, jit_threshold=1).execute(source)
