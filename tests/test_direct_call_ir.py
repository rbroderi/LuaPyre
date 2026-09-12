from __future__ import annotations

import pytest

from luapyre import LuaQuotaError, LuaRuntime, LuaRuntimeError
from luapyre.call_ir import analyze_direct_calls


_SOURCE = """-- luapyre: typed
local function outer(x: integer): integer
    local function choose(y: integer): integer
        if y < 0 then
            return y - 1
        end
        return y + 1
    end
    local result: integer = choose(x)
    return result
end
local answer: integer = outer(10)
return answer
"""


def _outer(runtime: LuaRuntime):
    root = runtime.compile(_SOURCE)
    return root, next(child for child in root.children if child.name == "outer")


def _compiled_filenames(runtime: LuaRuntime) -> set[str]:
    return {
        compiled.runner.__code__.co_filename
        for _proto, compiled in runtime.vm.jit._function_cache.values()
        if compiled is not None
    }


def test_branchy_lexical_child_is_direct_call_ir_not_value_inline():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    root, outer = _outer(runtime)

    plan = analyze_direct_calls(outer)
    assert len(plan.direct_sites) == 1
    site = plan.direct_sites[0]
    assert outer.children[site.child_index].name == "choose"

    assert runtime.vm.run(root) == 11
    filenames = _compiled_filenames(runtime)
    assert "<luapyre-direct-call-ir-function>" in filenames
    assert runtime.jit_stats.escape_plans == 1
    assert runtime.jit_stats.virtual_frame_elisions == 1
    assert runtime.jit_stats.virtual_frame_materializations == 0


def _fuel_outcome(*, jit: bool, fuel: int):
    runtime = LuaRuntime(jit=jit, jit_threshold=1, fuel=fuel)
    try:
        return ("ok", runtime.execute(_SOURCE))
    except LuaQuotaError as exc:
        return ("quota", str(exc))


def test_direct_call_ir_preserves_every_nearby_quota_boundary():
    for fuel in range(1, 80):
        assert _fuel_outcome(jit=True, fuel=fuel) == _fuel_outcome(
            jit=False, fuel=fuel
        )


def _stack_outcome(*, jit: bool, max_frames: int):
    runtime = LuaRuntime(
        jit=jit,
        jit_threshold=1,
        fuel=2_000_000,
        max_frames=max_frames,
    )
    try:
        return ("ok", runtime.execute(_SOURCE))
    except LuaRuntimeError as exc:
        return ("runtime", str(exc))


def test_virtual_frame_preserves_real_child_stack_limit():
    # root -> outer uses two frames. choose() must still require a third frame.
    assert _stack_outcome(jit=True, max_frames=2) == _stack_outcome(
        jit=False, max_frames=2
    )
    assert _stack_outcome(jit=True, max_frames=2)[0] == "runtime"

    assert _stack_outcome(jit=True, max_frames=3) == _stack_outcome(
        jit=False, max_frames=3
    ) == ("ok", 11)


def test_virtual_frame_materializes_at_exact_quota_side_exit():
    runtime = LuaRuntime(jit_threshold=1, fuel=13)
    root, _outer_proto = _outer(runtime)
    with pytest.raises(LuaQuotaError):
        runtime.vm.run(root)
    assert runtime.jit_stats.virtual_frame_materializations == 1
    assert runtime.jit_stats.virtual_frame_elisions == 0


_MULTIVALUE_SOURCE = """-- luapyre: typed
local function outer(x: integer): integer, integer
    local function pair(y: integer): integer, integer
        if y < 0 then
            return y - 1, y
        end
        return y + 1, y
    end
    return pair(x)
end
local a: integer, b: integer = outer(10)
return a, b
"""


def _multivalue_outcome(*, jit: bool, fuel: int):
    runtime = LuaRuntime(jit=jit, jit_threshold=1, fuel=fuel)
    root = runtime.compile(_MULTIVALUE_SOURCE)
    try:
        return ("ok", runtime.vm.run(root)), runtime
    except LuaQuotaError as exc:
        return ("quota", str(exc)), runtime


def test_open_results_are_scalar_replaced_until_returnv():
    outcome, runtime = _multivalue_outcome(jit=True, fuel=2_000_000)
    assert outcome == ("ok", (11, 10))
    assert runtime.jit_stats.virtual_frame_elisions == 1
    assert runtime.jit_stats.virtual_multivalue_elisions == 1
    assert runtime.jit_stats.virtual_multivalue_materializations == 0


def test_virtual_multivalue_materializes_only_on_caller_suspension():
    for fuel in range(1, 70):
        jit_outcome, _runtime = _multivalue_outcome(jit=True, fuel=fuel)
        plain_outcome, _plain = _multivalue_outcome(jit=False, fuel=fuel)
        assert jit_outcome == plain_outcome

    _outcome, runtime = _multivalue_outcome(jit=True, fuel=22)
    assert runtime.jit_stats.virtual_multivalue_elisions == 1
    assert runtime.jit_stats.virtual_multivalue_materializations == 1


_ERROR_SOURCE = """-- luapyre: typed
local function outer(n: integer): integer
    local function bad(x: integer): integer
        local total: integer = 0
        for i = 1, x, 0 do total = total + i end
        return total
    end
    local value: integer = bad(n)
    return value
end
local answer: integer = outer(2)
return answer
"""


def _error_outcome(*, jit: bool):
    runtime = LuaRuntime(jit=jit, jit_threshold=1, fuel=2_000_000)
    root = runtime.compile(_ERROR_SOURCE)
    with pytest.raises(LuaRuntimeError) as caught:
        runtime.vm.run(root)
    return (str(caught.value), caught.value.value), runtime


def test_virtual_frame_materializes_before_runtime_error_unwind():
    optimized, runtime = _error_outcome(jit=True)
    baseline, _plain = _error_outcome(jit=False)
    assert optimized == baseline
    assert b":5: 'for' step is zero" in optimized[1]
    assert runtime.jit_stats.virtual_frame_materializations == 1
