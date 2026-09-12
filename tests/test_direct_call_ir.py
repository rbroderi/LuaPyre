from __future__ import annotations

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


def test_direct_call_ir_keeps_real_child_frame_stack_limit():
    # root -> outer uses two frames. choose() must still require a third frame.
    assert _stack_outcome(jit=True, max_frames=2) == _stack_outcome(
        jit=False, max_frames=2
    )
    assert _stack_outcome(jit=True, max_frames=2)[0] == "runtime"

    assert _stack_outcome(jit=True, max_frames=3) == _stack_outcome(
        jit=False, max_frames=3
    ) == ("ok", 11)
