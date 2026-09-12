from __future__ import annotations

from luapyre import LuaQuotaError, LuaRuntime, LuaRuntimeError


_SOURCE = """-- luapyre: typed
local function outer(a: integer, b: integer): integer
    local function add(x: integer, y: integer): integer
        local first = x + y
        local same = x + y
        local dead = x * y
        return same
    end
    local result = add(a, b)
    return result + 1
end
local answer: integer = outer(10, 20)
return answer
"""


def _outcome(*, jit: bool, fuel: int):
    runtime = LuaRuntime(jit=jit, jit_threshold=1, fuel=fuel)
    try:
        return ("ok", runtime.execute(_SOURCE))
    except LuaQuotaError as exc:
        return ("quota", str(exc))


def test_static_call_inline_has_identical_quota_boundary_to_interpreter():
    # The optimized whole-function runner performs one aggregate budget check.
    # If the complete parent + inlined child cannot fit, it suspends at pc 0 and
    # Tier 0 consumes fuel instruction-by-instruction. Therefore every budget
    # around the transition must have the same success/quota outcome and error.
    for fuel in range(1, 65):
        assert _outcome(jit=True, fuel=fuel) == _outcome(jit=False, fuel=fuel)


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


def test_static_call_inline_preserves_child_frame_stack_limit():
    # root -> outer fills a two-frame VM. The lexical add() would need a third
    # frame and must therefore fail exactly as Tier 0 does. Value IR detects that
    # its omitted child frame would overflow and abandons the optimized function
    # before pc 0, letting the interpreter raise at the real CALL.
    assert _stack_outcome(jit=True, max_frames=2) == _stack_outcome(
        jit=False, max_frames=2
    )
    assert _stack_outcome(jit=True, max_frames=2)[0] == "runtime"

    # With one slot available, both paths complete normally.
    assert _stack_outcome(jit=True, max_frames=3) == _stack_outcome(
        jit=False, max_frames=3
    ) == ("ok", 31)
