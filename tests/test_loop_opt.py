from __future__ import annotations

from luapyre import LuaQuotaError, LuaRuntime
from luapyre.cfg_value_ir import CFGValueIRCompiler
from luapyre.loop_opt import CFGLoopOptimizer


_SOURCE = """-- luapyre: typed
local function sum_bias(n: integer, a: integer, b: integer): integer
    local i: integer = 0
    local total: integer = 0
    while i < n do
        local bias: integer = a + b
        total = total + bias + i
        i = i + 1
    end
    return total
end
local first: integer = sum_bias(20, 3, 4)
local second: integer = sum_bias(20, 3, 4)
return first + second
"""

_ZERO_TRIP = """-- luapyre: typed
local function zero(n: integer, a: integer, b: integer): integer
    local i: integer = 0
    local total: integer = 5
    while i < n do
        local bias: integer = a + b
        total = total + bias
        i = i + 1
    end
    return total
end
local first: integer = zero(0, 7, 9)
local second: integer = zero(0, 7, 9)
return first + second
"""

_DYNAMIC_SOURCE = """-- luapyre: typed
local function dynamic_bias(n: integer, a: integer): integer
    local i: integer = 0
    local total: integer = 0
    while i < n do
        local bias: integer = a + i
        total = total + bias
        i = i + 1
    end
    return total
end
return dynamic_bias(8, 3)
"""

_SHARED_UPDATE_SOURCE = """-- luapyre: typed
local function shared_update(n: integer): integer
    local i: integer = 0
    local total: integer = 0
    while i < n do
        i = i + 1
        total = total + i
    end
    return total
end
return shared_update(8)
"""


def _child(runtime: LuaRuntime, source: str, name: str):
    root = runtime.compile(source)
    return next(child for child in root.children if child.name == name)


def _optimized(runtime: LuaRuntime, source: str, name: str):
    child = _child(runtime, source, name)
    cfg = CFGValueIRCompiler(child).compile()
    assert cfg is not None
    optimized = CFGLoopOptimizer(child, cfg).analyze()
    assert len(optimized.loops) == 1
    return cfg, optimized.loops[0]


def _compiled_function_names(runtime: LuaRuntime) -> set[str]:
    return {
        compiled.runner.__name__
        for _proto, compiled in runtime.vm.jit._function_cache.values()
        if compiled is not None
    }


def test_loop_optimizer_proves_licm_and_integer_induction():
    runtime = LuaRuntime(jit=False)
    cfg, loop = _optimized(runtime, _SOURCE, "sum_bias")

    invariant_ops = {cfg.node(node_id).op for node_id in loop.invariant_nodes}
    assert "add_i" in invariant_ops

    assert loop.induction_variables
    induction = next(item for item in loop.induction_variables if item.step == 1)
    assert induction.direct_update
    assert induction.compare_op == "lt"
    assert induction.limit_node is not None


def test_licm_rejects_expression_depending_on_loop_carried_phi():
    runtime = LuaRuntime(jit=False)
    _cfg, loop = _optimized(runtime, _DYNAMIC_SOURCE, "dynamic_bias")
    assert loop.invariant_nodes == ()


def test_induction_direct_update_requires_no_other_consumer():
    runtime = LuaRuntime(jit=False)
    _cfg, loop = _optimized(runtime, _SHARED_UPDATE_SOURCE, "shared_update")
    induction = next(item for item in loop.induction_variables if item.step == 1)
    assert not induction.direct_update


def test_optimized_cfg_loop_executes_licm_and_direct_induction_backend():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    assert runtime.execute(_SOURCE) == 660
    assert "_jit_cfg_value_ir_loop_opt" in _compiled_function_names(runtime)


def test_licm_does_not_change_zero_trip_result():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    assert runtime.execute(_ZERO_TRIP) == 10
    assert "_jit_cfg_value_ir_loop_opt" in _compiled_function_names(runtime)


def _outcome(source: str, *, jit: bool, fuel: int):
    runtime = LuaRuntime(jit=jit, jit_threshold=1, fuel=fuel)
    try:
        return ("ok", runtime.execute(source))
    except LuaQuotaError as exc:
        return ("quota", str(exc))


def test_licm_and_induction_preserve_nearby_fuel_boundaries():
    for fuel in range(1, 420):
        assert _outcome(_SOURCE, jit=True, fuel=fuel) == _outcome(
            _SOURCE, jit=False, fuel=fuel
        )
