from __future__ import annotations

from luapyre import LuaQuotaError, LuaRuntime
from luapyre.cfg_value_ir import CFGValueIRCompiler


_SOURCE = """-- luapyre: typed
local function choose(flag: boolean, a: integer, b: integer): integer
    local value = a
    if flag then
        value = b + 1
    else
        value = a + 1
    end
    local doubled = value + value
    return doubled
end
local first: integer = choose(true, 10, 20)
local second: integer = choose(false, 10, 20)
return first + second
"""

_DAG_SOURCE = """-- luapyre: typed
local function classify(first: boolean, second: boolean, x: integer): integer
    local value = x
    if first then
        value = value + 1
    else
        value = value + 2
    end
    if second then
        value = value * 2
    else
        value = value * 3
    end
    return value
end
local a: integer = classify(true, false, 5)
local b: integer = classify(false, true, 5)
return a + b
"""


def _compiled_function_filenames(runtime: LuaRuntime) -> set[str]:
    return {
        compiled.runner.__code__.co_filename
        for _proto, compiled in runtime.vm.jit._function_cache.values()
        if compiled is not None
    }


def test_cfg_value_ir_builds_phi_for_branch_merge():
    runtime = LuaRuntime(jit=False)
    root = runtime.compile(_SOURCE)
    choose = next(child for child in root.children if child.name == "choose")
    plan = CFGValueIRCompiler(choose).compile()
    assert plan is not None
    assert len(plan.blocks) >= 4
    assert plan.phi_nodes
    assert any(node.type_name == "integer" for node in plan.phi_nodes)


def test_cfg_value_ir_executes_both_merge_predecessors_through_structured_diamond():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    assert runtime.execute(_SOURCE) == 64
    assert "<luapyre-cfg-value-ir-diamond>" in _compiled_function_filenames(runtime)


def test_cfg_value_ir_generic_forward_dag_handles_multiple_joins():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    assert runtime.execute(_DAG_SOURCE) == 32
    assert "<luapyre-cfg-value-ir-function>" in _compiled_function_filenames(runtime)


def _outcome(source: str, *, jit: bool, fuel: int):
    runtime = LuaRuntime(jit=jit, jit_threshold=1, fuel=fuel)
    try:
        return ("ok", runtime.execute(source))
    except LuaQuotaError as exc:
        return ("quota", str(exc))


def test_cfg_value_ir_block_side_exit_preserves_every_nearby_fuel_boundary():
    for fuel in range(1, 96):
        assert _outcome(_SOURCE, jit=True, fuel=fuel) == _outcome(
            _SOURCE, jit=False, fuel=fuel
        )


def test_cfg_value_ir_generic_dag_preserves_every_nearby_fuel_boundary():
    for fuel in range(1, 128):
        assert _outcome(_DAG_SOURCE, jit=True, fuel=fuel) == _outcome(
            _DAG_SOURCE, jit=False, fuel=fuel
        )
