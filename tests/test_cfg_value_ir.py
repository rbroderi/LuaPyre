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

_DOMINANCE_SOURCE = """-- luapyre: typed
local function reuse(flag: boolean, a: integer, b: integer): integer
    local base = a + b
    if flag then
        local same = a + b
        return same
    end
    return base
end
local first: integer = reuse(true, 4, 7)
local second: integer = reuse(false, 4, 7)
return first + second
"""

_SIBLING_SOURCE = """-- luapyre: typed
local function sibling(flag: boolean, a: integer, b: integer): integer
    local value: integer = 0
    if flag then
        value = a + b
    else
        value = a + b
    end
    local again = a + b
    return value + again
end
local first: integer = sibling(true, 4, 7)
local second: integer = sibling(false, 4, 7)
return first + second
"""

_LOOP_SOURCE = """-- luapyre: typed
local function sum_to(n: integer): integer
    local i: integer = 0
    local total: integer = 0
    while i < n do
        total = total + i
        i = i + 1
    end
    return total
end
local first: integer = sum_to(20)
local second: integer = sum_to(20)
return first + second
"""

_BRANCHY_LOOP_SOURCE = """-- luapyre: typed
local function branchy(n: integer): integer
    local i: integer = 0
    local total: integer = 0
    while i < n do
        if i < 5 then
            total = total + i
        else
            total = total + 1
        end
        i = i + 1
    end
    return total
end
local first: integer = branchy(10)
local second: integer = branchy(10)
return first + second
"""

_NESTED_LOOP_SOURCE = """-- luapyre: typed
local function nested(n: integer, m: integer): integer
    local i: integer = 0
    local total: integer = 0
    while i < n do
        local j: integer = 0
        while j < m do
            total = total + i + j
            j = j + 1
        end
        i = i + 1
    end
    return total
end
local first: integer = nested(3, 4)
local second: integer = nested(3, 4)
return first + second
"""


def _compiled_function_filenames(runtime: LuaRuntime) -> set[str]:
    return {
        compiled.runner.__code__.co_filename
        for _proto, compiled in runtime.vm.jit._function_cache.values()
        if compiled is not None
    }


def _child(runtime: LuaRuntime, source: str, name: str):
    root = runtime.compile(source)
    return next(child for child in root.children if child.name == name)


def test_cfg_value_ir_builds_phi_for_branch_merge():
    runtime = LuaRuntime(jit=False)
    choose = _child(runtime, _SOURCE, "choose")
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


def test_cfg_value_ir_reuses_expression_only_through_dominance():
    runtime = LuaRuntime(jit=False)
    reuse = _child(runtime, _DOMINANCE_SOURCE, "reuse")
    plan = CFGValueIRCompiler(reuse).compile()
    assert plan is not None
    assert plan.cross_block_cse_pcs
    assert not plan.backedges
    assert all(block.index in plan.dominators[block.index] for block in plan.blocks)
    assert any(len(dominators) > 1 for dominators in plan.dominators[1:])


def test_cfg_value_ir_does_not_reuse_sibling_expression_without_dominance():
    runtime = LuaRuntime(jit=False)
    sibling = _child(runtime, _SIBLING_SOURCE, "sibling")
    plan = CFGValueIRCompiler(sibling).compile()
    assert plan is not None
    assert plan.phi_nodes
    assert not plan.cross_block_cse_pcs


def test_cfg_value_ir_builds_loop_carried_phi_values_from_natural_backedge():
    runtime = LuaRuntime(jit=False)
    sum_to = _child(runtime, _LOOP_SOURCE, "sum_to")
    plan = CFGValueIRCompiler(sum_to).compile()
    assert plan is not None
    assert plan.has_cycles
    assert len(plan.natural_loops) == 1
    loop = plan.natural_loops[0]
    assert (loop.latch, loop.header) in plan.backedges
    assert loop.header in plan.dominators[loop.latch]
    header = plan.blocks[loop.header]
    assert header.phi_nodes
    assert any(plan.node(node_id).type_name == "integer" for node_id in header.phi_nodes)
    assert all(len(plan.node(node_id).args) == 2 for node_id in header.phi_nodes)


def test_cfg_value_ir_structures_simple_natural_loop_without_state_dispatch():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    assert runtime.execute(_LOOP_SOURCE) == 380
    assert "<luapyre-cfg-value-ir-loop>" in _compiled_function_filenames(runtime)


def test_cfg_value_ir_branchy_reducible_loop_uses_generic_cyclic_backend():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    assert runtime.execute(_BRANCHY_LOOP_SOURCE) == 30
    assert "<luapyre-cfg-value-ir-function>" in _compiled_function_filenames(runtime)


def test_cfg_value_ir_nested_natural_loops_share_the_cyclic_value_graph():
    runtime = LuaRuntime(jit=False)
    nested = _child(runtime, _NESTED_LOOP_SOURCE, "nested")
    plan = CFGValueIRCompiler(nested).compile()
    assert plan is not None
    assert len(plan.natural_loops) == 2
    assert len(plan.backedges) == 2
    for loop in plan.natural_loops:
        assert loop.header in plan.dominators[loop.latch]
        assert plan.blocks[loop.header].phi_nodes

    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    assert runtime.execute(_NESTED_LOOP_SOURCE) == 60
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


def test_cfg_value_ir_loop_side_exit_preserves_every_nearby_fuel_boundary():
    for fuel in range(1, 300):
        assert _outcome(_LOOP_SOURCE, jit=True, fuel=fuel) == _outcome(
            _LOOP_SOURCE, jit=False, fuel=fuel
        )
