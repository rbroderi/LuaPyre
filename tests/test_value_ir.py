from __future__ import annotations

from copy import copy

from luapyre import LuaRuntime
from luapyre.bytecode import Ins, Op, Proto
from luapyre.typed_ir import TypedIRCompiler
from luapyre.typesys import INTEGER
from luapyre.value_ir import ValueIRCompiler, ValueKind


def _value_plan(proto: Proto):
    block = tuple(enumerate(proto.code))
    typed = TypedIRCompiler(proto).compile((block,))
    plan = ValueIRCompiler(proto, typed).compile()
    assert plan is not None
    return plan


def _reachable_value_plan(proto: Proto):
    terminal = next(
        pc for pc, ins in enumerate(proto.code) if ins.op in (Op.RETURN, Op.HALT)
    )
    analysis = copy(proto)
    analysis.code = list(proto.code[: terminal + 1])
    typed = TypedIRCompiler(analysis).compile((tuple(enumerate(analysis.code)),))
    return ValueIRCompiler(analysis, typed).compile()


def test_value_ir_constant_folds_exact_signed_64_wrap():
    proto = Proto(
        "fold",
        code=[
            Ins(Op.LOADK, 0, 0),
            Ins(Op.LOADK, 1, 1),
            Ins(Op.ADD_I, 2, 0, 1),
            Ins(Op.RETURN, 2, 1),
        ],
        constants=[(1 << 63) - 1, 1],
        register_count=3,
        jit_trust_types=True,
        jit_fully_typed=True,
    )
    plan = _value_plan(proto)
    result = plan.node(plan.return_values[0])
    assert result.kind is ValueKind.LITERAL
    assert result.payload == -(1 << 63)
    assert plan.folded_pcs == (2,)


def test_value_ir_cses_identical_typed_expression():
    proto = Proto(
        "cse",
        code=[
            Ins(Op.ADD_I, 2, 0, 1),
            Ins(Op.ADD_I, 3, 0, 1),
            Ins(Op.RETURN, 3, 1),
        ],
        register_count=4,
        param_count=2,
        param_types=[INTEGER, INTEGER],
        jit_trust_types=True,
        jit_fully_typed=True,
    )
    plan = _value_plan(proto)
    assert plan.cse_pcs == (1,)
    assert plan.eliminated_pcs == (1,)
    node = plan.node(plan.return_values[0])
    assert node.kind is ValueKind.EXPRESSION
    assert node.def_pc == 0


def test_value_ir_dses_unused_pure_expression_but_keeps_fuel_cost():
    proto = Proto(
        "dse",
        code=[
            Ins(Op.ADD_I, 2, 0, 1),
            Ins(Op.MUL_I, 3, 0, 1),
            Ins(Op.RETURN, 2, 1),
        ],
        register_count=4,
        param_count=2,
        param_types=[INTEGER, INTEGER],
        jit_trust_types=True,
        jit_fully_typed=True,
    )
    plan = _value_plan(proto)
    assert 1 in plan.eliminated_pcs
    assert plan.instruction_count == len(proto.code)


def _compiled_function_filenames(runtime: LuaRuntime) -> set[str]:
    return {
        compiled.runner.__code__.co_filename
        for _proto, compiled in runtime.vm.jit._function_cache.values()
        if compiled is not None
    }


def test_pure_typed_leaf_executes_through_value_ir_backend():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    source = """-- luapyre: typed
local function calc(a: integer, b: integer): integer
    local first = a + b
    local same = a + b
    local dead = a * b
    return same
end
local result: integer = calc(100, 23)
return result
"""
    assert runtime.execute(source) == 123
    assert "<luapyre-value-ir-function>" in _compiled_function_filenames(runtime)


def test_constant_only_leaf_folds_without_changing_result():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    source = """-- luapyre: typed
local function folded(): integer
    local a = 9223372036854775807
    local b = 1
    local value = a + b
    return value
end
local result: integer = folded()
return result
"""
    assert runtime.execute(source) == -(1 << 63)
    assert "<luapyre-value-ir-function>" in _compiled_function_filenames(runtime)


def test_static_leaf_call_is_represented_and_inlined_in_value_ir():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    source = """-- luapyre: typed
local function outer(a: integer, b: integer): integer
    local function add(x: integer, y: integer): integer
        local first = x + y
        local same = x + y
        return same
    end
    local result = add(a, b)
    return result
end
local answer: integer = outer(100, 23)
return answer
"""
    root = runtime.compile(source)
    outer = next(child for child in root.children if child.name == "outer")
    plan = _reachable_value_plan(outer)
    assert plan is not None
    assert len(plan.call_sites) == 1
    call = plan.call_sites[0]
    assert outer.children[call.child_index].name == "add"
    caller_terminal = next(
        pc for pc, ins in enumerate(outer.code) if ins.op in (Op.RETURN, Op.HALT)
    )
    add = outer.children[call.child_index]
    add_terminal = next(
        pc for pc, ins in enumerate(add.code) if ins.op in (Op.RETURN, Op.HALT)
    )
    assert plan.instruction_count == (caller_terminal + 1) + (add_terminal + 1)

    assert runtime.vm.run(root) == 123
    assert "<luapyre-value-ir-function>" in _compiled_function_filenames(runtime)


def test_captured_child_call_fails_closed_to_existing_function_compiler():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    source = """-- luapyre: typed
local function outer(x: integer): integer
    local bias = 3
    local function add(y: integer): integer
        return y + bias
    end
    local result = add(x)
    return result
end
local answer: integer = outer(4)
return answer
"""
    root = runtime.compile(source)
    outer = next(child for child in root.children if child.name == "outer")
    assert outer.children and outer.children[0].upvalues
    assert _reachable_value_plan(outer) is None
    assert runtime.vm.run(root) == 7
