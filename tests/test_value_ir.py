from __future__ import annotations

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
