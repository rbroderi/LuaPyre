from __future__ import annotations

from luapyre.bytecode import Ins, Op, Proto, UpvalueDesc
from luapyre.typed_ir import TypedIRCompiler
from luapyre.typed_ir_passes import eliminate_rematerializable_dead_defs


def _optimized(proto: Proto, blocks):
    plan = TypedIRCompiler(proto).compile(blocks)
    return eliminate_rematerializable_dead_defs(plan, blocks)


def test_liveness_kills_earlier_virtual_key_despite_later_register_reuse():
    proto = Proto(
        "reuse",
        code=[
            Ins(Op.LOADK, 1, 0),
            Ins(Op.GETTABLE, 2, 0, 1),
            Ins(Op.LOADK, 1, 1),
            Ins(Op.RETURN, 1, 1),
        ],
        constants=[b"answer", 42],
        register_count=3,
        env_reg=0,
        jit_fully_typed=True,
    )
    blocks = (((0, proto.code[0]), (1, proto.code[1]), (2, proto.code[2]), (3, proto.code[3])),)

    plan = _optimized(proto, blocks)

    assert plan.instruction(0).dead_definition
    assert not plan.instruction(2).dead_definition


def test_liveness_flows_across_blocks_for_virtual_constant_key():
    proto = Proto(
        "blocks",
        code=[
            Ins(Op.LOADK, 1, 0),
            Ins(Op.JMP, 2),
            Ins(Op.GETTABLE, 2, 0, 1),
            Ins(Op.RETURN, 2, 1),
        ],
        constants=[b"answer"],
        register_count=3,
        env_reg=0,
        jit_fully_typed=True,
    )
    blocks = (
        ((0, proto.code[0]), (1, proto.code[1])),
        ((2, proto.code[2]), (3, proto.code[3])),
    )

    plan = _optimized(proto, blocks)

    assert plan.instruction(0).dead_definition


def test_liveness_keeps_materialized_constant_across_blocks():
    proto = Proto(
        "blocks",
        code=[
            Ins(Op.LOADK, 1, 0),
            Ins(Op.JMP, 2),
            Ins(Op.ADD_I, 2, 1, 0),
            Ins(Op.RETURN, 2, 1),
        ],
        constants=[7],
        register_count=3,
        jit_fully_typed=True,
    )
    blocks = (
        ((0, proto.code[0]), (1, proto.code[1])),
        ((2, proto.code[2]), (3, proto.code[3])),
    )

    plan = _optimized(proto, blocks)

    assert not plan.instruction(0).dead_definition


def test_liveness_can_remove_direct_env_upvalue_for_specialized_access():
    proto = Proto(
        "upvalue",
        code=[
            Ins(Op.GETUPVAL, 0, 0),
            Ins(Op.LOADK, 1, 0),
            Ins(Op.GETTABLE, 2, 0, 1),
            Ins(Op.RETURN, 2, 1),
        ],
        constants=[b"answer"],
        upvalues=[UpvalueDesc("upvalue", 0, "_ENV")],
        register_count=3,
        jit_fully_typed=True,
    )
    blocks = (((0, proto.code[0]), (1, proto.code[1]), (2, proto.code[2]), (3, proto.code[3])),)

    plan = _optimized(proto, blocks)

    assert plan.instruction(0).dead_definition
    assert plan.instruction(1).dead_definition
