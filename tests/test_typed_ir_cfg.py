from __future__ import annotations

from luapyre.bytecode import Ins, Op, Proto
from luapyre.typed_ir import IRValueKind, TypedIRCompiler


def test_constant_key_propagates_across_basic_block_edge():
    proto = Proto(
        "cfg",
        code=[
            Ins(Op.LOADK, 1, 0),
            Ins(Op.JMP, 2),
            Ins(Op.GETTABLE, 2, 0, 1),
        ],
        constants=[b"answer"],
        register_count=3,
        env_reg=0,
        jit_fully_typed=True,
    )
    blocks = (
        ((0, proto.code[0]), (1, proto.code[1])),
        ((2, proto.code[2]),),
    )
    plan = TypedIRCompiler(proto).compile(blocks)

    load = plan.instruction(0)
    get = plan.instruction(2)
    assert load is not None and load.dead_definition
    assert get is not None and get.specialization == "global_get"
    assert get.value_for(1).kind is IRValueKind.CONSTANT
    assert plan.virtual_state(2)[1].kind is IRValueKind.CONSTANT


def test_identical_branch_facts_survive_cfg_join():
    proto = Proto(
        "diamond",
        code=[
            Ins(Op.JMPIF, 3, 0),
            Ins(Op.LOADK, 1, 0),
            Ins(Op.JMP, 4),
            Ins(Op.LOADK, 1, 0),
            Ins(Op.GETTABLE, 2, 5, 1),
        ],
        constants=[b"value"],
        register_count=6,
        jit_fully_typed=True,
    )
    blocks = (
        ((0, proto.code[0]),),
        ((1, proto.code[1]), (2, proto.code[2])),
        ((3, proto.code[3]),),
        ((4, proto.code[4]),),
    )
    plan = TypedIRCompiler(proto).compile(blocks)

    get = plan.instruction(4)
    assert get is not None and get.specialization == "table_get_const"
    assert get.value_for(1).kind is IRValueKind.CONSTANT


def test_conflicting_branch_facts_do_not_cross_cfg_join():
    proto = Proto(
        "diamond",
        code=[
            Ins(Op.JMPIF, 3, 0),
            Ins(Op.LOADK, 1, 0),
            Ins(Op.JMP, 4),
            Ins(Op.LOADK, 1, 1),
            Ins(Op.GETTABLE, 2, 5, 1),
        ],
        constants=[b"left", b"right"],
        register_count=6,
        jit_fully_typed=True,
    )
    blocks = (
        ((0, proto.code[0]),),
        ((1, proto.code[1]), (2, proto.code[2])),
        ((3, proto.code[3]),),
        ((4, proto.code[4]),),
    )
    plan = TypedIRCompiler(proto).compile(blocks)

    get = plan.instruction(4)
    assert get is not None and get.specialization is None
    assert get.value_for(1).kind is IRValueKind.REGISTER
    assert 1 not in plan.virtual_state(4)


def test_backedge_cannot_invent_first_entry_fact():
    proto = Proto(
        "loop",
        code=[
            Ins(Op.JMPIF, 2, 0),
            Ins(Op.LOADK, 1, 0),
            Ins(Op.JMP, 0),
        ],
        constants=[b"later"],
        register_count=2,
        jit_fully_typed=True,
    )
    blocks = (
        ((0, proto.code[0]),),
        ((1, proto.code[1]),),
        ((2, proto.code[2]),),
    )
    plan = TypedIRCompiler(proto).compile(blocks)

    # The backedge can carry r1, but the external first entry cannot. The meet
    # at block 0 must therefore keep r1 unknown.
    assert 1 not in plan.virtual_state(0)
