from __future__ import annotations

from .bytecode import Op


# Keep structural eligibility independent of the Python backend.  This is the
# Tier-1 quickening policy; a later native backend can consume the same marker.
JIT_LOOP_BODY_OPS = frozenset({
    Op.LOADK,
    Op.MOVE,
    Op.LOCAL,
    Op.NEWTABLE,
    Op.GETTABLE,
    Op.SETTABLE,
    Op.ADD,
    Op.ADD_I,
    Op.ADD_F,
    Op.SUB,
    Op.SUB_I,
    Op.SUB_F,
    Op.MUL,
    Op.MUL_I,
    Op.MUL_F,
    Op.MOD,
    Op.EQ,
    Op.LT,
    Op.LE,
    Op.NOT,
    Op.TOBOOL,
})


def can_jit_natural_loop(code, start_pc: int, end_pc: int) -> bool:
    return start_pc < end_pc and all(
        ins.op in JIT_LOOP_BODY_OPS for ins in code[start_pc:end_pc]
    )
