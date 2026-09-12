from __future__ import annotations

from .bytecode import Op


# Keep structural eligibility independent of the Python backend. This is the
# Tier-1 quickening policy; a later native backend can consume the same marker.
# 0.14 admits acyclic internal control flow and explicit type guards so the
# region backend can compile whole branchy loop bodies rather than bouncing
# through Tier 0 at every conditional.
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
    Op.GUARD,
    Op.JMP,
    Op.JMPIF,
    Op.JMPIFNOT,
})


# Fully typed source is LuaPyre's performance target. Once the compiler has
# certified a Proto we can admit a substantially larger region and let the AST
# backend fail closed if a particular runtime call/table shape is not safe.
# GETUPVAL is admitted so the typed IR can recognize the common child-function
# pattern ``GETUPVAL _ENV; LOADK field; GETTABLE`` and virtualize the temporary
# environment/key registers before Python AST emission.
TYPED_JIT_LOOP_BODY_OPS = JIT_LOOP_BODY_OPS | frozenset({
    Op.GETUPVAL,
    Op.DIV,
    Op.IDIV,
    Op.POW,
    Op.CONCAT,
    Op.LEN,
    Op.NEG,
    Op.BAND,
    Op.BOR,
    Op.BXOR,
    Op.SHL,
    Op.SHR,
    Op.BNOT,
    Op.JMPIFNIL,
    Op.FORPREP,
    Op.FORLOOP,
    Op.JFORLOOP,
    Op.CALL,
})


def can_jit_natural_loop(code, start_pc: int, end_pc: int) -> bool:
    return start_pc < end_pc and all(
        ins.op in JIT_LOOP_BODY_OPS for ins in code[start_pc:end_pc]
    )


def can_jit_typed_loop(code, start_pc: int, end_pc: int) -> bool:
    """Return whether a fully typed loop is eligible for typed region JIT.

    Eligibility is intentionally structural rather than a guarantee of
    compilation. Runtime-sensitive operations (notably CALL and table access)
    still carry guards/deoptimization in the backend.
    """

    return start_pc < end_pc and all(
        ins.op in TYPED_JIT_LOOP_BODY_OPS for ins in code[start_pc:end_pc]
    )
