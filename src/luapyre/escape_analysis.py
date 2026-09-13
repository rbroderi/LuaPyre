from __future__ import annotations

from dataclasses import dataclass

from .bytecode import Op, Proto
from .call_ir import CallIRPlan


VIRTUAL_FRAME_OPS = frozenset(
    {
        Op.LOADK, Op.MOVE, Op.LOCAL,
        Op.ADD_I, Op.SUB_I, Op.MUL_I,
        Op.ADD_F, Op.SUB_F, Op.MUL_F,
        Op.DIV,
        Op.NOT, Op.TOBOOL, Op.LT, Op.LE, Op.GUARD,
        Op.JMP, Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL,
        Op.FORPREP, Op.FORLOOP, Op.JFORLOOP,
        Op.RETURN, Op.HALT,
    }
)


@dataclass(frozen=True, slots=True)
class EscapeSite:
    """Escape and materialization facts for a statically resolved call."""

    pc: int
    virtual_frame: bool
    virtual_multivalue: bool
    multivalue_consumers: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class EscapePlan:
    """Backend-neutral allocation facts consumed by JIT backends."""

    sites: tuple[EscapeSite, ...]

    def site(self, pc: int) -> EscapeSite | None:
        return next((site for site in self.sites if site.pc == pc), None)


def _open_result_consumers(proto: Proto, call_pc: int, result_reg: int) -> tuple[int, ...] | None:
    consumers: list[int] = []
    for pc in range(call_pc + 1, len(proto.code)):
        ins = proto.code[pc]
        if (
            ins.op is Op.UNPACK
            and ins.b == result_reg
            and not (ins.a <= result_reg < ins.a + ins.c)
        ):
            consumers.append(pc)
            continue
        if ins.op is Op.RETURNV and ins.c == result_reg:
            consumers.append(pc)
            return tuple(consumers)
        if ins.op in {
            Op.LOADK, Op.MOVE, Op.LOCAL, Op.CLOSURE,
            Op.ADD_I, Op.SUB_I, Op.MUL_I,
            Op.ADD_F, Op.SUB_F, Op.MUL_F, Op.NOT, Op.TOBOOL,
        } and ins.a == result_reg:
            return tuple(consumers)
        if result_reg in (ins.a, ins.b, ins.c, ins.d, ins.e):
            return None
    return tuple(consumers)


def analyze_escapes(proto: Proto, calls: CallIRPlan) -> EscapePlan:
    """Prove where Frame and MultiValue identities are unobservable.

    Virtual frames are fully typed lexical leaves without heap effects. Their
    scalar state may live in backend locals and is materialized only at a
    suspension. Open results are scalar-replaceable only when every use is a
    known UNPACK/RETURNV consumer.
    """

    sites: list[EscapeSite] = []
    for call in calls.direct_sites:
        child = proto.children[call.child_index]
        virtual_frame = (
            child.jit_fully_typed
            and not child.is_vararg
            and not child.upvalues
            and not child.children
            and bool(child.code)
            and all(ins.op in VIRTUAL_FRAME_OPS for ins in child.code)
        )
        consumers: tuple[int, ...] = ()
        virtual_multivalue = False
        if virtual_frame and call.result_count == -1:
            found = _open_result_consumers(proto, call.pc, call.result_base)
            if found is not None:
                consumers = found
                virtual_multivalue = True
        sites.append(EscapeSite(call.pc, virtual_frame, virtual_multivalue, consumers))
    return EscapePlan(tuple(sites))
