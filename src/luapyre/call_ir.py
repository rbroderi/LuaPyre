from __future__ import annotations

from dataclasses import dataclass

from .bytecode import Op, Proto


@dataclass(frozen=True, slots=True)
class StaticClosureRef:
    """Compile-time identity of a lexical child closure."""

    child_index: int


@dataclass(frozen=True, slots=True)
class StaticCallSite:
    """Backend-neutral description of one statically resolved inline CALL.

    The call remains charged as a Lua CALL plus the complete reachable callee
    bytecode. A backend may inline ``child_index`` only when its optimizer has
    separately proved that the child is a pure fully typed leaf and that the
    closure value cannot escape.
    """

    pc: int
    child_index: int
    arg_count: int
    result_count: int
    callee_instruction_count: int

    @property
    def total_inline_cost(self) -> int:
        """Extra cost beyond the caller's already-counted CALL instruction."""

        return self.callee_instruction_count


@dataclass(frozen=True, slots=True)
class DirectCallSite:
    """A statically resolved CALL that retains a real Lua child frame.

    Unlike ``StaticCallSite`` this record is not permission to erase the call.
    It proves only callee identity. The backend must still materialize the real
    closure, create a normal Lua frame, share the exact fuel meter, obey
    ``max_frames``, and preserve suspension/error behavior.
    """

    pc: int
    child_index: int
    function_register: int
    arg_base: int
    arg_count: int
    result_base: int
    result_count: int


@dataclass(frozen=True, slots=True)
class CallIRPlan:
    """Backend-neutral static CALL facts for one straight-line function."""

    direct_sites: tuple[DirectCallSite, ...]

    def direct_site(self, pc: int) -> DirectCallSite | None:
        for site in self.direct_sites:
            if site.pc == pc:
                return site
        return None


# Operations whose destination register can destroy a tracked lexical closure
# identity. CALL writes are handled separately because their result range is
# encoded by ``e`` rather than a single conventional destination.
_SINGLE_DEST_OPS = frozenset(
    {
        Op.LOADK,
        Op.MOVE,
        Op.LOCAL,
        Op.GETGLOBAL,
        Op.GETUPVAL,
        Op.GETCELL,
        Op.CLOSURE,
        Op.NEWTABLE,
        Op.GETTABLE,
        Op.LEN,
        Op.ADD,
        Op.ADD_I,
        Op.ADD_F,
        Op.SUB,
        Op.SUB_I,
        Op.SUB_F,
        Op.MUL,
        Op.MUL_I,
        Op.MUL_F,
        Op.DIV,
        Op.IDIV,
        Op.MOD,
        Op.POW,
        Op.BAND,
        Op.BOR,
        Op.BXOR,
        Op.SHL,
        Op.SHR,
        Op.BNOT,
        Op.CONCAT,
        Op.NEG,
        Op.NOT,
        Op.TOBOOL,
        Op.EQ,
        Op.LT,
        Op.LE,
    }
)


def analyze_direct_calls(proto: Proto) -> CallIRPlan:
    """Resolve exact lexical child CALLs without deciding how to lower them.

    This is deliberately a small data-flow analysis. Closure identities may flow
    through MOVE/LOCAL. Any unsupported use simply drops the identity; it never
    guesses. A direct site is recorded only for a CALL to a fully
    typed, non-vararg child with no captured upvalues or nested children. Those
    restrictions let the first Python backend retain exact closure allocation
    while delegating the actual child execution to the proven real-frame call
    machinery.
    """

    closures: dict[int, StaticClosureRef] = {}
    sites: list[DirectCallSite] = []

    for pc, ins in enumerate(proto.code):
        op = ins.op
        if op is Op.CLOSURE:
            if 0 <= ins.b < len(proto.children):
                closures[ins.a] = StaticClosureRef(ins.b)
            else:
                closures.pop(ins.a, None)
            continue

        if op in (Op.MOVE, Op.LOCAL):
            source = closures.get(ins.b)
            if source is None:
                closures.pop(ins.a, None)
            else:
                closures[ins.a] = source
            continue

        if op is Op.CALL:
            ref = closures.get(ins.b)
            if ref is not None:
                child = proto.children[ref.child_index]
                if (
                    child.jit_fully_typed
                    and not child.is_vararg
                    and not child.upvalues
                    and not child.children
                    and ins.d == child.param_count
                ):
                    sites.append(
                        DirectCallSite(
                            pc=pc,
                            child_index=ref.child_index,
                            function_register=ins.b,
                            arg_base=ins.c,
                            arg_count=ins.d,
                            result_base=ins.a,
                            result_count=ins.e,
                        )
                    )
            if ins.e > 0:
                for reg in range(ins.a, ins.a + ins.e):
                    closures.pop(reg, None)
            continue

        if op in _SINGLE_DEST_OPS:
            closures.pop(ins.a, None)

    return CallIRPlan(tuple(sites))
