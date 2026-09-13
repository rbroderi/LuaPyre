from __future__ import annotations

from dataclasses import dataclass

from .bytecode import Op, Proto


INT_MIN = -(1 << 63)
INT_MAX = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class IntRange:
    minimum: int
    maximum: int

    def fits_i64(self) -> bool:
        return INT_MIN <= self.minimum and self.maximum <= INT_MAX


@dataclass(frozen=True, slots=True)
class IntegerRangePlan:
    """Conservative integer intervals at arithmetic instructions."""

    inputs: tuple[tuple[int, tuple[tuple[int, IntRange], ...]], ...]
    overflow_free_pcs: frozenset[int]

    def range_at(self, pc: int, register: int) -> IntRange | None:
        values = next((items for item_pc, items in self.inputs if item_pc == pc), ())
        return next((value for reg, value in values if reg == register), None)

    def overflow_free(self, pc: int) -> bool:
        return pc in self.overflow_free_pcs


_ARITH = {Op.ADD_I, Op.SUB_I, Op.MUL_I}
_WRITE_A = {
    Op.LOADK, Op.MOVE, Op.LOCAL, Op.GETGLOBAL, Op.GETUPVAL, Op.GETCELL,
    Op.CLOSURE, Op.NEWTABLE, Op.GETTABLE, Op.LEN,
    Op.ADD, Op.ADD_I, Op.ADD_F, Op.SUB, Op.SUB_I, Op.SUB_F,
    Op.MUL, Op.MUL_I, Op.MUL_F, Op.DIV, Op.IDIV, Op.MOD, Op.POW,
    Op.BAND, Op.BOR, Op.BXOR, Op.SHL, Op.SHR, Op.BNOT, Op.CONCAT,
    Op.NEG, Op.NOT, Op.TOBOOL, Op.EQ, Op.LT, Op.LE,
    Op.VARARG, Op.PVARARG, Op.PGETVARG, Op.GUARD,
}


def _writes(ins) -> set[int]:
    if ins.op in (Op.FORLOOP, Op.JFORLOOP, Op.PFORLOOP):
        return {ins.a}
    if ins.op in _WRITE_A:
        return {ins.a}
    if ins.op is Op.CALL:
        if ins.e == -1:
            return {ins.a}
        return set(range(ins.a, ins.a + max(0, ins.e)))
    if ins.op is Op.CALLV:
        return {ins.a}
    if ins.op is Op.UNPACK:
        return set(range(ins.a, ins.a + max(0, ins.c)))
    return set()


def _reaching_int_constant(proto: Proto, register: int, before: int) -> int | None:
    seen: set[tuple[int, int]] = set()
    while before > 0:
        key = (register, before)
        if key in seen:
            return None
        seen.add(key)
        before -= 1
        ins = proto.code[before]
        if ins.op in {
            Op.JMP, Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL,
            Op.FORPREP, Op.FORLOOP, Op.JFORLOOP, Op.RETURN, Op.RETURNV, Op.HALT,
        }:
            return None
        if register not in _writes(ins):
            continue
        if ins.op is Op.LOADK:
            value = proto.constants[ins.b]
            return value if type(value) is int else None
        if ins.op in (Op.MOVE, Op.LOCAL):
            register = ins.b
            continue
        return None
    return None


def _loop_seeds(proto: Proto):
    seeds: dict[int, tuple[dict[int, IntRange], set[int]]] = {}
    for backedge_pc, backedge in enumerate(proto.code):
        if backedge.op not in (Op.FORLOOP, Op.JFORLOOP) or backedge.d >= backedge_pc:
            continue
        prep_pc = next(
            (
                pc for pc in range(backedge.d - 1, -1, -1)
                if proto.code[pc].op is Op.FORPREP
                and (proto.code[pc].a, proto.code[pc].b, proto.code[pc].c)
                == (backedge.a, backedge.b, backedge.c)
            ),
            None,
        )
        if prep_pc is None:
            continue
        initial = _reaching_int_constant(proto, backedge.a, prep_pc)
        limit = _reaching_int_constant(proto, backedge.b, prep_pc)
        step = _reaching_int_constant(proto, backedge.c, prep_pc)
        if initial is None or limit is None or step is None or step == 0:
            continue
        if (step > 0 and initial > limit) or (step < 0 and initial < limit):
            continue
        assigned: set[int] = set()
        for ins in proto.code[backedge.d:backedge_pc]:
            assigned.update(_writes(ins))
        bounds = IntRange(min(initial, limit), max(initial, limit))
        values = {backedge.a: bounds}
        if backedge.b not in assigned:
            values[backedge.b] = IntRange(limit, limit)
        if backedge.c not in assigned:
            values[backedge.c] = IntRange(step, step)
        seeds[backedge.d] = (values, assigned)
    return seeds


def analyze_integer_ranges(proto: Proto) -> IntegerRangePlan:
    """Prove signed-64-bit arithmetic results without speculative assumptions.

    Parameters begin at the full Lua integer interval. Literal numeric loops
    seed their induction register from compile-time bounds. Other values remain
    unknown, and loop-assigned registers are killed at the header so a first
    iteration cannot be mistaken for a loop invariant.
    """

    full = IntRange(INT_MIN, INT_MAX)
    ranges: dict[int, IntRange] = {
        reg: full
        for reg, typ in enumerate(proto.param_types[:proto.param_count])
        if typ.name in ("integer", "integer_lua")
    }
    seeds = _loop_seeds(proto)
    merge_points: set[int] = set()
    for pc, ins in enumerate(proto.code):
        if ins.op is Op.JMP:
            merge_points.update((ins.a, pc + 1))
        elif ins.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
            merge_points.update((ins.a, pc + 1))
        elif ins.op is Op.FORPREP:
            merge_points.update((ins.d, pc + 1))
        elif ins.op in (Op.FORLOOP, Op.JFORLOOP):
            merge_points.update((ins.d, pc + 1))
        elif ins.op in (Op.RETURN, Op.RETURNV, Op.HALT):
            merge_points.add(pc + 1)
    snapshots: list[tuple[int, tuple[tuple[int, IntRange], ...]]] = []
    safe: set[int] = set(proto.jit_no_overflow_pcs)

    for pc, ins in enumerate(proto.code):
        # Facts from one linear predecessor cannot be carried through a CFG
        # merge without a full fixed-point dataflow join.  Clearing them is
        # conservative; literal-loop seeds below recover the useful induction
        # ranges without making a path-sensitive assumption.
        if pc in merge_points:
            ranges.clear()
        seed = seeds.get(pc)
        if seed is not None:
            values, assigned = seed
            for reg in assigned:
                ranges.pop(reg, None)
            ranges.update(values)

        snapshots.append((pc, tuple(sorted(ranges.items()))))
        if ins.op is Op.LOADK:
            value = proto.constants[ins.b]
            if type(value) is int:
                ranges[ins.a] = IntRange(value, value)
            else:
                ranges.pop(ins.a, None)
        elif ins.op in (Op.MOVE, Op.LOCAL):
            value = ranges.get(ins.b)
            if value is None:
                ranges.pop(ins.a, None)
            else:
                ranges[ins.a] = value
        elif ins.op in _ARITH:
            left, right = ranges.get(ins.b), ranges.get(ins.c)
            if left is None or right is None:
                ranges.pop(ins.a, None)
                continue
            if ins.op is Op.ADD_I:
                result = IntRange(left.minimum + right.minimum, left.maximum + right.maximum)
            elif ins.op is Op.SUB_I:
                result = IntRange(left.minimum - right.maximum, left.maximum - right.minimum)
            else:
                products = (
                    left.minimum * right.minimum, left.minimum * right.maximum,
                    left.maximum * right.minimum, left.maximum * right.maximum,
                )
                result = IntRange(min(products), max(products))
            if result.fits_i64():
                safe.add(pc)
                ranges[ins.a] = result
            else:
                ranges[ins.a] = full
        else:
            for reg in _writes(ins):
                ranges.pop(reg, None)

    return IntegerRangePlan(tuple(snapshots), frozenset(safe))
