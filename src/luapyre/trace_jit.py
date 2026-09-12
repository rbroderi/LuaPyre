from __future__ import annotations

from dataclasses import dataclass
from types import FunctionType

from .bytecode import Op, Proto
from .cfg_value_ir import CFGValueIRCompiler
from .values import i64


TRACE_CONTINUE = 0
TRACE_SIDE_EXIT = 1
TRACE_RETURN = 2

_TRACE_OPS = frozenset(
    {
        Op.LOADK,
        Op.MOVE,
        Op.LOCAL,
        Op.ADD_I,
        Op.ADD_F,
        Op.SUB_I,
        Op.SUB_F,
        Op.MUL_I,
        Op.MUL_F,
        Op.NOT,
        Op.TOBOOL,
        Op.EQ,
        Op.LT,
        Op.LE,
        Op.JMP,
        Op.JMPIF,
        Op.JMPIFNOT,
        Op.JMPIFNIL,
        Op.RETURN,
        Op.HALT,
    }
)


@dataclass(frozen=True, slots=True)
class TracePlan:
    proto: Proto
    start_pc: int
    pcs: tuple[int, ...]
    expected_edges: tuple[tuple[int, int], ...]
    loop: bool
    exit_pc: int


@dataclass(frozen=True, slots=True)
class CompiledTrace:
    plan: TracePlan
    runner: FunctionType


class TraceJIT:
    """Profile-guided, trace-shaped CFG compiler with exact register OSR."""

    def __init__(self, *, threshold: int = 32, max_length: int = 64):
        self.threshold = threshold
        self.max_length = max_length
        self.edge_counts: dict[tuple[int, int, int], tuple[Proto, int]] = {}
        self.entry_counts: dict[tuple[int, int], tuple[Proto, int]] = {}
        self.side_exits: dict[tuple[int, int], tuple[Proto, int]] = {}
        self.cache: dict[tuple[int, int], tuple[Proto, CompiledTrace | None]] = {}

    @staticmethod
    def _key(proto: Proto, pc: int) -> tuple[int, int]:
        return id(proto), pc

    def record_edge(self, proto: Proto, source: int, target: int) -> int:
        key = (id(proto), source, target)
        old = self.edge_counts.get(key)
        count = old[1] + 1 if old is not None and old[0] is proto else 1
        self.edge_counts[key] = (proto, count)
        entry_key = self._key(proto, target)
        old_entry = self.entry_counts.get(entry_key)
        entries = (
            old_entry[1] + 1
            if old_entry is not None and old_entry[0] is proto
            else 1
        )
        self.entry_counts[entry_key] = (proto, entries)
        return entries

    def record_side_exit(self, proto: Proto, pc: int) -> int:
        key = self._key(proto, pc)
        old = self.side_exits.get(key)
        count = old[1] + 1 if old is not None and old[0] is proto else 1
        self.side_exits[key] = (proto, count)
        entry = self.entry_counts.get(key)
        entry_count = entry[1] + 1 if entry is not None and entry[0] is proto else 1
        self.entry_counts[key] = (proto, entry_count)
        return count

    def _hottest_edge(self, proto: Proto, pc: int, candidates: tuple[int, ...]) -> int | None:
        ranked = [
            (self.edge_counts.get((id(proto), pc, target), (proto, 0))[1], target)
            for target in candidates
        ]
        count, target = max(ranked, default=(0, -1))
        return target if count else None

    def _build_plan(self, proto: Proto, start_pc: int) -> TracePlan | None:
        if (
            not proto.jit_fully_typed
            or proto.children
            or start_pc < 0
            or start_pc >= len(proto.code)
        ):
            return None
        # Reuse the certified scalar CFG admission boundary. This excludes
        # metatable-sensitive comparisons, cells, calls, and any operation for
        # which register-direct trace execution is not already proven exact.
        if CFGValueIRCompiler(proto).compile() is None:
            return None
        pcs: list[int] = []
        expected: list[tuple[int, int]] = []
        seen: set[int] = set()
        pc = start_pc
        loop = False
        for _ in range(self.max_length):
            if pc == start_pc and pcs:
                loop = True
                break
            if pc in seen or pc < 0 or pc >= len(proto.code):
                break
            ins = proto.code[pc]
            if ins.op not in _TRACE_OPS:
                break
            seen.add(pc)
            pcs.append(pc)
            if ins.op is Op.JMP:
                pc = ins.a
            elif ins.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
                target = self._hottest_edge(proto, pc, (ins.a, pc + 1))
                if target is None:
                    return None
                expected.append((pc, target))
                pc = target
            elif ins.op in (Op.RETURN, Op.HALT):
                pc = len(proto.code)
                break
            else:
                pc += 1
        if len(pcs) < 2:
            return None
        return TracePlan(proto, start_pc, tuple(pcs), tuple(expected), loop, pc)

    @staticmethod
    def _preflight(lines: list[str], pc: int, indent: str) -> None:
        lines.extend(
            [
                f"{indent}if budget - used < 1:",
                f"{indent}    frame.pc = {pc}",
                f"{indent}    return used, _TRACE_CONTINUE, None",
                f"{indent}used += 1",
            ]
        )

    def _compile(self, plan: TracePlan) -> CompiledTrace:
        expected = dict(plan.expected_edges)
        lines = [
            "def _jit_cfg_trace(vm, frame, budget):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
            "    used = 0",
        ]
        indent = "    "
        if plan.loop:
            lines.append("    while True:")
            indent = "        "
        for pc in plan.pcs:
            ins = plan.proto.code[pc]
            self._preflight(lines, pc, indent)
            a, b, c = f"regs[{ins.a}]", f"regs[{ins.b}]", f"regs[{ins.c}]"
            if ins.op is Op.LOADK:
                lines.append(f"{indent}{a} = consts[{ins.b}]")
            elif ins.op in (Op.MOVE, Op.LOCAL):
                lines.append(f"{indent}{a} = {b}")
            elif ins.op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                symbol = {Op.ADD_I: "+", Op.SUB_I: "-", Op.MUL_I: "*"}[ins.op]
                lines.append(f"{indent}{a} = _i64({b} {symbol} {c})")
            elif ins.op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                symbol = {Op.ADD_F: "+", Op.SUB_F: "-", Op.MUL_F: "*"}[ins.op]
                lines.append(f"{indent}{a} = float({b} {symbol} {c})")
            elif ins.op is Op.NOT:
                lines.append(f"{indent}{a} = ({b} is None or {b} is False)")
            elif ins.op is Op.TOBOOL:
                lines.append(f"{indent}{a} = not ({b} is None or {b} is False)")
            elif ins.op is Op.EQ:
                lines.append(f"{indent}{a} = {b} == {c}")
            elif ins.op in (Op.LT, Op.LE):
                symbol = "<" if ins.op is Op.LT else "<="
                lines.append(f"{indent}{a} = {b} {symbol} {c}")
            elif ins.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
                if ins.op is Op.JMPIF:
                    test = f"not ({b} is None or {b} is False)"
                elif ins.op is Op.JMPIFNOT:
                    test = f"({b} is None or {b} is False)"
                else:
                    test = f"{b} is None"
                lines.append(f"{indent}_target = {ins.a} if {test} else {pc + 1}")
                lines.append(f"{indent}if _target != {expected[pc]}:")
                lines.extend(
                    [
                        f"{indent}    frame.pc = _target",
                        f"{indent}    return used, _TRACE_SIDE_EXIT, None",
                    ]
                )
            elif ins.op is Op.RETURN:
                values = ", ".join(f"regs[{ins.a + i}]" for i in range(ins.b))
                if ins.b == 1:
                    values += ","
                lines.append(f"{indent}return used, _TRACE_RETURN, ({values})")
            elif ins.op is Op.HALT:
                lines.append(f"{indent}return used, _TRACE_RETURN, ()")
        if plan.loop:
            lines.append(f"{indent}continue")
        else:
            lines.extend(
                [
                    f"{indent}frame.pc = {plan.exit_pc}",
                    f"{indent}return used, _TRACE_CONTINUE, None",
                ]
            )
        namespace = {
            "_i64": i64,
            "_TRACE_CONTINUE": TRACE_CONTINUE,
            "_TRACE_SIDE_EXIT": TRACE_SIDE_EXIT,
            "_TRACE_RETURN": TRACE_RETURN,
        }
        exec(compile("\n".join(lines), "<luapyre-cfg-trace>", "exec"), namespace)
        return CompiledTrace(plan, namespace["_jit_cfg_trace"])

    def maybe_trace(self, proto: Proto, start_pc: int) -> CompiledTrace | None:
        key = self._key(proto, start_pc)
        cached = self.cache.get(key)
        if cached is not None and cached[0] is proto:
            return cached[1]
        hot = self.entry_counts.get(key)
        if hot is None or hot[0] is not proto or hot[1] < self.threshold:
            return None
        plan = self._build_plan(proto, start_pc)
        compiled = None if plan is None else self._compile(plan)
        self.cache[key] = (proto, compiled)
        return compiled
