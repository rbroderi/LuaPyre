from __future__ import annotations

from dataclasses import dataclass
from types import FunctionType

from .bytecode import Op, Proto
from .opdispatch import _float_divide, _float_modulo
from .table import LuaTable
from .values import i64, lua_equal


COROUTINE_DEOPT = 0
COROUTINE_YIELD = 1
COROUTINE_RETURN = 2

_CONTROL = frozenset(
    {
        Op.JMP,
        Op.JMPIF,
        Op.JMPIFNOT,
        Op.JMPIFNIL,
        Op.FORPREP,
        Op.FORLOOP,
        Op.JFORLOOP,
        Op.RETURN,
        Op.HALT,
        Op.CALL,
    }
)

_OPS = frozenset(
    {
        Op.LOADK,
        Op.MOVE,
        Op.LOCAL,
        Op.GETUPVAL,
        Op.GETTABLE,
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
        Op.MOD,
        Op.EQ,
        Op.LT,
        Op.LE,
        Op.NOT,
        Op.TOBOOL,
        Op.JMP,
        Op.JMPIF,
        Op.JMPIFNOT,
        Op.JMPIFNIL,
        Op.FORPREP,
        Op.FORLOOP,
        Op.JFORLOOP,
        Op.CALL,
        Op.RETURN,
        Op.HALT,
    }
)


@dataclass(frozen=True, slots=True)
class CompiledCoroutine:
    proto: Proto
    runner: FunctionType


def compile_coroutine(proto: Proto) -> CompiledCoroutine | None:
    """Compile an eligible coroutine body into resumable basic blocks.

    The real Lua Frame remains the authoritative state object. Generated code
    advances ``frame.pc`` at every suspension boundary, so yield/resume, GC
    roots, tracebacks, and interpreter fallback all observe the same registers.
    CALL is admitted only as a guarded call to the runtime's exact
    ``coroutine.yield`` HostFunction; any other target deoptimizes before the
    call executes.
    """
    if proto.is_vararg or proto.children or not proto.code:
        return None
    if any(ins.op not in _OPS for ins in proto.code):
        return None

    leaders = {0, len(proto.code)}
    for pc, ins in enumerate(proto.code):
        if ins.op is Op.JMP:
            leaders.update((ins.a, pc + 1))
        elif ins.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
            leaders.update((ins.a, pc + 1))
        elif ins.op in (Op.FORPREP, Op.FORLOOP, Op.JFORLOOP):
            leaders.update((ins.d, pc + 1))
        elif ins.op in (Op.CALL, Op.RETURN, Op.HALT):
            leaders.add(pc + 1)
    if any(pc < 0 or pc > len(proto.code) for pc in leaders):
        return None
    ordered = sorted(leaders)
    blocks = [
        (start, ordered[index + 1])
        for index, start in enumerate(ordered[:-1])
        if start < ordered[index + 1]
    ]
    block_for = {start: index for index, (start, _end) in enumerate(blocks)}
    if any(
        target not in block_for and target != len(proto.code)
        for pc, ins in enumerate(proto.code)
        for target in (
            (ins.a, pc + 1)
            if ins.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL)
            else (ins.a,)
            if ins.op is Op.JMP
            else (ins.d, pc + 1)
            if ins.op in (Op.FORPREP, Op.FORLOOP, Op.JFORLOOP)
            else ()
        )
    ):
        return None

    lines = [
        "def _run(vm, thread, frame, budget):",
        "    regs = frame.regs",
        "    consts = frame.proto.constants",
        "    upvalues = frame.closure.upvalues",
        "    used = 0",
        "    state = frame.pc",
        "    while True:",
        "        match state:",
    ]

    def deopt(pc: int, indent: str, condition: str) -> None:
        lines.extend(
            [
                f"{indent}if {condition}:",
                f"{indent}    frame.pc = {pc}",
                f"{indent}    return used, _DEOPT, None",
            ]
        )

    def state(target: int) -> int:
        return target

    for start, end in blocks:
        lines.append(f"            case {start}:")
        indent = "                "
        cost = end - start
        lines.extend(
            [
                f"{indent}if budget - used < {cost}:",
                f"{indent}    frame.pc = {start}",
                f"{indent}    return used, _DEOPT, None",
            ]
        )
        terminal = proto.code[end - 1]
        terminal_control = terminal.op in _CONTROL
        ordinary_end = end - 1 if terminal_control else end

        for pc in range(start, ordinary_end):
            ins = proto.code[pc]
            a, b, c = f"regs[{ins.a}]", f"regs[{ins.b}]", f"regs[{ins.c}]"
            if ins.op is Op.LOADK:
                lines.extend([f"{indent}{a} = consts[{ins.b}]", f"{indent}used += 1"])
            elif ins.op in (Op.MOVE, Op.LOCAL):
                lines.extend([f"{indent}{a} = {b}", f"{indent}used += 1"])
            elif ins.op is Op.GETUPVAL:
                lines.extend(
                    [f"{indent}{a} = upvalues[{ins.b}].value", f"{indent}used += 1"]
                )
            elif ins.op is Op.GETTABLE:
                deopt(pc, indent, f"not isinstance({b}, _LuaTable) or {b}.metatable is not None")
                lines.extend([f"{indent}{a} = {b}.rawget({c})", f"{indent}used += 1"])
            elif ins.op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                symbol = {Op.ADD_I: "+", Op.SUB_I: "-", Op.MUL_I: "*"}[ins.op]
                deopt(pc, indent, f"type({b}) is not int or type({c}) is not int")
                lines.extend([f"{indent}{a} = _i64({b} {symbol} {c})", f"{indent}used += 1"])
            elif ins.op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                symbol = {Op.ADD_F: "+", Op.SUB_F: "-", Op.MUL_F: "*"}[ins.op]
                deopt(pc, indent, f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES")
                lines.extend([f"{indent}{a} = float({b} {symbol} {c})", f"{indent}used += 1"])
            elif ins.op in (Op.ADD, Op.SUB, Op.MUL):
                symbol = {Op.ADD: "+", Op.SUB: "-", Op.MUL: "*"}[ins.op]
                deopt(pc, indent, f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES")
                lines.extend(
                    [
                        f"{indent}_left, _right = {b}, {c}",
                        f"{indent}{a} = _i64(_left {symbol} _right) if type(_left) is int and type(_right) is int else _left {symbol} _right",
                        f"{indent}used += 1",
                    ]
                )
            elif ins.op is Op.DIV:
                deopt(pc, indent, f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES")
                lines.extend([f"{indent}{a} = _float_divide({b}, {c})", f"{indent}used += 1"])
            elif ins.op is Op.MOD:
                deopt(pc, indent, f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES")
                lines.extend(
                    [
                        f"{indent}_left, _right = {b}, {c}",
                        f"{indent}if type(_left) is int and type(_right) is int:",
                        f"{indent}    if _right == 0:",
                        f"{indent}        frame.pc = {pc}",
                        f"{indent}        return used, _DEOPT, None",
                        f"{indent}    {a} = _i64(_left % _right)",
                        f"{indent}else:",
                        f"{indent}    {a} = _float_modulo(_left, _right)",
                        f"{indent}used += 1",
                    ]
                )
            elif ins.op is Op.EQ:
                lines.extend([f"{indent}{a} = _lua_equal({b}, {c})", f"{indent}used += 1"])
            elif ins.op in (Op.LT, Op.LE):
                symbol = "<" if ins.op is Op.LT else "<="
                deopt(
                    pc,
                    indent,
                    f"not ((type({b}) in _NUM_TYPES and type({c}) in _NUM_TYPES) or (isinstance({b}, bytes) and isinstance({c}, bytes)))",
                )
                lines.extend([f"{indent}{a} = {b} {symbol} {c}", f"{indent}used += 1"])
            elif ins.op is Op.NOT:
                lines.extend([f"{indent}{a} = ({b} is None or {b} is False)", f"{indent}used += 1"])
            elif ins.op is Op.TOBOOL:
                lines.extend([f"{indent}{a} = not ({b} is None or {b} is False)", f"{indent}used += 1"])
            else:
                return None

        if not terminal_control:
            lines.extend([f"{indent}state = {state(end)}", f"{indent}continue"])
            continue

        pc, ins = end - 1, terminal
        if ins.op is Op.JMP:
            lines.extend([f"{indent}used += 1", f"{indent}state = {state(ins.a)}", f"{indent}continue"])
        elif ins.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
            if ins.op is Op.JMPIFNIL:
                condition = f"regs[{ins.b}] is None"
            else:
                condition = f"not (regs[{ins.b}] is None or regs[{ins.b}] is False)"
                if ins.op is Op.JMPIFNOT:
                    condition = f"not ({condition})"
            lines.extend(
                [
                    f"{indent}used += 1",
                    f"{indent}state = {state(ins.a)} if {condition} else {state(end)}",
                    f"{indent}continue",
                ]
            )
        elif ins.op is Op.FORPREP:
            lines.extend(
                [
                    f"{indent}frame.pc = {pc + 1}",
                    f"{indent}_enter = vm._forprep(regs, frame.proto.code[{pc}])",
                    f"{indent}used += 1",
                    f"{indent}state = {state(end)} if _enter else {state(ins.d)}",
                    f"{indent}continue",
                ]
            )
        elif ins.op in (Op.FORLOOP, Op.JFORLOOP):
            lines.extend(
                [
                    f"{indent}_again = vm._forloop(regs, frame.proto.code[{pc}])",
                    f"{indent}used += 1",
                    f"{indent}state = {state(ins.d)} if _again else {state(end)}",
                    f"{indent}continue",
                ]
            )
        elif ins.op is Op.CALL:
            args = ", ".join(f"regs[{ins.c + index}]" for index in range(ins.d))
            if ins.d == 1:
                args += ","
            lines.extend(
                [
                    f"{indent}_fn = regs[{ins.b}]",
                    f"{indent}if _fn is not vm._coroutine_yield_function:",
                    f"{indent}    frame.pc = {pc}",
                    f"{indent}    return used, _DEOPT, None",
                    f"{indent}_values = ({args})",
                    f"{indent}used += 1",
                    f"{indent}frame.pc = {pc + 1}",
                    f"{indent}thread.yielded = _values",
                    f"{indent}thread.yield_target = (frame, {ins.a}, {ins.e}, False)",
                    f"{indent}thread.yield_prefix = ()",
                    f"{indent}return used, _YIELD, _values",
                ]
            )
        elif ins.op is Op.RETURN:
            values = ", ".join(f"regs[{ins.a + index}]" for index in range(ins.b))
            if ins.b == 1:
                values += ","
            lines.extend(
                [
                    f"{indent}used += 1",
                    f"{indent}frame.pc = {pc + 1}",
                    f"{indent}return used, _RETURN, ({values})",
                ]
            )
        elif ins.op is Op.HALT:
            lines.extend(
                [
                    f"{indent}used += 1",
                    f"{indent}frame.pc = {pc + 1}",
                    f"{indent}return used, _RETURN, ()",
                ]
            )
        else:
            return None

    lines.extend(
        [
            "            case _:",
            "                frame.pc = state",
            "                return used, _DEOPT, None",
        ]
    )
    namespace = {
        "_DEOPT": COROUTINE_DEOPT,
        "_YIELD": COROUTINE_YIELD,
        "_RETURN": COROUTINE_RETURN,
        "_LuaTable": LuaTable,
        "_NUM_TYPES": (int, float),
        "_i64": i64,
        "_float_divide": _float_divide,
        "_float_modulo": _float_modulo,
        "_lua_equal": lua_equal,
    }
    exec(compile("\n".join(lines), "<luapyre-coroutine-jit>", "exec"), namespace)
    return CompiledCoroutine(proto, namespace["_run"])
