from __future__ import annotations

from dataclasses import dataclass
import math
from .bytecode import Op, Proto, Closure
from .errors import LuaRuntimeError, LuaQuotaError
from .typesys import python_value_type

MASK64 = (1 << 64) - 1
SIGN64 = 1 << 63


def i64(value: int) -> int:
    value &= MASK64
    return value - (1 << 64) if value & SIGN64 else value


def truthy(value):
    return value is not None and value is not False


@dataclass(slots=True)
class HostFunction:
    fn: object


@dataclass(slots=True)
class Frame:
    proto: Proto
    regs: list
    pc: int = 0
    return_reg: int = -1


class VM:
    def __init__(self, globals=None, fuel=1_000_000):
        self.globals = {} if globals is None else globals
        self.default_fuel = fuel

    def run(self, proto: Proto, fuel=None):
        remaining = self.default_fuel if fuel is None else fuel
        frames = [Frame(proto, [None] * max(1, proto.register_count))]
        final = None

        while frames:
            remaining -= 1
            if remaining < 0:
                raise LuaQuotaError("execution quota exceeded")

            frame = frames[-1]
            if frame.pc >= len(frame.proto.code):
                value = None
                frames.pop()
                if frames and frame.return_reg >= 0:
                    frames[-1].regs[frame.return_reg] = value
                else:
                    final = value
                continue

            ins = frame.proto.code[frame.pc]
            frame.pc += 1
            op = ins.op
            regs = frame.regs
            constants = frame.proto.constants

            if op is Op.LOADK:
                regs[ins.a] = constants[ins.b]
            elif op is Op.MOVE:
                regs[ins.a] = regs[ins.b]
            elif op is Op.GETGLOBAL:
                regs[ins.a] = self.globals.get(constants[ins.b])
            elif op is Op.SETGLOBAL:
                self.globals[constants[ins.b]] = regs[ins.a]
            elif op in (
                Op.ADD, Op.ADD_I, Op.ADD_F,
                Op.SUB, Op.SUB_I, Op.SUB_F,
                Op.MUL, Op.MUL_I, Op.MUL_F,
            ):
                a, b = regs[ins.b], regs[ins.c]
                try:
                    if op in (Op.ADD, Op.ADD_I, Op.ADD_F):
                        value = a + b
                    elif op in (Op.SUB, Op.SUB_I, Op.SUB_F):
                        value = a - b
                    else:
                        value = a * b
                except Exception as exc:
                    raise LuaRuntimeError(str(exc)) from None
                if op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                    value = i64(value)
                elif op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                    value = float(value)
                regs[ins.a] = value
            elif op is Op.DIV:
                regs[ins.a] = float(regs[ins.b]) / float(regs[ins.c])
            elif op is Op.IDIV:
                regs[ins.a] = math.floor(regs[ins.b] / regs[ins.c])
            elif op is Op.MOD:
                regs[ins.a] = regs[ins.b] % regs[ins.c]
            elif op is Op.POW:
                regs[ins.a] = float(regs[ins.b]) ** float(regs[ins.c])
            elif op is Op.NEG:
                regs[ins.a] = -regs[ins.b]
            elif op is Op.NOT:
                regs[ins.a] = not truthy(regs[ins.b])
            elif op is Op.EQ:
                regs[ins.a] = regs[ins.b] == regs[ins.c]
            elif op is Op.LT:
                regs[ins.a] = regs[ins.b] < regs[ins.c]
            elif op is Op.LE:
                regs[ins.a] = regs[ins.b] <= regs[ins.c]
            elif op is Op.JMP:
                frame.pc = ins.a
            elif op is Op.JMPIFNOT:
                if not truthy(regs[ins.b]):
                    frame.pc = ins.a
            elif op is Op.GUARD:
                expected = constants[ins.b]
                actual = python_value_type(regs[ins.a]).name
                if expected == "number" and actual in ("integer", "float"):
                    pass
                elif expected != actual:
                    raise LuaRuntimeError(f"expected {expected}, got {actual}")
            elif op is Op.CALL:
                fn = regs[ins.b]
                meta = frame.proto.code[frame.pc]
                frame.pc += 1
                if meta.op is not Op.LOADK:
                    raise LuaRuntimeError("invalid CALL metadata")
                argc = constants[meta.b]
                args = [regs[ins.c + i] for i in range(argc)]

                if isinstance(fn, HostFunction):
                    regs[ins.a] = fn.fn(*args)
                elif isinstance(fn, Closure):
                    child = fn.proto
                    child_regs = [None] * max(1, child.register_count)
                    for i, arg in enumerate(args[:child.param_count]):
                        typ = child.param_types[i].name
                        actual = python_value_type(arg).name
                        if typ != "Any" and not (
                            typ == "number" and actual in ("integer", "float")
                        ) and actual != typ:
                            raise LuaRuntimeError(
                                f"argument {i + 1}: expected {typ}, got {actual}"
                            )
                        child_regs[i] = arg
                    frames.append(Frame(child, child_regs, 0, ins.a))
                else:
                    raise LuaRuntimeError("attempt to call a non-function value")
            elif op is Op.RETURN:
                value = regs[ins.a] if ins.b else None
                frames.pop()
                if frames:
                    if frame.return_reg >= 0:
                        frames[-1].regs[frame.return_reg] = value
                else:
                    final = value
            elif op is Op.HALT:
                frames.pop()
                final = None if not frames else final
            else:
                raise LuaRuntimeError(f"unsupported opcode {op}")

        return final
