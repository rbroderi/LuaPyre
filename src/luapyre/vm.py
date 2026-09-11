from __future__ import annotations

from dataclasses import dataclass, field
import math

from .bytecode import Op, Proto, Closure, Cell
from .errors import LuaRuntimeError, LuaQuotaError
from .table import LuaTable
from .values import MultiValue, i64, lua_equal, static_value_type, truthy, type_matches


@dataclass(slots=True)
class HostFunction:
    fn: object
    name: str = "?"


@dataclass(slots=True)
class Frame:
    closure: Closure
    regs: list
    pc: int = 0
    return_reg: int = -1
    return_want: int = 0
    varargs: tuple[object, ...] = ()
    cells: dict[int, Cell] = field(default_factory=dict)

    @property
    def proto(self):
        return self.closure.proto


def _is_number(value):
    return type(value) in (int, float)


def _need_number(value):
    if not _is_number(value):
        raise LuaRuntimeError(f"attempt to perform arithmetic on a {static_value_type(value).name} value")
    return value


def _to_int(value):
    if type(value) is int:
        return i64(value)
    if type(value) is float and math.isfinite(value) and value.is_integer():
        iv = int(value)
        if -(1 << 63) <= iv <= (1 << 63) - 1:
            return iv
    raise LuaRuntimeError("number has no integer representation")


def _float_div(a, b):
    a = float(a); b = float(b)
    if b != 0.0:
        return a / b
    if a == 0.0:
        return math.nan
    return math.copysign(math.inf, a * (1.0 if math.copysign(1.0, b) > 0 else -1.0))


def _shift_left(a, n):
    a = _to_int(a) & ((1 << 64) - 1)
    n = _to_int(n)
    if n < 0:
        return _shift_right(a, -n)
    if n >= 64:
        return 0
    return i64((a << n) & ((1 << 64) - 1))


def _shift_right(a, n):
    a = _to_int(a) & ((1 << 64) - 1)
    n = _to_int(n)
    if n < 0:
        return _shift_left(a, -n)
    if n >= 64:
        return 0
    return i64(a >> n)


def _to_lua_string(value):
    if isinstance(value, bytes):
        return value
    if type(value) is int:
        return str(value).encode("ascii")
    if type(value) is float:
        return repr(value).encode("ascii")
    raise LuaRuntimeError("attempt to concatenate a non-string value")


class VM:
    def __init__(self, globals: LuaTable | None = None, fuel=1_000_000, max_frames=1000):
        self.globals = LuaTable() if globals is None else globals
        self.default_fuel = fuel
        self.max_frames = max_frames

    def run(self, proto: Proto, fuel=None):
        remaining = self.default_fuel if fuel is None else fuel
        root = Closure(proto, [], self.globals)
        root_regs = [None] * max(1, proto.register_count)
        if proto.env_reg >= 0:
            root_regs[proto.env_reg] = self.globals
        frames = [Frame(root, root_regs)]
        final_values: tuple[object, ...] = ()

        while frames:
            remaining -= 1
            if remaining < 0:
                raise LuaQuotaError("execution quota exceeded")

            frame = frames[-1]
            if frame.pc >= len(frame.proto.code):
                final_values = self._return(frames, frame, ())
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
            elif op is Op.LOCAL:
                value = regs[ins.b]
                regs[ins.a] = value
                if ins.a in frame.cells:
                    frame.cells[ins.a] = Cell(value)
            elif op is Op.GETCELL:
                cell = frame.cells.get(ins.b)
                regs[ins.a] = regs[ins.b] if cell is None else cell.value
            elif op is Op.SETCELL:
                cell = frame.cells.get(ins.a)
                if cell is None:
                    cell = frame.cells[ins.a] = Cell(regs[ins.a])
                cell.value = regs[ins.b]
                regs[ins.a] = regs[ins.b]
            elif op is Op.GETUPVAL:
                regs[ins.a] = frame.closure.upvalues[ins.b].value
            elif op is Op.SETUPVAL:
                frame.closure.upvalues[ins.a].value = regs[ins.b]
            elif op is Op.CLOSURE:
                child = frame.proto.children[ins.b]
                upvalues = []
                for desc in child.upvalues:
                    if desc.kind == "local":
                        cell = frame.cells.get(desc.index)
                        if cell is None:
                            cell = frame.cells[desc.index] = Cell(regs[desc.index])
                        upvalues.append(cell)
                    else:
                        upvalues.append(frame.closure.upvalues[desc.index])
                regs[ins.a] = Closure(child, upvalues, frame.closure.env)
            elif op is Op.GETGLOBAL:
                regs[ins.a] = frame.closure.env.rawget(constants[ins.b])
            elif op is Op.SETGLOBAL:
                frame.closure.env.rawset(constants[ins.b], regs[ins.a])
            elif op is Op.NEWTABLE:
                regs[ins.a] = LuaTable()
            elif op is Op.GETTABLE:
                table = regs[ins.b]
                if not isinstance(table, LuaTable):
                    raise LuaRuntimeError(f"attempt to index a {static_value_type(table).name} value")
                regs[ins.a] = table.rawget(regs[ins.c])
            elif op is Op.SETTABLE:
                table = regs[ins.a]
                if not isinstance(table, LuaTable):
                    raise LuaRuntimeError(f"attempt to index a {static_value_type(table).name} value")
                table.rawset(regs[ins.b], regs[ins.c])
            elif op is Op.SETLISTV:
                table = regs[ins.a]
                mv = regs[ins.c]
                if not isinstance(table, LuaTable) or not isinstance(mv, MultiValue):
                    raise LuaRuntimeError("invalid table list expansion")
                for offset, value in enumerate(mv.values):
                    table.rawset(ins.b + offset, value)
            elif op in (Op.ADD, Op.ADD_I, Op.ADD_F, Op.SUB, Op.SUB_I, Op.SUB_F, Op.MUL, Op.MUL_I, Op.MUL_F):
                a, b = regs[ins.b], regs[ins.c]
                _need_number(a); _need_number(b)
                if op in (Op.ADD, Op.ADD_I, Op.ADD_F):
                    value = a + b
                elif op in (Op.SUB, Op.SUB_I, Op.SUB_F):
                    value = a - b
                else:
                    value = a * b
                if op in (Op.ADD_I, Op.SUB_I, Op.MUL_I) or (op in (Op.ADD, Op.SUB, Op.MUL) and type(a) is int and type(b) is int):
                    value = i64(value)
                elif op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                    value = float(value)
                regs[ins.a] = value
            elif op is Op.DIV:
                regs[ins.a] = _float_div(_need_number(regs[ins.b]), _need_number(regs[ins.c]))
            elif op is Op.IDIV:
                a, b = _need_number(regs[ins.b]), _need_number(regs[ins.c])
                if b == 0:
                    raise LuaRuntimeError("attempt to divide by zero")
                q = math.floor(a / b)
                regs[ins.a] = i64(q) if type(a) is int and type(b) is int else float(q)
            elif op is Op.MOD:
                a, b = _need_number(regs[ins.b]), _need_number(regs[ins.c])
                if b == 0:
                    raise LuaRuntimeError("attempt to perform 'n%0'")
                value = a % b
                regs[ins.a] = i64(value) if type(a) is int and type(b) is int else float(value)
            elif op is Op.POW:
                regs[ins.a] = float(_need_number(regs[ins.b])) ** float(_need_number(regs[ins.c]))
            elif op is Op.NEG:
                value = _need_number(regs[ins.b])
                regs[ins.a] = i64(-value) if type(value) is int else -value
            elif op is Op.BAND:
                regs[ins.a] = i64(_to_int(regs[ins.b]) & _to_int(regs[ins.c]))
            elif op is Op.BOR:
                regs[ins.a] = i64(_to_int(regs[ins.b]) | _to_int(regs[ins.c]))
            elif op is Op.BXOR:
                regs[ins.a] = i64(_to_int(regs[ins.b]) ^ _to_int(regs[ins.c]))
            elif op is Op.BNOT:
                regs[ins.a] = i64(~_to_int(regs[ins.b]))
            elif op is Op.SHL:
                regs[ins.a] = _shift_left(regs[ins.b], regs[ins.c])
            elif op is Op.SHR:
                regs[ins.a] = _shift_right(regs[ins.b], regs[ins.c])
            elif op is Op.CONCAT:
                regs[ins.a] = _to_lua_string(regs[ins.b]) + _to_lua_string(regs[ins.c])
            elif op is Op.LEN:
                value = regs[ins.b]
                if isinstance(value, bytes):
                    regs[ins.a] = len(value)
                elif isinstance(value, LuaTable):
                    regs[ins.a] = value.rawlen()
                else:
                    raise LuaRuntimeError(f"attempt to get length of a {static_value_type(value).name} value")
            elif op is Op.NOT:
                regs[ins.a] = not truthy(regs[ins.b])
            elif op is Op.EQ:
                regs[ins.a] = lua_equal(regs[ins.b], regs[ins.c])
            elif op in (Op.LT, Op.LE):
                a, b = regs[ins.b], regs[ins.c]
                if _is_number(a) and _is_number(b):
                    result = a < b if op is Op.LT else a <= b
                elif isinstance(a, bytes) and isinstance(b, bytes):
                    result = a < b if op is Op.LT else a <= b
                else:
                    raise LuaRuntimeError("attempt to compare incompatible values")
                regs[ins.a] = result
            elif op is Op.JMP:
                frame.pc = ins.a
            elif op is Op.JMPIF:
                if truthy(regs[ins.b]):
                    frame.pc = ins.a
            elif op is Op.JMPIFNOT:
                if not truthy(regs[ins.b]):
                    frame.pc = ins.a
            elif op is Op.GUARD:
                expected = constants[ins.b]
                if not type_matches(expected, regs[ins.a]):
                    actual = static_value_type(regs[ins.a]).name
                    raise LuaRuntimeError(f"expected {expected}, got {actual}")
            elif op is Op.VARARG:
                if ins.b == -1:
                    regs[ins.a] = MultiValue(frame.varargs)
                else:
                    for i in range(ins.b):
                        regs[ins.a + i] = frame.varargs[i] if i < len(frame.varargs) else None
            elif op is Op.UNPACK:
                mv = regs[ins.b]
                values = mv.values if isinstance(mv, MultiValue) else (mv,)
                for i in range(ins.c):
                    regs[ins.a + i] = values[i] if i < len(values) else None
            elif op in (Op.CALL, Op.CALLV):
                fn = regs[ins.b]
                args = [regs[ins.c + i] for i in range(ins.d)]
                want = ins.e if op is Op.CALL else -1
                if op is Op.CALLV:
                    mv = regs[ins.e]
                    if isinstance(mv, MultiValue):
                        args.extend(mv.values)
                    else:
                        args.append(mv)
                if isinstance(fn, HostFunction):
                    try:
                        result = fn.fn(*args)
                    except LuaRuntimeError:
                        raise
                    except Exception as exc:
                        raise LuaRuntimeError(str(exc)) from None
                    values = result.values if isinstance(result, MultiValue) else (result,)
                    self._write_results(regs, ins.a, want, values)
                elif isinstance(fn, Closure):
                    if len(frames) >= self.max_frames:
                        raise LuaRuntimeError("stack overflow")
                    frames.append(self._new_frame(fn, args, ins.a, want))
                else:
                    raise LuaRuntimeError(f"attempt to call a {static_value_type(fn).name} value")
            elif op is Op.RETURN:
                values = tuple(regs[ins.a + i] for i in range(ins.b))
                final_values = self._return(frames, frame, values)
            elif op is Op.RETURNV:
                values = [regs[ins.a + i] for i in range(ins.b)]
                mv = regs[ins.c]
                if isinstance(mv, MultiValue):
                    values.extend(mv.values)
                else:
                    values.append(mv)
                final_values = self._return(frames, frame, tuple(values))
            elif op is Op.HALT:
                final_values = self._return(frames, frame, ())
            else:
                raise LuaRuntimeError(f"unsupported opcode {op}")

        if len(final_values) == 0:
            return None
        if len(final_values) == 1:
            return final_values[0]
        return final_values

    def _new_frame(self, closure: Closure, args, return_reg, return_want):
        proto = closure.proto
        regs = [None] * max(1, proto.register_count)
        for i in range(proto.param_count):
            arg = args[i] if i < len(args) else None
            typ = proto.param_types[i].name
            if not type_matches(typ, arg):
                raise LuaRuntimeError(f"argument {i + 1}: expected {typ}, got {static_value_type(arg).name}")
            regs[i] = arg
        extras = tuple(args[proto.param_count:]) if proto.is_vararg else ()
        if proto.is_vararg and proto.vararg_type.name != "Any":
            for i, arg in enumerate(extras, 1):
                if not type_matches(proto.vararg_type.name, arg):
                    raise LuaRuntimeError(f"vararg {i}: expected {proto.vararg_type.name}, got {static_value_type(arg).name}")
        if proto.vararg_name_reg >= 0:
            table = LuaTable.from_sequence(extras)
            table.rawset(b"n", len(extras))
            regs[proto.vararg_name_reg] = table
        return Frame(closure, regs, 0, return_reg, return_want, extras)

    @staticmethod
    def _write_results(regs, dest, want, values):
        if want == 0:
            return
        if want == -1:
            regs[dest] = MultiValue(tuple(values))
            return
        for i in range(want):
            regs[dest + i] = values[i] if i < len(values) else None

    def _return(self, frames, frame, values):
        frames.pop()
        if not frames:
            return tuple(values)
        parent = frames[-1]
        self._write_results(parent.regs, frame.return_reg, frame.return_want, values)
        return ()
