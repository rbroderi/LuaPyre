from __future__ import annotations

from dataclasses import dataclass, field
import math

from .bytecode import Op, Proto, Closure, Cell
from .errors import LuaRuntimeError, LuaRaisedError, LuaQuotaError
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
    close_stack: list[object] = field(default_factory=list)
    pending_close_target: int | None = None
    pending_error: LuaRuntimeError | None = None

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
    a = float(a)
    b = float(b)
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
    MAXTAGLOOP = 2000
    ARITH_TM = {
        Op.ADD: b"__add", Op.SUB: b"__sub", Op.MUL: b"__mul",
        Op.DIV: b"__div", Op.IDIV: b"__idiv", Op.MOD: b"__mod", Op.POW: b"__pow",
        Op.BAND: b"__band", Op.BOR: b"__bor", Op.BXOR: b"__bxor",
        Op.SHL: b"__shl", Op.SHR: b"__shr",
    }

    def __init__(self, globals: LuaTable | None = None, fuel=1_000_000, max_frames=1000):
        self.globals = LuaTable() if globals is None else globals
        self.default_fuel = fuel
        self.max_frames = max_frames

    def _tm(self, value, name):
        if isinstance(value, LuaTable) and isinstance(value.metatable, LuaTable):
            return value.metatable.rawget(name)
        return None

    def _first_tm(self, left, right, name):
        return self._tm(left, name) or self._tm(right, name)

    def _host_values(self, fn, args):
        try:
            result = fn.fn(*args)
        except LuaRuntimeError:
            raise
        except Exception as exc:
            raise LuaRuntimeError(str(exc)) from None
        return result.values if isinstance(result, MultiValue) else (result,)

    def _invoke(self, frames, parent, fn, args, dest, want, tail=False):
        args = list(args)
        for _ in range(self.MAXTAGLOOP):
            if isinstance(fn, HostFunction):
                values = self._host_values(fn, args)
                if tail:
                    return self._return(frames, parent, values)
                self._write_results(parent.regs, dest, want, values)
                return None
            if isinstance(fn, Closure):
                if tail:
                    frames[-1] = self._new_frame(fn, args, parent.return_reg, parent.return_want)
                    return None
                if len(frames) >= self.max_frames:
                    raise LuaRuntimeError("stack overflow")
                frames.append(self._new_frame(fn, args, dest, want))
                return None
            tm = self._tm(fn, b"__call")
            if tm is None:
                raise LuaRuntimeError(f"attempt to call a {static_value_type(fn).name} value")
            args.insert(0, fn)
            fn = tm
        raise LuaRuntimeError("'__call' chain too long; possible loop")

    def _gettable(self, frames, frame, obj, key, dest):
        for _ in range(self.MAXTAGLOOP):
            if isinstance(obj, LuaTable):
                if obj.rawhas(key):
                    frame.regs[dest] = obj.rawget(key)
                    return
                tm = self._tm(obj, b"__index")
                if tm is None:
                    frame.regs[dest] = None
                    return
            else:
                tm = self._tm(obj, b"__index")
                if tm is None:
                    raise LuaRuntimeError(f"attempt to index a {static_value_type(obj).name} value")
            if isinstance(tm, (Closure, HostFunction)) or self._tm(tm, b"__call") is not None:
                self._invoke(frames, frame, tm, [obj, key], dest, 1)
                return
            obj = tm
        raise LuaRuntimeError("'__index' chain too long; possible loop")

    def _settable(self, frames, frame, obj, key, value):
        for _ in range(self.MAXTAGLOOP):
            if isinstance(obj, LuaTable):
                if obj.rawhas(key):
                    obj.rawset(key, value)
                    return
                tm = self._tm(obj, b"__newindex")
                if tm is None:
                    obj.rawset(key, value)
                    return
            else:
                tm = self._tm(obj, b"__newindex")
                if tm is None:
                    raise LuaRuntimeError(f"attempt to index a {static_value_type(obj).name} value")
            if isinstance(tm, (Closure, HostFunction)) or self._tm(tm, b"__call") is not None:
                self._invoke(frames, frame, tm, [obj, key, value], 0, 0)
                return
            obj = tm
        raise LuaRuntimeError("'__newindex' chain too long; possible loop")

    def _arith_primitive(self, op, a, b):
        if op in (Op.ADD, Op.SUB, Op.MUL):
            if not (_is_number(a) and _is_number(b)):
                return False, None
            value = a + b if op is Op.ADD else a - b if op is Op.SUB else a * b
            return True, i64(value) if type(a) is int and type(b) is int else value
        if op is Op.DIV:
            if not (_is_number(a) and _is_number(b)):
                return False, None
            return True, _float_div(a, b)
        if op is Op.IDIV:
            if not (_is_number(a) and _is_number(b)):
                return False, None
            if b == 0:
                raise LuaRuntimeError("attempt to divide by zero")
            q = math.floor(a / b)
            return True, i64(q) if type(a) is int and type(b) is int else float(q)
        if op is Op.MOD:
            if not (_is_number(a) and _is_number(b)):
                return False, None
            if b == 0:
                raise LuaRuntimeError("attempt to perform 'n%0'")
            value = a % b
            return True, i64(value) if type(a) is int and type(b) is int else float(value)
        if op is Op.POW:
            if not (_is_number(a) and _is_number(b)):
                return False, None
            return True, float(a) ** float(b)
        if op in (Op.BAND, Op.BOR, Op.BXOR, Op.SHL, Op.SHR):
            try:
                ai = _to_int(a)
                bi = _to_int(b)
            except LuaRuntimeError:
                return False, None
            if op is Op.BAND:
                value = i64(ai & bi)
            elif op is Op.BOR:
                value = i64(ai | bi)
            elif op is Op.BXOR:
                value = i64(ai ^ bi)
            elif op is Op.SHL:
                value = _shift_left(ai, bi)
            else:
                value = _shift_right(ai, bi)
            return True, value
        return False, None

    def _generic_binary(self, frames, frame, op, a, b, dest):
        ok, value = self._arith_primitive(op, a, b)
        if ok:
            frame.regs[dest] = value
            return
        tm = self._first_tm(a, b, self.ARITH_TM[op])
        if tm is None:
            raise LuaRuntimeError(f"attempt to perform arithmetic on a {static_value_type(a).name} value")
        self._invoke(frames, frame, tm, [a, b], dest, 1)

    def _compare(self, frames, frame, op, a, b, dest):
        if op is Op.EQ:
            raw = lua_equal(a, b)
            if raw:
                frame.regs[dest] = True
                return
            if not (isinstance(a, LuaTable) and isinstance(b, LuaTable)):
                frame.regs[dest] = False
                return
            tm = self._first_tm(a, b, b"__eq")
            if tm is None:
                frame.regs[dest] = False
                return
            self._invoke(frames, frame, tm, [a, b], dest, 1)
            return
        if _is_number(a) and _is_number(b):
            frame.regs[dest] = a < b if op is Op.LT else a <= b
            return
        if isinstance(a, bytes) and isinstance(b, bytes):
            frame.regs[dest] = a < b if op is Op.LT else a <= b
            return
        tm = self._first_tm(a, b, b"__lt" if op is Op.LT else b"__le")
        if tm is None:
            raise LuaRuntimeError("attempt to compare incompatible values")
        self._invoke(frames, frame, tm, [a, b], dest, 1)

    def _forprep(self, regs, ins):
        idx, limit, step = regs[ins.a], regs[ins.b], regs[ins.c]
        if not all(_is_number(value) for value in (idx, limit, step)):
            raise LuaRuntimeError("'for' limit must be a number")
        if step == 0:
            raise LuaRuntimeError("'for' step is zero")
        if any(type(value) is float for value in (idx, limit, step)):
            idx, limit, step = float(idx), float(limit), float(step)
            regs[ins.a], regs[ins.b], regs[ins.c] = idx, limit, step
        return idx <= limit if step > 0 else idx >= limit

    def _forloop(self, regs, ins):
        idx, limit, step = regs[ins.a], regs[ins.b], regs[ins.c]
        if type(idx) is int and type(limit) is int and type(step) is int:
            nxt = idx + step
            if nxt < -(1 << 63) or nxt > (1 << 63) - 1:
                return False
            if (step > 0 and nxt > limit) or (step < 0 and nxt < limit):
                return False
            regs[ins.a] = nxt
            return True
        nxt = float(idx) + float(step)
        if (step > 0 and nxt > limit) or (step < 0 and nxt < limit):
            return False
        regs[ins.a] = nxt
        return True

    @staticmethod
    def _error_object(error):
        if isinstance(error, LuaRaisedError):
            return error.value
        return str(error).encode("utf-8", "replace")

    def _drive_pending(self, frames, frame):
        if frame.pending_error is not None:
            if frame.close_stack:
                value = frame.close_stack.pop()
                if value is None or value is False:
                    return True
                tm = self._tm(value, b"__close")
                if tm is None:
                    frame.pending_error = LuaRuntimeError("attempt to close a non-closable value")
                    return True
                self._invoke(
                    frames,
                    frame,
                    tm,
                    [value, self._error_object(frame.pending_error)],
                    0,
                    0,
                )
                return True
            error = frame.pending_error
            frames.pop()
            if frames:
                frames[-1].pending_error = error
                return True
            raise error

        if frame.pending_close_target is not None:
            target = frame.pending_close_target
            if len(frame.close_stack) > target:
                value = frame.close_stack.pop()
                if value is None or value is False:
                    return True
                tm = self._tm(value, b"__close")
                if tm is None:
                    raise LuaRuntimeError("attempt to close a non-closable value")
                self._invoke(frames, frame, tm, [value], 0, 0)
                return True
            frame.pending_close_target = None
        return False

    def run(self, proto: Proto, fuel=None):
        remaining = self.default_fuel if fuel is None else fuel
        root = Closure(proto, [], self.globals)
        root_regs = [None] * max(1, proto.register_count)
        if proto.env_reg >= 0:
            root_regs[proto.env_reg] = self.globals
        frames = [Frame(root, root_regs)]
        final_values = ()

        while frames:
            try:
                frame = frames[-1]
                if self._drive_pending(frames, frame):
                    continue

                remaining -= 1
                if remaining < 0:
                    raise LuaQuotaError("execution quota exceeded")
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
                    self._gettable(frames, frame, regs[ins.b], regs[ins.c], ins.a)
                elif op is Op.SETTABLE:
                    self._settable(frames, frame, regs[ins.a], regs[ins.b], regs[ins.c])
                elif op is Op.SETLISTV:
                    table = regs[ins.a]
                    mv = regs[ins.c]
                    if not isinstance(table, LuaTable) or not isinstance(mv, MultiValue):
                        raise LuaRuntimeError("invalid table list expansion")
                    for offset, value in enumerate(mv.values):
                        table.rawset(ins.b + offset, value)
                elif op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                    a, b = _need_number(regs[ins.b]), _need_number(regs[ins.c])
                    value = a + b if op is Op.ADD_I else a - b if op is Op.SUB_I else a * b
                    regs[ins.a] = i64(value)
                elif op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                    a, b = _need_number(regs[ins.b]), _need_number(regs[ins.c])
                    value = a + b if op is Op.ADD_F else a - b if op is Op.SUB_F else a * b
                    regs[ins.a] = float(value)
                elif op in self.ARITH_TM:
                    self._generic_binary(frames, frame, op, regs[ins.b], regs[ins.c], ins.a)
                elif op is Op.NEG:
                    value = regs[ins.b]
                    if _is_number(value):
                        regs[ins.a] = i64(-value) if type(value) is int else -value
                    else:
                        tm = self._tm(value, b"__unm")
                        if tm is None:
                            raise LuaRuntimeError(f"attempt to perform arithmetic on a {static_value_type(value).name} value")
                        self._invoke(frames, frame, tm, [value], ins.a, 1)
                elif op is Op.BNOT:
                    value = regs[ins.b]
                    try:
                        regs[ins.a] = i64(~_to_int(value))
                    except LuaRuntimeError:
                        tm = self._tm(value, b"__bnot")
                        if tm is None:
                            raise
                        self._invoke(frames, frame, tm, [value], ins.a, 1)
                elif op is Op.CONCAT:
                    a, b = regs[ins.b], regs[ins.c]
                    try:
                        regs[ins.a] = _to_lua_string(a) + _to_lua_string(b)
                    except LuaRuntimeError:
                        tm = self._first_tm(a, b, b"__concat")
                        if tm is None:
                            raise
                        self._invoke(frames, frame, tm, [a, b], ins.a, 1)
                elif op is Op.LEN:
                    value = regs[ins.b]
                    if isinstance(value, bytes):
                        regs[ins.a] = len(value)
                    elif isinstance(value, LuaTable):
                        tm = self._tm(value, b"__len")
                        if tm is None:
                            regs[ins.a] = value.rawlen()
                        else:
                            self._invoke(frames, frame, tm, [value], ins.a, 1)
                    else:
                        tm = self._tm(value, b"__len")
                        if tm is None:
                            raise LuaRuntimeError(f"attempt to get length of a {static_value_type(value).name} value")
                        self._invoke(frames, frame, tm, [value], ins.a, 1)
                elif op is Op.NOT:
                    regs[ins.a] = not truthy(regs[ins.b])
                elif op is Op.TOBOOL:
                    regs[ins.a] = truthy(regs[ins.b])
                elif op in (Op.EQ, Op.LT, Op.LE):
                    self._compare(frames, frame, op, regs[ins.b], regs[ins.c], ins.a)
                elif op is Op.JMP:
                    frame.pc = ins.a
                elif op is Op.JMPIF:
                    if truthy(regs[ins.b]):
                        frame.pc = ins.a
                elif op is Op.JMPIFNOT:
                    if not truthy(regs[ins.b]):
                        frame.pc = ins.a
                elif op is Op.JMPIFNIL:
                    if regs[ins.b] is None:
                        frame.pc = ins.a
                elif op is Op.FORPREP:
                    if not self._forprep(regs, ins):
                        frame.pc = ins.d
                elif op is Op.FORLOOP:
                    if self._forloop(regs, ins):
                        frame.pc = ins.d
                elif op is Op.GUARD:
                    expected = constants[ins.b]
                    if not type_matches(expected, regs[ins.a]):
                        raise LuaRuntimeError(f"expected {expected}, got {static_value_type(regs[ins.a]).name}")
                elif op is Op.CHECKNIL:
                    if regs[ins.a] is not None:
                        name = constants[ins.b]
                        if isinstance(name, bytes):
                            name = name.decode("utf-8", "replace")
                        raise LuaRuntimeError(f"global '{name}' already has a value")
                elif op is Op.TBC:
                    value = regs[ins.a]
                    if value is not None and value is not False:
                        if self._tm(value, b"__close") is None:
                            raise LuaRuntimeError("variable got a non-closable value")
                        frame.close_stack.append(value)
                elif op is Op.CLOSE:
                    if ins.a < 0 or ins.a > len(frame.close_stack):
                        raise LuaRuntimeError("invalid close depth")
                    frame.pending_close_target = ins.a
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
                elif op in (Op.CALL, Op.CALLV, Op.TAILCALL, Op.TAILCALLV):
                    tail = op in (Op.TAILCALL, Op.TAILCALLV)
                    fn = regs[ins.b]
                    args = [regs[ins.c + i] for i in range(ins.d)]
                    want = ins.e if op is Op.CALL else -1
                    if op in (Op.CALLV, Op.TAILCALLV):
                        mv = regs[ins.e]
                        args.extend(mv.values if isinstance(mv, MultiValue) else (mv,))
                    returned = self._invoke(frames, frame, fn, args, ins.a, want, tail=tail)
                    if tail and returned is not None:
                        final_values = returned
                elif op is Op.RETURN:
                    final_values = self._return(frames, frame, tuple(regs[ins.a + i] for i in range(ins.b)))
                elif op is Op.RETURNV:
                    values = [regs[ins.a + i] for i in range(ins.b)]
                    mv = regs[ins.c]
                    values.extend(mv.values if isinstance(mv, MultiValue) else (mv,))
                    final_values = self._return(frames, frame, tuple(values))
                elif op is Op.HALT:
                    final_values = self._return(frames, frame, ())
                else:
                    raise LuaRuntimeError(f"unsupported opcode {op}")

            except LuaQuotaError:
                raise
            except LuaRuntimeError as exc:
                if not frames:
                    raise
                frames[-1].pending_error = exc

        if len(final_values) == 0:
            return None
        if len(final_values) == 1:
            return final_values[0]
        return final_values

    def _new_frame(self, closure, args, return_reg, return_want):
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
