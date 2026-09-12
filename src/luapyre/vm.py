from __future__ import annotations

from dataclasses import dataclass, field

from .bytecode import Op, Proto, Closure, Cell
from .errors import LuaRuntimeError, LuaRaisedError, LuaQuotaError
from .opdispatch import OPCODE_HANDLERS
from .table import LuaTable
from .values import MultiValue, i64, static_value_type, truthy, type_matches


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
    # PUC-Lua bytecode closes values by register threshold rather than by the
    # compiler-maintained lexical close depth used by native LuaPyre bytecode.
    puc_close_stack: list[tuple[int, object]] = field(default_factory=list)
    pending_puc_close_reg: int | None = None
    # Tiered VMs bind this lazily to their per-Proto call-site array. Keeping
    # the array on the active frame removes dictionary/key construction from
    # the monomorphic CALL hot path. Tier 0 leaves it as None.
    jit_call_sites: list[object | None] | None = None

    @property
    def proto(self):
        return self.closure.proto


def _is_number(value):
    return type(value) in (int, float)


















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

    def _invoke_site(self, frames, parent, fn, args, dest, want, tail=False):
        """Call-site hook used by adaptive VMs; Tier 0 stays cache-free."""
        return self._invoke(frames, parent, fn, args, dest, want, tail=tail)

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
            if frame.puc_close_stack:
                _reg, value = frame.puc_close_stack.pop()
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

        if frame.pending_puc_close_reg is not None:
            target = frame.pending_puc_close_reg
            if frame.puc_close_stack and frame.puc_close_stack[-1][0] >= target:
                _reg, value = frame.puc_close_stack.pop()
                if value is None or value is False:
                    return True
                tm = self._tm(value, b"__close")
                if tm is None:
                    raise LuaRuntimeError("attempt to close a non-closable value")
                self._invoke(frames, frame, tm, [value], 0, 0)
                return True
            frame.pending_puc_close_reg = None

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
        handlers = OPCODE_HANDLERS

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
                result = handlers[ins.op](
                    self,
                    frames,
                    frame,
                    ins,
                    frame.regs,
                    frame.proto.constants,
                )
                if result is not None:
                    final_values = result

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
