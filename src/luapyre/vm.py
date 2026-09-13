from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

from .bytecode import Ins, Op, Proto, Closure, Cell
from .errors import LuaRuntimeError, LuaRaisedError, LuaQuotaError
from .opdispatch import OPCODE_HANDLERS
from .table import LuaTable
from .values import MultiValue, i64, lua_type_name, parse_lua_number, static_value_type, truthy, type_matches


@dataclass(slots=True)
class HostFunction:
    fn: object
    name: str = "?"
    max_args: int | None = None
    protected_mode: str | None = None
    _gc_refs: tuple[object, ...] = ()
    _gc_owner: object = None
    _gc_age: int = 0


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
    return_prefix: tuple[object, ...] = ()
    return_limit: int = -1
    protected_handler: object | None = None
    protected_name: str | None = None
    protected_error: LuaRuntimeError | None = None
    protected_error_depth: int = 0
    trace_name: str | None = None
    call_name: str | None = None
    call_namewhat: str = ""
    hook_call_pending: bool = True
    hook_last_pc: int = -1
    hook_last_line: int = -1
    hook_call_values: tuple[object, ...] = ()
    is_tailcall: bool = False

    @property
    def proto(self):
        return self.closure.proto


def _is_number(value):
    return type(value) in (int, float)


















class VM:
    # Lua 5.5 accepts at most fifteen chained tag-method redirects.  The
    # extra iteration observes the callable at the end of a valid chain.
    MAXTAGLOOP = 16
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
            if fn.max_args is not None and len(args) > fn.max_args:
                args = args[: fn.max_args]
            result = fn.fn(*args)
        except LuaRuntimeError as error:
            origins = getattr(self, "_host_call_origins", ())
            origin = origins[-1] if origins else None
            message = str(error)
            if origin is not None and origin[0] == "method" and message.startswith("bad argument #"):
                match = re.match(r"bad argument #(\d+)(.*)", message)
                if match:
                    index = int(match.group(1))
                    message = (
                        "bad self" + match.group(2)
                        if index == 1
                        else f"bad argument #{index - 1}" + match.group(2)
                    )
                    error.args = (message,)
            raise
        except RecursionError:
            # CPython's recursion guard is an implementation detail.  Lua code
            # observes the conventional protected-call diagnostic instead.
            raise LuaRuntimeError("C stack overflow") from None
        except Exception as exc:
            raise LuaRuntimeError(str(exc)) from None
        return result.values if isinstance(result, MultiValue) else (result,)

    def _invoke(self, frames, parent, fn, args, dest, want, tail=False):
        args = list(args)
        for _ in range(self.MAXTAGLOOP):
            if isinstance(fn, HostFunction):
                if fn.protected_mode is not None:
                    return self._invoke_protected(
                        frames, parent, fn.protected_mode, args, dest, want, tail
                    )
                values = self._host_values(fn, args)
                if tail:
                    return self._return(frames, parent, values)
                self._write_results(parent.regs, dest, want, values)
                return None
            if isinstance(fn, Closure):
                if tail:
                    replacement = self._new_frame(
                        fn, args, parent.return_reg, parent.return_want
                    )
                    replacement.return_prefix = parent.return_prefix
                    replacement.return_limit = parent.return_limit
                    replacement.protected_handler = parent.protected_handler
                    replacement.protected_name = parent.protected_name
                    replacement.protected_error = parent.protected_error
                    replacement.protected_error_depth = parent.protected_error_depth
                    replacement.trace_name = parent.trace_name
                    replacement.call_name = parent.call_name
                    replacement.call_namewhat = parent.call_namewhat
                    frames[-1] = replacement
                    return None
                if len(frames) >= self.max_frames:
                    raise LuaRuntimeError("stack overflow")
                frames.append(self._new_frame(fn, args, dest, want))
                return None
            tm = self._tm(fn, b"__call")
            if tm is None:
                raise LuaRuntimeError(f"attempt to call a {lua_type_name(fn)} value")
            args.insert(0, fn)
            fn = tm
        raise LuaRuntimeError("'__call' chain too long; possible loop")

    def _callable_target(self, fn, args):
        args = list(args)
        for _ in range(self.MAXTAGLOOP):
            if isinstance(fn, (Closure, HostFunction)):
                return fn, args
            tm = self._tm(fn, b"__call")
            if tm is None:
                raise LuaRuntimeError(f"attempt to call a {lua_type_name(fn)} value")
            args.insert(0, fn)
            fn = tm
        raise LuaRuntimeError("'__call' chain too long; possible loop")

    def _invoke_protected(self, frames, parent, mode, args, dest, want, tail=False):
        if not args:
            self._write_results(parent.regs, dest, want, (False, b"function expected"))
            return None
        target_args = args[2:] if mode == "xpcall" else args[1:]
        handler = args[1] if mode == "xpcall" and len(args) > 1 else None
        try:
            target, target_args = self._callable_target(args[0], target_args)
            if isinstance(target, HostFunction) and target.protected_mode is not None:
                if tail:
                    return_reg, return_want = parent.return_reg, parent.return_want
                else:
                    return_reg, return_want = dest, want
                trampoline_proto = Proto(
                    "<protected-call>",
                    code=[Ins(Op.RETURNV, 0, 0, 0)],
                    register_count=1,
                    source=None,
                )
                trampoline = Frame(
                    Closure(trampoline_proto, [], self.globals),
                    [None],
                    return_reg=return_reg,
                    return_want=return_want,
                )
                trampoline.return_prefix = (True,)
                trampoline.protected_handler = handler
                trampoline.protected_name = mode
                if tail:
                    frames[-1] = trampoline
                else:
                    if len(frames) >= self.max_frames:
                        raise LuaRuntimeError("stack overflow")
                    frames.append(trampoline)
                return self._invoke_protected(
                    frames,
                    trampoline,
                    target.protected_mode,
                    target_args,
                    0,
                    -1,
                    False,
                )
            if isinstance(target, HostFunction) and target.protected_mode is None:
                previous_direct = getattr(self, "_direct_protected_host", False)
                self._direct_protected_host = True
                try:
                    values = self._host_values(target, target_args)
                finally:
                    self._direct_protected_host = previous_direct
                values = (True, *values)
                if tail:
                    return self._return(frames, parent, values)
                self._write_results(parent.regs, dest, want, values)
                return None
            if not isinstance(target, Closure):
                raise LuaRuntimeError("cannot protect this call target")
            if tail:
                if len(frames) >= self.max_frames:
                    raise LuaRuntimeError("stack overflow")
                trampoline_proto = Proto(
                    "<protected-call>",
                    code=[Ins(Op.RETURNV, 0, 0, 0)],
                    register_count=1,
                    source=None,
                )
                trampoline = Frame(
                    Closure(trampoline_proto, [], self.globals),
                    [None],
                    return_reg=parent.return_reg,
                    return_want=parent.return_want,
                )
                trampoline.return_prefix = parent.return_prefix
                trampoline.return_limit = parent.return_limit
                trampoline.protected_handler = parent.protected_handler
                trampoline.protected_name = parent.protected_name
                trampoline.protected_error = parent.protected_error
                trampoline.protected_error_depth = parent.protected_error_depth
                trampoline.trace_name = parent.trace_name
                frames[-1] = trampoline
                frame = self._new_frame(target, target_args, 0, -1)
                frames.append(frame)
            else:
                if len(frames) >= self.max_frames:
                    raise LuaRuntimeError("stack overflow")
                frame = self._new_frame(target, target_args, dest, want)
                frames.append(frame)
            frame.return_prefix = (True,)
            frame.protected_handler = handler
            frame.protected_name = mode
            return None
        except LuaQuotaError:
            raise
        except LuaRuntimeError as error:
            if handler is not None:
                error_object = self._error_object(error)
                try:
                    resolved, handler_args = self._callable_target(
                        handler, [error_object]
                    )
                    if isinstance(resolved, HostFunction) and resolved.protected_mode is None:
                        try:
                            handled = self._host_values(resolved, handler_args)
                            replacement = handled[0] if handled else None
                        except LuaRuntimeError:
                            replacement = b"error in error handling"
                        values = (False, replacement)
                        if tail:
                            return self._return(frames, parent, values)
                        self._write_results(parent.regs, dest, want, values)
                        return None
                    if isinstance(resolved, Closure):
                        callback = self._new_frame(resolved, handler_args, dest, want)
                        callback.return_prefix = (False,)
                        callback.return_limit = 1
                        callback.protected_name = "xpcall handler"
                        callback.protected_handler = resolved
                        callback.protected_error = error
                        callback.protected_error_depth = 1
                        if tail:
                            frames[-1] = callback
                        else:
                            if len(frames) >= self.max_frames:
                                raise LuaRuntimeError("stack overflow")
                            frames.append(callback)
                        return None
                except LuaRuntimeError:
                    error = LuaRaisedError(b"error in error handling")
            values = (False, self._error_object(error))
            if tail:
                return self._return(frames, parent, values)
            self._write_results(parent.regs, dest, want, values)
            return None

    def _finish_protected_error(self, frames, frame, error):
        error_object = self._error_object(error)
        fatal_handler_error = (
            frame.protected_name == "xpcall handler"
            and isinstance(error_object, bytes)
            and b"stack overflow" in error_object
        )
        if fatal_handler_error:
            error = LuaRaisedError(b"error in error handling")
        handler = frame.protected_handler
        if fatal_handler_error:
            handler = None
        handler_depth = frame.protected_error_depth
        return_reg, return_want = frame.return_reg, frame.return_want
        frames.pop()
        if not frames:
            raise error
        parent = frames[-1]
        if handler is None:
            self._write_results(
                parent.regs, return_reg, return_want,
                (False, self._error_object(error)),
            )
            return True
        if handler_depth >= 200:
            current = self._error_object(error)
            exhausted = (
                b"error in error handling"
                if isinstance(current, bytes) and b"stack overflow" in current
                else b"C stack overflow"
            )
            self._write_results(
                parent.regs,
                return_reg,
                return_want,
                (False, exhausted),
            )
            return True
        try:
            handler, handler_args = self._callable_target(
                handler, [self._error_object(error)]
            )
            if isinstance(handler, HostFunction) and handler.protected_mode is None:
                previous_error = getattr(self, "_active_protected_error", None)
                self._active_protected_error = error
                try:
                    try:
                        values = self._host_values(handler, handler_args)
                        replacement = values[0] if values else None
                    except LuaRuntimeError as handler_error:
                        replacement = self._error_object(handler_error)
                finally:
                    self._active_protected_error = previous_error
                self._write_results(
                    parent.regs, return_reg, return_want, (False, replacement)
                )
                return True
            if not isinstance(handler, Closure):
                raise LuaRuntimeError("error handler is not callable")
            callback = self._new_frame(handler, handler_args, return_reg, return_want)
            callback.return_prefix = (False,)
            callback.return_limit = 1
            callback.protected_name = "xpcall handler"
            callback.protected_error = error
            callback.protected_handler = handler
            callback.protected_error_depth = handler_depth + 1
            frames.append(callback)
            return True
        except LuaRuntimeError as handler_error:
            self._write_results(
                parent.regs, return_reg, return_want,
                (False, self._error_object(handler_error)),
            )
            return True

    def _invoke_site(self, frames, parent, fn, args, dest, want, tail=False):
        """Call-site hook used by adaptive VMs; Tier 0 stays cache-free."""
        return self._invoke(frames, parent, fn, args, dest, want, tail=tail)

    def _invoke_metamethod(self, frames, parent, fn, args, dest, want, name):
        depth = len(frames)
        result = self._invoke(frames, parent, fn, args, dest, want)
        if len(frames) > depth:
            frames[-1].call_name = name.removeprefix(b"__").decode("ascii")
            frames[-1].call_namewhat = "metamethod"
        return result

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
                    raise LuaRuntimeError(f"attempt to index a {lua_type_name(obj)} value")
            if isinstance(tm, (Closure, HostFunction)):
                self._invoke_metamethod(
                    frames, frame, tm, [obj, key], dest, 1, b"__index"
                )
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
                    raise LuaRuntimeError(f"attempt to index a {lua_type_name(obj)} value")
            if isinstance(tm, (Closure, HostFunction)):
                self._invoke_metamethod(
                    frames, frame, tm, [obj, key, value], 0, 0, b"__newindex"
                )
                return
            obj = tm
        raise LuaRuntimeError("'__newindex' chain too long; possible loop")




    def _forprep(self, regs, ins):
        idx, limit, step = regs[ins.a], regs[ins.b], regs[ins.c]
        converted = []
        for label, value in (("initial", idx), ("limit", limit), ("step", step)):
            number = value if _is_number(value) else parse_lua_number(value)
            if number is None:
                typename = lua_type_name(value)
                custom = self._tm(value, b"__name")
                if isinstance(custom, bytes):
                    typename = custom.decode("utf-8", "replace")
                raise LuaRuntimeError(
                    f"'for' {label} value must be a number "
                    f"(got {typename})"
                )
            converted.append(number)
        idx, limit, step = converted
        regs[ins.a], regs[ins.b], regs[ins.c] = idx, limit, step
        if step == 0:
            raise LuaRuntimeError("'for' step is zero")
        if type(idx) is int and type(step) is int:
            if type(limit) is float:
                if math.isnan(limit):
                    return False
                if step > 0:
                    if limit < -(1 << 63):
                        return False
                    limit = (
                        (1 << 63) - 1
                        if math.isinf(limit)
                        else min((1 << 63) - 1, math.floor(limit))
                    )
                else:
                    if limit > (1 << 63) - 1:
                        return False
                    limit = (
                        -(1 << 63)
                        if math.isinf(limit)
                        else max(-(1 << 63), math.ceil(limit))
                    )
                regs[ins.b] = limit
        else:
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
                    frame.pending_error = LuaRuntimeError("metamethod 'close' is not callable")
                    return True
                depth = len(frames)
                self._invoke(
                    frames,
                    frame,
                    tm,
                    [value, self._error_object(frame.pending_error)],
                    0,
                    0,
                )
                if len(frames) > depth:
                    frames[-1].trace_name = "__close"
                return True
            if frame.close_stack:
                value = frame.close_stack.pop()
                if value is None or value is False:
                    return True
                tm = self._tm(value, b"__close")
                if tm is None:
                    frame.pending_error = LuaRuntimeError("metamethod 'close' is not callable")
                    return True
                depth = len(frames)
                self._invoke(
                    frames,
                    frame,
                    tm,
                    [value, self._error_object(frame.pending_error)],
                    0,
                    0,
                )
                if len(frames) > depth:
                    frames[-1].trace_name = "__close"
                return True
            error = frame.pending_error
            if frame.protected_name is not None:
                return self._finish_protected_error(frames, frame, error)
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
                    raise LuaRuntimeError("metamethod 'close' is not callable")
                depth = len(frames)
                self._invoke(frames, frame, tm, [value], 0, 0)
                if len(frames) > depth:
                    frames[-1].trace_name = "__close"
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
                    raise LuaRuntimeError("metamethod 'close' is not callable")
                depth = len(frames)
                self._invoke(frames, frame, tm, [value], 0, 0)
                if len(frames) > depth:
                    frames[-1].trace_name = "__close"
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
        frame = Frame(closure, regs, 0, return_reg, return_want, extras)
        frame.hook_call_values = tuple(args[:proto.param_count])
        return frame

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
        values = tuple(values)
        if frame.return_limit >= 0:
            values = values[:frame.return_limit]
        if frame.return_prefix:
            values = (*frame.return_prefix, *values)
        frames.pop()
        if not frames:
            return tuple(values)
        parent = frames[-1]
        self._write_results(parent.regs, frame.return_reg, frame.return_want, values)
        return ()
