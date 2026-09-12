from __future__ import annotations

from dataclasses import dataclass, field

from .bytecode import Cell, Closure, Op, Proto
from .errors import LuaQuotaError, LuaRaisedError, LuaRuntimeError
from .table import LuaTable
from .values import MultiValue, lua_equal, static_value_type, truthy, type_matches
from .vm import (
    Frame,
    HostFunction,
    VM as BaseVM,
    _is_number,
    _need_number,
    _to_int,
    _to_lua_string,
)


class _YieldSignal(BaseException):
    def __init__(self, values):
        self.values = tuple(values)


class _CloseSelfSignal(BaseException):
    pass


@dataclass(slots=True)
class LuaThread:
    entry: object | None
    frames: list[Frame] = field(default_factory=list)
    status: str = "suspended"
    started: bool = False
    error: LuaRuntimeError | None = None
    yielded: tuple[object, ...] = ()
    yield_target: tuple[Frame, int, int, bool] | None = None
    pending_tail_resume: tuple[object, ...] | None = None
    is_main: bool = False

    def __repr__(self):
        return f"thread: 0x{id(self):x}"


class CoroutineVM(BaseVM):
    """Base register VM plus persistent Lua coroutine stacks.

    The ordinary main chunk still uses the proven BaseVM execution path. Lua
    coroutines use the same Frame objects and opcodes, but their frame lists are
    owned by LuaThread so they can survive an explicit yield and later resume at
    the exact suspended instruction continuation.
    """

    def __init__(self, globals: LuaTable | None = None, fuel=1_000_000, max_frames=1000):
        super().__init__(globals, fuel=fuel, max_frames=max_frames)
        self.main_thread = LuaThread(None, status="running", started=True, is_main=True)
        self.current_thread: LuaThread | None = None
        self._coroutine_fuel = self.default_fuel

    def run(self, proto: Proto, fuel=None):
        previous = self.current_thread
        self.current_thread = self.main_thread
        self.main_thread.status = "running"
        try:
            return super().run(proto, fuel=fuel)
        finally:
            self.current_thread = previous

    def _invoke(self, frames, parent, fn, args, dest, want, tail=False):
        args = list(args)
        for _ in range(self.MAXTAGLOOP):
            if isinstance(fn, HostFunction):
                try:
                    values = self._host_values(fn, args)
                except _YieldSignal as signal:
                    thread = self.current_thread
                    if thread is None or thread.is_main:
                        raise LuaRuntimeError("attempt to yield from outside a coroutine")
                    thread.yielded = signal.values
                    thread.yield_target = (parent, dest, want, tail)
                    raise
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

    @staticmethod
    def _error_value(error):
        if isinstance(error, LuaRaisedError):
            return error.value
        return str(error).encode("utf-8", "replace")

    def create_thread(self, fn):
        if not isinstance(fn, (Closure, HostFunction)):
            raise LuaRuntimeError("bad argument #1 to 'create' (function expected)")
        return LuaThread(fn)

    def running_thread(self):
        thread = self.current_thread or self.main_thread
        return MultiValue((thread, thread.is_main))

    def coroutine_status(self, thread):
        if not isinstance(thread, LuaThread):
            raise LuaRuntimeError("bad argument #1 to 'status' (thread expected)")
        if thread is self.current_thread:
            return b"running"
        return thread.status.encode("ascii")

    def is_yieldable(self, thread=None):
        if thread is None:
            thread = self.current_thread or self.main_thread
        if not isinstance(thread, LuaThread):
            raise LuaRuntimeError("bad argument #1 to 'isyieldable' (thread expected)")
        return not thread.is_main and thread.status != "dead"

    def yield_current(self, *values):
        thread = self.current_thread
        if thread is None or thread.is_main:
            raise LuaRuntimeError("attempt to yield from outside a coroutine")
        raise _YieldSignal(values)

    def resume_thread(self, thread, *args):
        if not isinstance(thread, LuaThread):
            raise LuaRuntimeError("bad argument #1 to 'resume' (thread expected)")
        if thread.status == "dead":
            return MultiValue((False, b"cannot resume dead coroutine"))
        if thread.status in ("running", "normal") or thread is self.current_thread:
            return MultiValue((False, b"cannot resume non-suspended coroutine"))

        if not thread.started:
            thread.started = True
            thread.error = None
            if isinstance(thread.entry, Closure):
                thread.frames = [self._new_frame(thread.entry, list(args), -1, 0)]
            else:
                parent = self.current_thread
                if parent is not None:
                    parent.status = "normal"
                previous = self.current_thread
                self.current_thread = thread
                thread.status = "running"
                try:
                    values = self._host_values(thread.entry, list(args))
                except _YieldSignal:
                    thread.status = "dead"
                    return MultiValue((False, b"attempt to yield across a host function"))
                except LuaRuntimeError as exc:
                    thread.status = "dead"
                    thread.error = exc
                    return MultiValue((False, self._error_value(exc)))
                finally:
                    self.current_thread = previous
                    if parent is not None:
                        parent.status = "running"
                thread.status = "dead"
                return MultiValue((True, *values))
        else:
            if thread.yield_target is None:
                return MultiValue((False, b"cannot resume dead coroutine"))
            frame, dest, want, tail = thread.yield_target
            thread.yield_target = None
            thread.yielded = ()
            if tail:
                thread.pending_tail_resume = tuple(args)
            else:
                self._write_results(frame.regs, dest, want, tuple(args))

        parent = self.current_thread
        if parent is not None:
            parent.status = "normal"
        previous = self.current_thread
        self.current_thread = thread
        thread.status = "running"
        self._coroutine_fuel = self.default_fuel
        try:
            event, values = self._execute_thread(thread)
        finally:
            self.current_thread = previous
            if parent is not None:
                parent.status = "running"

        if event == "yield":
            thread.status = "suspended"
            return MultiValue((True, *values))
        if event == "return":
            thread.status = "dead"
            thread.frames.clear()
            thread.error = None
            return MultiValue((True, *values))
        thread.status = "dead"
        return MultiValue((False, *values))

    def wrap_thread(self, fn):
        thread = self.create_thread(fn)

        def wrapped(*args):
            result = self.resume_thread(thread, *args)
            values = result.values
            if not values[0]:
                error = values[1] if len(values) > 1 else b"coroutine failed"
                self.close_thread(thread)
                raise LuaRaisedError(error)
            return MultiValue(tuple(values[1:]))

        return HostFunction(wrapped, "coroutine.wrap")

    def close_thread(self, thread=None):
        if thread is None:
            thread = self.current_thread
        if not isinstance(thread, LuaThread):
            raise LuaRuntimeError("bad argument #1 to 'close' (thread expected)")
        if thread is self.current_thread:
            if thread.is_main:
                raise LuaRuntimeError("cannot close the main thread")
            raise _CloseSelfSignal()
        if thread.status == "normal" or thread.status == "running":
            raise LuaRuntimeError("cannot close a normal coroutine")
        if not thread.frames:
            if thread.error is None:
                thread.status = "dead"
                return True
            return MultiValue((False, self._error_value(thread.error)))

        error = self._force_close(thread, thread.error)
        thread.status = "dead"
        thread.error = error
        if error is None:
            return True
        return MultiValue((False, self._error_value(error)))

    def _force_close(self, thread, error):
        previous = self.current_thread
        self.current_thread = thread
        old_status = thread.status
        thread.status = "running"
        try:
            while thread.frames:
                frame = thread.frames[-1]
                while frame.close_stack:
                    value = frame.close_stack.pop()
                    if value is None or value is False:
                        continue
                    tm = self._tm(value, b"__close")
                    if tm is None:
                        error = LuaRuntimeError("attempt to close a non-closable value")
                        continue
                    args = [value]
                    if error is not None:
                        args.append(self._error_value(error))
                    baseline = len(thread.frames)
                    try:
                        self._invoke(thread.frames, frame, tm, args, 0, 0)
                        if len(thread.frames) > baseline:
                            event, values = self._execute_thread(thread, stop_depth=baseline)
                            if event == "yield":
                                error = LuaRuntimeError("attempt to yield while closing a coroutine")
                                while len(thread.frames) > baseline:
                                    thread.frames.pop()
                            elif event == "error":
                                error = thread.error or LuaRuntimeError("error closing coroutine")
                                while len(thread.frames) > baseline:
                                    thread.frames.pop()
                    except _YieldSignal:
                        error = LuaRuntimeError("attempt to yield while closing a coroutine")
                    except LuaRuntimeError as exc:
                        error = exc
                if thread.frames and thread.frames[-1] is frame:
                    thread.frames.pop()
            return error
        finally:
            thread.status = "dead" if not thread.frames else old_status
            self.current_thread = previous

    def _tick(self):
        self._coroutine_fuel -= 1
        if self._coroutine_fuel < 0:
            raise LuaQuotaError("execution quota exceeded")

    def _execute_thread(self, thread: LuaThread, stop_depth: int | None = None):
        frames = thread.frames
        final_values = ()

        if thread.pending_tail_resume is not None and frames:
            values = thread.pending_tail_resume
            thread.pending_tail_resume = None
            final_values = self._return(frames, frames[-1], values)
            if not frames:
                return "return", tuple(final_values)

        while frames:
            if stop_depth is not None and len(frames) <= stop_depth:
                return "callback", ()
            try:
                frame = frames[-1]
                if self._drive_pending(frames, frame):
                    if stop_depth is not None and len(frames) <= stop_depth:
                        return "callback", ()
                    continue

                self._tick()
                if frame.pc >= len(frame.proto.code):
                    final_values = self._return(frames, frame, ())
                    if not frames:
                        return "return", tuple(final_values)
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
                    from .values import i64
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
                        from .values import i64
                        regs[ins.a] = i64(-value) if type(value) is int else -value
                    else:
                        tm = self._tm(value, b"__unm")
                        if tm is None:
                            raise LuaRuntimeError(f"attempt to perform arithmetic on a {static_value_type(value).name} value")
                        self._invoke(frames, frame, tm, [value], ins.a, 1)
                elif op is Op.BNOT:
                    value = regs[ins.b]
                    try:
                        from .values import i64
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
                        if not frames:
                            return "return", tuple(final_values)
                elif op is Op.RETURN:
                    final_values = self._return(frames, frame, tuple(regs[ins.a + i] for i in range(ins.b)))
                    if not frames:
                        return "return", tuple(final_values)
                elif op is Op.RETURNV:
                    values = [regs[ins.a + i] for i in range(ins.b)]
                    mv = regs[ins.c]
                    values.extend(mv.values if isinstance(mv, MultiValue) else (mv,))
                    final_values = self._return(frames, frame, tuple(values))
                    if not frames:
                        return "return", tuple(final_values)
                elif op is Op.HALT:
                    final_values = self._return(frames, frame, ())
                    if not frames:
                        return "return", tuple(final_values)
                else:
                    raise LuaRuntimeError(f"unsupported opcode {op}")

            except _YieldSignal as signal:
                thread.status = "suspended"
                return "yield", signal.values
            except _CloseSelfSignal:
                error = self._force_close(thread, None)
                thread.status = "dead"
                thread.error = error
                if error is None:
                    return "return", ()
                return "error", (self._error_value(error),)
            except LuaQuotaError:
                raise
            except LuaRuntimeError as exc:
                thread.error = exc
                thread.status = "dead"
                return "error", (self._error_value(exc),)

        return "return", tuple(final_values)
