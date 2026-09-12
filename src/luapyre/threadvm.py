from __future__ import annotations

from dataclasses import dataclass, field

from .bytecode import Closure, Proto
from .errors import LuaQuotaError, LuaRaisedError, LuaRuntimeError
from .opdispatch import OPCODE_HANDLERS
from .table import LuaTable
from .values import MultiValue, static_value_type
from .vm import Frame, HostFunction, VM as BaseVM


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
    _gc_owner: object = None
    _gc_age: int = 0

    def __repr__(self):
        return f"thread: 0x{id(self):x}"


class CoroutineVM(BaseVM):
    """Base register VM plus persistent Lua coroutine stacks.

    Main-chunk and coroutine execution share the same opcode handler table.
    LuaThread owns coroutine frame lists so suspension preserves registers,
    PCs, closures, upvalues, and pending close state exactly where execution
    stopped.
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
        thread = LuaThread(fn)
        gc = getattr(self, "gc", None)
        if gc is not None:
            gc.safepoint(self._active_frames or ())
            gc.adopt(thread)
        return thread

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
        handlers = OPCODE_HANDLERS

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
                    if not frames:
                        return "return", tuple(final_values)

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
