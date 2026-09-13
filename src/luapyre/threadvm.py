from __future__ import annotations

from dataclasses import dataclass, field

from .bytecode import Closure, Ins, Op, Proto
from .errors import LuaQuotaError, LuaRaisedError, LuaRuntimeError
from .opdispatch import OPCODE_HANDLERS
from .table import LuaTable
from .values import MultiValue, lua_type_name, static_value_type
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
    yield_prefix: tuple[object, ...] = ()
    pending_tail_resume: tuple[object, ...] | None = None
    is_main: bool = False
    hook: object | None = None
    hook_mask: bytes = b""
    hook_count: int = 0
    hook_counter: int = 0
    hook_running: bool = False
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

    def __init__(
        self,
        globals: LuaTable | None = None,
        fuel=1_000_000,
        max_frames=1000,
        *,
        debug_hooks_enabled: bool = False,
    ):
        super().__init__(globals, fuel=fuel, max_frames=max_frames)
        self.debug_hooks_enabled = debug_hooks_enabled
        self.main_thread = LuaThread(None, status="running", started=True, is_main=True)
        self.current_thread: LuaThread | None = None
        self._coroutine_fuel = self.default_fuel

    def _hook_thread(self):
        return self.current_thread or self.main_thread

    def hooks_active(self) -> bool:
        return self.debug_hooks_enabled and self._hook_thread().hook is not None

    def set_hook(self, thread, hook, mask=b"", count=0):
        if not self.debug_hooks_enabled:
            raise LuaRuntimeError("debug hooks are disabled for this runtime")
        if thread is None:
            thread = self._hook_thread()
        if not isinstance(thread, LuaThread):
            raise LuaRuntimeError("bad argument #1 to 'sethook' (thread expected)")
        if hook is None:
            thread.hook = None
            thread.hook_mask = b""
            thread.hook_count = 0
            thread.hook_counter = 0
            return None
        if not isinstance(hook, (Closure, HostFunction)):
            raise LuaRuntimeError("bad argument to 'sethook' (function expected)")
        if not isinstance(mask, bytes):
            raise LuaRuntimeError("bad argument to 'sethook' (string expected)")
        if type(count) is float and count.is_integer():
            count = int(count)
        if type(count) is not int or not 0 <= count <= (1 << 24) - 1:
            raise LuaRuntimeError("bad argument to 'sethook' (count out of range)")
        thread.hook = hook
        thread.hook_mask = bytes(ch for ch in b"crl" if ch in mask)
        thread.hook_count = count
        thread.hook_counter = count
        gc = getattr(self, "gc", None)
        if gc is not None:
            gc.adopt(hook)
            gc.write_barrier(thread, hook)
        frames = (
            getattr(self, "_active_frames", None)
            if thread is self._hook_thread()
            else thread.frames
        ) or ()
        for frame in frames:
            pc = max(0, frame.pc - 1)
            frame.hook_call_pending = False
            frame.hook_last_pc = pc
            frame.hook_last_line = frame.proto.line_for_pc(pc)
        return None

    def get_hook(self, thread=None):
        if thread is None:
            thread = self._hook_thread()
        if not isinstance(thread, LuaThread):
            raise LuaRuntimeError("bad argument #1 to 'gethook' (thread expected)")
        return MultiValue((thread.hook, thread.hook_mask, thread.hook_count))

    def _dispatch_hook(
        self, event: bytes, line=None, *, subject=None, transfer=(), transfer_base=0
    ):
        thread = self._hook_thread()
        hook = thread.hook
        if hook is None or thread.hook_running:
            return
        thread.hook_running = True
        previous_subject = getattr(self, "_active_hook_subject", None)
        previous_transfer = getattr(self, "_active_hook_transfer", ())
        previous_transfer_base = getattr(self, "_active_hook_transfer_base", 0)
        self._active_hook_subject = subject
        self._active_hook_transfer = tuple(transfer)
        self._active_hook_transfer_base = transfer_base
        try:
            self.call_sync(hook, (event, line))
        finally:
            self._active_hook_subject = previous_subject
            self._active_hook_transfer = previous_transfer
            self._active_hook_transfer_base = previous_transfer_base
            thread.hook_running = False

    def _hook_instruction(self, frame, pc):
        thread = self._hook_thread()
        if thread.hook is None or thread.hook_running:
            return
        if frame.hook_call_pending:
            frame.hook_call_pending = False
            if b"c" in thread.hook_mask:
                self._dispatch_hook(
                    b"call",
                    transfer=frame.hook_call_values,
                    transfer_base=1,
                )
        # LOCAL and MOVE are LuaPyre register-shaping operations; PUC folds
        # them into surrounding instructions and does not charge separate
        # count-hook ticks for the equivalent empty-loop path.
        if thread.hook_count and frame.proto.code[pc].op not in (
            Op.LOADK, Op.LOCAL, Op.MOVE
        ):
            thread.hook_counter -= 1
            if thread.hook_counter <= 0:
                thread.hook_counter = thread.hook_count
                self._dispatch_hook(b"count")
        if b"l" in thread.hook_mask:
            line = frame.proto.line_for_pc(pc)
            if (
                line != frame.hook_last_line
                or pc <= frame.hook_last_pc
                or (frame.proto.debug_stripped and frame.hook_last_pc < 0)
            ):
                frame.hook_last_line = line
                frame.hook_last_pc = pc
                self._dispatch_hook(b"line", line if line >= 0 else None)
            else:
                frame.hook_last_pc = pc

    def hook_handlers(self, handlers):
        if not self.debug_hooks_enabled:
            return handlers
        wrapped = {}
        for op, handler in handlers.items():
            def hook_handler(vm, frames, frame, ins, regs, constants, _handler=handler):
                vm._hook_instruction(frame, frame.pc - 1)
                return _handler(vm, frames, frame, ins, regs, constants)
            wrapped[op] = hook_handler
        return wrapped

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
                if fn.protected_mode is not None:
                    return self._invoke_protected(
                        frames, parent, fn.protected_mode, args, dest, want, tail
                    )
                thread = self._hook_thread()
                trace_host = (
                    self.debug_hooks_enabled
                    and thread.hook is not None
                    and not thread.hook_running
                )
                if trace_host and b"c" in thread.hook_mask:
                    self._dispatch_hook(
                        b"call", subject=fn, transfer=args, transfer_base=1
                    )
                try:
                    values = self._host_values(fn, args)
                except _YieldSignal as signal:
                    thread = self.current_thread
                    if thread is None or thread.is_main:
                        raise LuaRuntimeError("attempt to yield from outside a coroutine")
                    if (
                        self.debug_hooks_enabled
                        and thread.hook is not None
                        and not thread.hook_running
                        and b"r" in thread.hook_mask
                    ):
                        self._dispatch_hook(b"return", subject=fn)
                    thread.yielded = signal.values
                    thread.yield_target = (parent, dest, want, tail)
                    thread.yield_prefix = ()
                    raise
                thread = self._hook_thread()
                if (
                    self.debug_hooks_enabled
                    and thread.hook is not None
                    and not thread.hook_running
                    and b"r" in thread.hook_mask
                ):
                    self._dispatch_hook(
                        b"return", subject=fn, transfer=values, transfer_base=1
                    )
                if tail:
                    return self._return(frames, parent, values)
                self._write_results(parent.regs, dest, want, values)
                return None
            if isinstance(fn, Closure):
                if tail:
                    thread = self._hook_thread()
                    parent.is_tailcall = True
                    if (
                        self.debug_hooks_enabled
                        and thread.hook is not None
                        and not thread.hook_running
                        and b"c" in thread.hook_mask
                    ):
                        self._dispatch_hook(
                            b"tail call",
                            transfer=args[:fn.proto.param_count],
                            transfer_base=1,
                        )
                    replacement = self._new_frame(
                        fn, args, parent.return_reg, parent.return_want
                    )
                    replacement.hook_call_pending = False
                    replacement.is_tailcall = True
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

    def _return(self, frames, frame, values):
        thread = self._hook_thread()
        if (
            self.debug_hooks_enabled
            and frame.proto.name != "<stdlib-callback>"
            and thread.hook is not None
            and not thread.hook_running
            and b"r" in thread.hook_mask
        ):
            active = [
                item
                for item in frame.proto.debug_locals
                if item[2] <= max(0, frame.pc - 1) <= item[3]
                and item[0] != "(vararg table)"
            ]
            self._dispatch_hook(
                b"return",
                transfer=values,
                transfer_base=len(active) + 1,
            )
        return super()._return(frames, frame, values)

    def _invoke_protected(self, frames, parent, mode, args, dest, want, tail=False):
        try:
            return super()._invoke_protected(
                frames, parent, mode, args, dest, want, tail
            )
        except _YieldSignal as signal:
            thread = self.current_thread
            if thread is None or thread.is_main:
                raise LuaRuntimeError("attempt to yield from outside a coroutine")
            thread.yielded = signal.values
            thread.yield_target = (parent, dest, want, tail)
            thread.yield_prefix = (True,)
            raise

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
        return (
            not thread.is_main
            and thread.status != "dead"
            and not getattr(self, "_sync_frame_prefixes", ())
        )

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
            elif isinstance(thread.entry, HostFunction) and thread.entry.protected_mode is not None:
                trampoline = Proto(
                    "<coroutine-entry>",
                    code=[Ins(Op.RETURNV, 0, 0, 0)],
                    register_count=1,
                    source=None,
                )
                parent_frame = Frame(Closure(trampoline, [], self.globals), [None])
                thread.frames = [parent_frame]
                self._invoke(
                    thread.frames,
                    parent_frame,
                    thread.entry,
                    list(args),
                    0,
                    -1,
                )
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
            args = (*thread.yield_prefix, *args)
            thread.yield_prefix = ()
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
        previous_direct_protected = getattr(self, "_direct_protected_host", False)
        self._direct_protected_host = False
        try:
            event, values = self._execute_thread(thread)
        finally:
            self._direct_protected_host = previous_direct_protected
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
            recursive_resume = thread is self.current_thread
            result = self.resume_thread(thread, *args)
            values = result.values
            if not values[0]:
                error = values[1] if len(values) > 1 else b"coroutine failed"
                # A failed wrapped coroutine is reset, which closes suspended
                # scopes and lets a __close failure replace the original
                # error. A recursive call to the currently running wrapper is
                # the one exception: that thread cannot be reset in-place.
                if not recursive_resume:
                    closed = self.close_thread(thread)
                    if isinstance(closed, MultiValue):
                        close_values = closed.values
                        if not close_values[0] and len(close_values) > 1:
                            error = close_values[1]
                raise LuaRaisedError(error)
            return MultiValue(tuple(values[1:]))

        result = HostFunction(wrapped, "coroutine.wrap", _gc_refs=(thread,))
        gc = getattr(self, "gc", None)
        if gc is not None:
            gc.adopt(result)
        return result

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
            error = self._error_value(thread.error)
            thread.error = None
            return MultiValue((False, error))

        error = self._force_close(thread, thread.error)
        thread.status = "dead"
        # Closing consumes the coroutine's stored failure. Report it once;
        # subsequent close calls on the dead thread succeed idempotently.
        thread.error = None
        if error is None:
            return True
        return MultiValue((False, self._error_value(error)))

    def _force_close(self, thread, error):
        previous = self.current_thread
        previous_direct_protected = getattr(self, "_direct_protected_host", False)
        self.current_thread = thread
        self._direct_protected_host = False
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
            self._direct_protected_host = previous_direct_protected
            self.current_thread = previous

    def _tick(self):
        self._coroutine_fuel -= 1
        if self._coroutine_fuel < 0:
            raise LuaQuotaError("execution quota exceeded")

    def _execute_thread(self, thread: LuaThread, stop_depth: int | None = None):
        frames = thread.frames
        final_values = ()
        handlers = self.hook_handlers(OPCODE_HANDLERS)

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
                if not frames or not any(frame.protected_name for frame in frames):
                    thread.error = exc
                    thread.status = "dead"
                    return "error", (self._error_value(exc),)
                frames[-1].pending_error = exc

        return "return", tuple(final_values)
