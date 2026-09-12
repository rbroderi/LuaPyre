from __future__ import annotations

from .bytecode import Cell, Closure, Proto
from .diagnostics import capture_error, error_value
from .errors import LuaQuotaError, LuaRuntimeError
from .gc import LuaGC
from .opdispatch import OPCODE_HANDLERS
from .table import LuaTable
from .threadvm import CoroutineVM, _CloseSelfSignal, _YieldSignal
from .values import MultiValue, truthy
from .vm import Frame


class GarbageCollectedVM(CoroutineVM):
    """Coroutine VM with Lua-level GC root tracking and source diagnostics.

    Allocation polls and host calls are safe points where ``collectgarbage`` can
    run. Keeping the active frame list and actual host-call arguments visible
    lets the collector trace semantic roots without adding a check to every
    opcode.

    Standard-library helpers sometimes need to call back into Lua synchronously
    (for example ``__pairs``, ``__tostring``, table sort comparators, and
    protected calls). Those callbacks reuse the same opcode handlers. A prefix
    of the suspended outer Lua frames is retained while the nested callback is
    executing so an explicit collection cannot lose caller roots.
    """

    def __init__(self, globals: LuaTable | None = None, fuel=1_000_000, max_frames=1000):
        super().__init__(globals, fuel=fuel, max_frames=max_frames)
        self._active_frames = None
        self._active_host_values = None
        self._active_call_result = None
        self._sync_frame_prefixes: list[list[Frame]] = []
        self.type_metatables: dict[bytes, LuaTable] = {}
        self.gc = LuaGC(self)
        self.gc.adopt(self.globals)
        self.gc.adopt(self.main_thread)

    def _new_table(self, frames=()):
        roots = frames or self._active_frames or ()
        self.gc.safepoint(roots)
        table = LuaTable()
        self.gc.adopt(table)
        return table

    def _new_cell(self, value=None, frames=()):
        roots = frames or self._active_frames or ()
        self.gc.safepoint(roots)
        cell = Cell(value)
        self.gc.adopt(cell)
        return cell

    def _new_closure(self, proto, upvalues, env, frames=()):
        roots = frames or self._active_frames or ()
        self.gc.safepoint(roots)
        closure = Closure(proto, upvalues, env)
        self.gc.adopt(closure)
        return closure

    @staticmethod
    def _error_value(error):
        return error_value(error)

    def metatable_for(self, value):
        if isinstance(value, LuaTable):
            return value.metatable
        if isinstance(value, bytes):
            return self.type_metatables.get(b"string")
        return None

    def _tm(self, value, name):
        mt = self.metatable_for(value)
        return mt.rawget(name) if isinstance(mt, LuaTable) else None

    def _new_frame(self, closure, args, return_reg, return_want):
        frame = super()._new_frame(closure, args, return_reg, return_want)
        # Top-level compiled chunks bind globals through a lexical _ENV register.
        # A chunk returned by load() enters through the ordinary Closure call path,
        # so initialise that register from the closure's explicit environment just
        # as VM.run() does for the main chunk.
        if frame.proto.env_reg >= 0:
            frame.regs[frame.proto.env_reg] = closure.env
        return frame

    def _invoke(self, frames, parent, fn, args, dest, want, tail=False):
        previous_frames = self._active_frames
        previous_result = self._active_call_result
        visible_frames = frames
        if self._sync_frame_prefixes:
            prefix = self._sync_frame_prefixes[-1]
            if frames is not prefix:
                visible_frames = [*prefix, *frames]
        self._active_frames = visible_frames
        self._active_call_result = (parent, dest, want, tail)
        try:
            return super()._invoke(frames, parent, fn, args, dest, want, tail=tail)
        finally:
            self._active_frames = previous_frames
            self._active_call_result = previous_result

    def _host_values(self, fn, args):
        previous = self._active_host_values
        self._active_host_values = (fn, *args)
        try:
            values = super()._host_values(fn, args)
            for value in values:
                self.gc.adopt(value)
            self.gc.safepoint((*(self._active_frames or ()), *values))
            return values
        finally:
            self._active_host_values = previous

    def run(self, proto: Proto, fuel=None):
        previous_thread = self.current_thread
        previous_frames = self._active_frames
        self.current_thread = self.main_thread
        self.main_thread.status = "running"
        try:
            remaining = self.default_fuel if fuel is None else fuel
            root = Closure(proto, [], self.globals)
            self.gc.adopt(root)
            root_regs = [None] * max(1, proto.register_count)
            if proto.env_reg >= 0:
                root_regs[proto.env_reg] = self.globals
            frames = [Frame(root, root_regs)]
            self._active_frames = frames
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
                    capture_error(exc, frames)
                    frames[-1].pending_error = exc

            self.gc.safepoint(final_values)
            if len(final_values) == 0:
                return None
            if len(final_values) == 1:
                return final_values[0]
            return final_values
        finally:
            self._active_frames = previous_frames
            self.current_thread = previous_thread

    def call_sync(self, fn, args=()):
        """Call a Lua/host callable to completion and return all results.

        This is the continuation boundary used by safe standard-library
        functions that need Lua callbacks. Yielding across this synchronous
        native-library boundary is deliberately rejected for now.
        """
        proto = Proto("<stdlib-callback>", register_count=1, source=None)
        parent_closure = Closure(proto, [], self.globals)
        self.gc.adopt(parent_closure)
        parent = Frame(parent_closure, [None])
        frames = [parent]
        prefix = list(self._active_frames or ())
        self._sync_frame_prefixes.append(prefix)
        try:
            try:
                self._invoke(frames, parent, fn, list(args), 0, -1)
                remaining = self.default_fuel
                while len(frames) > 1:
                    try:
                        frame = frames[-1]
                        if self._drive_pending(frames, frame):
                            continue

                        remaining -= 1
                        if remaining < 0:
                            raise LuaQuotaError("execution quota exceeded")
                        if frame.pc >= len(frame.proto.code):
                            self._return(frames, frame, ())
                            continue

                        ins = frame.proto.code[frame.pc]
                        frame.pc += 1
                        OPCODE_HANDLERS[ins.op](
                            self,
                            frames,
                            frame,
                            ins,
                            frame.regs,
                            frame.proto.constants,
                        )
                    except _YieldSignal:
                        raise LuaRuntimeError(
                            "attempt to yield across a standard library callback"
                        ) from None
                    except LuaQuotaError:
                        raise
                    except LuaRuntimeError as exc:
                        visible = [*prefix, *frames[1:]]
                        if visible:
                            capture_error(exc, visible)
                        if not frames:
                            raise
                        frames[-1].pending_error = exc
            except _YieldSignal:
                raise LuaRuntimeError(
                    "attempt to yield across a standard library callback"
                ) from None

            if parent.pending_error is not None:
                raise parent.pending_error
            result = parent.regs[0]
            if isinstance(result, MultiValue):
                return tuple(result.values)
            return () if result is None else (result,)
        finally:
            self._sync_frame_prefixes.pop()

    def _execute_thread(self, thread, stop_depth: int | None = None):
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
                capture_error(exc, frames)
                thread.error = exc
                thread.status = "dead"
                return "error", (self._error_value(exc),)

        return "return", tuple(final_values)

    def index_sync(self, obj, key):
        """Perform one ordinary Lua indexing operation synchronously."""
        for _ in range(self.MAXTAGLOOP):
            if isinstance(obj, LuaTable):
                if obj.rawhas(key):
                    return obj.rawget(key)
                tm = self._tm(obj, b"__index")
                if tm is None:
                    return None
            else:
                tm = self._tm(obj, b"__index")
                if tm is None:
                    raise LuaRuntimeError("attempt to index a value without __index")

            if isinstance(tm, LuaTable):
                obj = tm
                continue
            values = self.call_sync(tm, (obj, key))
            return values[0] if values else None
        raise LuaRuntimeError("'__index' chain too long; possible loop")

    def less_than_sync(self, left, right) -> bool:
        if type(left) in (int, float) and type(right) in (int, float):
            return left < right
        if isinstance(left, bytes) and isinstance(right, bytes):
            return left < right
        tm = self._first_tm(left, right, b"__lt")
        if tm is None:
            raise LuaRuntimeError("attempt to compare incompatible values")
        values = self.call_sync(tm, (left, right))
        return truthy(values[0] if values else None)
