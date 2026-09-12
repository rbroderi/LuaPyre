from __future__ import annotations

from .bytecode import Closure, Proto
from .errors import LuaQuotaError, LuaRuntimeError
from .gc import LuaGC
from .opdispatch import OPCODE_HANDLERS
from .table import LuaTable
from .threadvm import CoroutineVM, _YieldSignal
from .values import MultiValue, truthy
from .vm import Frame


class GarbageCollectedVM(CoroutineVM):
    """Coroutine VM with Lua-level GC root tracking.

    Host calls are the safe points where ``collectgarbage`` can run. Keeping the
    active frame list and actual host-call arguments visible while a host
    function executes lets the collector trace semantic roots without changing
    the hot opcode loop.

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
            return super()._host_values(fn, args)
        finally:
            self._active_host_values = previous

    def call_sync(self, fn, args=()):
        """Call a Lua/host callable to completion and return all results.

        This is the continuation boundary used by safe standard-library
        functions that need Lua callbacks.  Yielding across this synchronous
        native-library boundary is deliberately rejected for now; persistent
        coroutine continuations for native-library callbacks remain a separate
        compatibility concern.
        """
        proto = Proto("<stdlib-callback>", register_count=1)
        parent = Frame(Closure(proto, [], self.globals), [None])
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
