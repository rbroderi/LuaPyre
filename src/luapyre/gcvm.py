from __future__ import annotations

from .gc import LuaGC
from .table import LuaTable
from .threadvm import CoroutineVM


class GarbageCollectedVM(CoroutineVM):
    """Coroutine VM with Lua-level GC root tracking.

    Host calls are the safe points where ``collectgarbage`` can run. Keeping the
    active frame list and actual host-call arguments visible while a host
    function executes lets the collector trace semantic roots without changing
    the hot opcode loop.
    """

    def __init__(self, globals: LuaTable | None = None, fuel=1_000_000, max_frames=1000):
        super().__init__(globals, fuel=fuel, max_frames=max_frames)
        self._active_frames = None
        self._active_host_values = None
        self._active_call_result = None
        self.gc = LuaGC(self)

    def _invoke(self, frames, parent, fn, args, dest, want, tail=False):
        previous_frames = self._active_frames
        previous_result = self._active_call_result
        self._active_frames = frames
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
