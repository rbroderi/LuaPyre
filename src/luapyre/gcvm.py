from __future__ import annotations

from .gc import LuaGC
from .table import LuaTable
from .threadvm import CoroutineVM


class GarbageCollectedVM(CoroutineVM):
    """Coroutine VM with Lua-level GC root tracking.

    Host calls are the safe points where ``collectgarbage`` can run. Keeping the
    active frame list visible while a host function executes lets the collector
    distinguish live Lua stack values from objects referenced only by weak
    tables or the finalization list without changing the hot opcode loop.
    """

    def __init__(self, globals: LuaTable | None = None, fuel=1_000_000, max_frames=1000):
        super().__init__(globals, fuel=fuel, max_frames=max_frames)
        self._active_frames = None
        self.gc = LuaGC(self)

    def _invoke(self, frames, parent, fn, args, dest, want, tail=False):
        previous = self._active_frames
        self._active_frames = frames
        try:
            return super()._invoke(frames, parent, fn, args, dest, want, tail=tail)
        finally:
            self._active_frames = previous
