from __future__ import annotations

from dataclasses import dataclass
import gc as python_gc
import sys
import warnings

from .bytecode import Cell, Closure, Proto
from .errors import LuaQuotaError, LuaRaisedError, LuaRuntimeError
from .opdispatch import OPCODE_HANDLERS
from .table import LuaTable
from .values import MultiValue
from .vm import Frame, HostFunction


_COLLECTABLE_TYPES = (LuaTable, Closure)
_VALID_WEAK_MODES = {b"k", b"v", b"kv"}


@dataclass(slots=True)
class GCStats:
    cycles: int = 0
    reachable_objects: int = 0
    approximate_bytes: int = 0
    weak_entries_cleared: int = 0
    finalized_objects: int = 0


class LuaGC:
    """Lua-level tracing collector for GC-observable semantics.

    Python still owns the physical memory. This collector models the part Lua
    programs can observe: weak tables/ephemerons, table finalization, and the
    basic ``collectgarbage`` control surface. It deliberately traces from Lua
    roots instead of using Python reference counts, because VM registers and
    table storage are implementation details and do not define Lua reachability.
    """

    DEFAULT_PARAMS = {
        b"minormul": 20,
        b"majorminor": 70,
        b"minormajor": 50,
        b"pause": 200,
        b"stepmul": 100,
        b"stepsize": 13,
    }

    def __init__(self, vm):
        self.vm = vm
        self.running = True
        self.mode = b"incremental"
        self.params = dict(self.DEFAULT_PARAMS)
        self.stats = GCStats()
        self.in_finalizer = False

        # Lua keeps marked-for-finalization objects alive until their finalizer
        # has run. Keeping strong Python references here models that list.
        self._finalizable: list[LuaTable] = []
        self._finalizable_ids: set[int] = set()
        self._finalized_ids: set[int] = set()

    @staticmethod
    def _is_thread(value) -> bool:
        # Import lazily so threadvm can subclass the base VM without a module
        # import cycle through gc.py.
        from .threadvm import LuaThread

        return isinstance(value, LuaThread)

    @classmethod
    def _is_collectable(cls, value) -> bool:
        return isinstance(value, _COLLECTABLE_TYPES) or cls._is_thread(value)

    @staticmethod
    def _weak_mode(table: LuaTable) -> bytes | None:
        mt = table.metatable
        if not isinstance(mt, LuaTable):
            return None
        mode = mt.rawget(b"__mode")
        return mode if isinstance(mode, bytes) and mode in _VALID_WEAK_MODES else None

    def mark_finalizable(self, table: LuaTable, metatable: LuaTable | None) -> None:
        """Mark a table when a metatable already containing ``__gc`` is set."""
        if not isinstance(table, LuaTable) or not isinstance(metatable, LuaTable):
            return
        if metatable.rawget(b"__gc") is None:
            return
        ident = id(table)
        if ident in self._finalizable_ids or ident in self._finalized_ids:
            return
        self._finalizable_ids.add(ident)
        self._finalizable.append(table)

    def _roots(self):
        roots = [self.vm.globals]
        frames = getattr(self.vm, "_active_frames", None)
        if frames:
            roots.extend(frames)

        current = getattr(self.vm, "current_thread", None)
        if current is not None:
            roots.append(current)
        main = getattr(self.vm, "main_thread", None)
        if main is not None and main is not current:
            roots.append(main)
        return roots

    @staticmethod
    def _iter_frame_values(frame: Frame):
        yield frame.closure
        yield from frame.regs
        for cell in frame.cells.values():
            yield cell.value
        yield from frame.close_stack
        if isinstance(frame.pending_error, LuaRaisedError):
            yield frame.pending_error.value

    def _trace(self):
        marked: dict[int, object] = {}
        weak_tables: dict[int, tuple[LuaTable, bytes]] = {}
        ephemerons: list[tuple[object, object]] = []

        def mark(value) -> bool:
            if isinstance(value, MultiValue):
                changed = False
                for item in value.values:
                    changed = mark(item) or changed
                return changed
            if isinstance(value, Cell):
                return mark(value.value)
            if isinstance(value, Frame):
                changed = False
                for item in self._iter_frame_values(value):
                    changed = mark(item) or changed
                return changed
            if isinstance(value, LuaRaisedError):
                return mark(value.value)

            collectable = self._is_collectable(value)
            if not collectable:
                return False
            ident = id(value)
            if ident in marked:
                return False
            marked[ident] = value

            if isinstance(value, LuaTable):
                mark(value.metatable)
                mode = self._weak_mode(value)
                if mode is not None:
                    weak_tables[ident] = (value, mode)

                if mode is None:
                    for key, item in value.items():
                        mark(key)
                        mark(item)
                elif mode == b"v":
                    for key, _item in value.items():
                        mark(key)
                elif mode == b"k":
                    for key, item in value.items():
                        if self._is_collectable(key):
                            ephemerons.append((key, item))
                        else:
                            # Numeric/string/etc. keys cannot disappear, so
                            # their values are ordinary strong references.
                            mark(item)
                else:  # b"kv"
                    pass
                return True

            if isinstance(value, Closure):
                mark(value.env)
                for cell in value.upvalues:
                    mark(cell.value)
                return True

            # LuaThread is imported lazily above. Its persistent frames are
            # part of the thread object graph and therefore survive suspension.
            mark(value.entry)
            for frame in value.frames:
                mark(frame)
            for item in value.yielded:
                mark(item)
            if value.yield_target is not None:
                mark(value.yield_target[0])
            if value.pending_tail_resume is not None:
                for item in value.pending_tail_resume:
                    mark(item)
            if isinstance(value.error, LuaRaisedError):
                mark(value.error.value)
            return True

        for root in self._roots():
            mark(root)

        # Ephemeron convergence: a value becomes reachable only after its key
        # is reachable. Marking that value can in turn make keys in other
        # ephemerons reachable, hence the fixed-point loop.
        while True:
            before = len(marked)
            index = 0
            while index < len(ephemerons):
                key, item = ephemerons[index]
                index += 1
                if not self._is_collectable(key) or id(key) in marked:
                    mark(item)
            if len(marked) == before:
                break

        return marked, weak_tables

    def _clear_weak_tables(
        self,
        weak_tables: dict[int, tuple[LuaTable, bytes]],
        marked: dict[int, object],
        *,
        preserve_weak_keys: set[int] | None = None,
    ) -> int:
        preserve_weak_keys = preserve_weak_keys or set()
        marked_ids = set(marked)
        cleared = 0

        for table, mode in weak_tables.values():
            doomed = []
            for key, value in list(table.items()):
                dead_key = (
                    b"k" in mode
                    and self._is_collectable(key)
                    and id(key) not in marked_ids
                    and id(key) not in preserve_weak_keys
                )
                dead_value = (
                    b"v" in mode
                    and self._is_collectable(value)
                    and id(value) not in marked_ids
                )
                if dead_key or dead_value:
                    doomed.append(key)
            for key in doomed:
                table.rawset(key, None)
                cleared += 1
        return cleared

    def _run_lua_callback(self, fn, args) -> None:
        """Run one non-yieldable Lua callback synchronously on shared handlers."""
        # A synthetic caller lets _invoke support closures, host functions, and
        # callable tables without adding a second call protocol.
        proto = Proto("<gc-finalizer>", register_count=1)
        parent = Frame(Closure(proto, [], self.vm.globals), [None])
        frames = [parent]
        remaining = self.vm.default_fuel
        handlers = OPCODE_HANDLERS

        self.vm._invoke(frames, parent, fn, list(args), 0, 0)
        while len(frames) > 1:
            try:
                frame = frames[-1]
                if self.vm._drive_pending(frames, frame):
                    continue
                remaining -= 1
                if remaining < 0:
                    raise LuaQuotaError("execution quota exceeded in finalizer")
                if frame.pc >= len(frame.proto.code):
                    self.vm._return(frames, frame, ())
                    continue
                ins = frame.proto.code[frame.pc]
                frame.pc += 1
                handlers[ins.op](
                    self.vm,
                    frames,
                    frame,
                    ins,
                    frame.regs,
                    frame.proto.constants,
                )
            except LuaQuotaError:
                raise
            except LuaRuntimeError as exc:
                if len(frames) <= 1:
                    raise
                frames[-1].pending_error = exc

        if parent.pending_error is not None:
            raise parent.pending_error

    def _finalize(self, table: LuaTable) -> None:
        fn = table.metatable.rawget(b"__gc") if isinstance(table.metatable, LuaTable) else None
        if fn is None:
            return

        previous_thread = getattr(self.vm, "current_thread", None)
        previous_frames = getattr(self.vm, "_active_frames", None)
        self.in_finalizer = True
        try:
            # Finalizers are non-yieldable. Running them as main-thread work
            # makes coroutine.yield reject the operation through normal VM rules.
            if hasattr(self.vm, "main_thread"):
                self.vm.current_thread = self.vm.main_thread
            self.vm._active_frames = None
            self._run_lua_callback(fn, (table,))
        except LuaRuntimeError as exc:
            warnings.warn(
                f"error in __gc metamethod: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        finally:
            self.vm._active_frames = previous_frames
            if hasattr(self.vm, "current_thread"):
                self.vm.current_thread = previous_thread
            self.in_finalizer = False

    def collect(self) -> None:
        if self.in_finalizer:
            raise LuaRuntimeError("cannot run garbage collector from a finalizer")

        marked, weak_tables = self._trace()
        dead = [table for table in self._finalizable if id(table) not in marked]
        resurrected_for_cycle = {id(table) for table in dead}

        # Weak values disappear before finalizers. Weak keys for objects being
        # finalized survive this cycle, matching Lua's resurrection rule.
        cleared = self._clear_weak_tables(
            weak_tables,
            marked,
            preserve_weak_keys=resurrected_for_cycle,
        )

        dead_ids = resurrected_for_cycle
        if dead_ids:
            self._finalizable = [t for t in self._finalizable if id(t) not in dead_ids]
            self._finalizable_ids.difference_update(dead_ids)
            self._finalized_ids.update(dead_ids)

        finalized = 0
        for table in reversed(dead):
            self._finalize(table)
            finalized += 1

        # Drop the cycle's temporary resurrection references, then ask Python
        # to reclaim implementation objects no longer referenced anywhere.
        dead.clear()
        python_gc.collect()

        final_marked, _ = self._trace()
        self.stats.cycles += 1
        self.stats.reachable_objects = len(final_marked)
        self.stats.approximate_bytes = self._estimate_bytes(final_marked.values())
        self.stats.weak_entries_cleared += cleared
        self.stats.finalized_objects += finalized

    @staticmethod
    def _estimate_bytes(values) -> int:
        total = 0
        seen: set[int] = set()
        for value in values:
            ident = id(value)
            if ident in seen:
                continue
            seen.add(ident)
            try:
                total += sys.getsizeof(value)
                if isinstance(value, LuaTable):
                    total += sys.getsizeof(value.array) + sys.getsizeof(value.hash)
                elif isinstance(value, Closure):
                    total += sys.getsizeof(value.upvalues)
            except TypeError:
                pass
        return total

    def count_kbytes(self) -> float:
        marked, _ = self._trace()
        return self._estimate_bytes(marked.values()) / 1024.0

    @staticmethod
    def _as_bytes(value, what: str) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("ascii", "strict")
        raise LuaRuntimeError(f"bad argument to 'collectgarbage' ({what} expected)")

    def command(self, option=b"collect", *args):
        option = self._as_bytes(option, "string")

        if option == b"collect":
            self.collect()
            return None
        if option == b"stop":
            self.running = False
            return None
        if option == b"restart":
            self.running = True
            return None
        if option == b"isrunning":
            return self.running
        if option == b"count":
            return self.count_kbytes()
        if option == b"step":
            # LuaPyre currently performs full deterministic cycles at explicit
            # safe points. A full cycle is a valid (completed) GC step.
            self.collect()
            return True
        if option in (b"incremental", b"generational"):
            previous = self.mode
            self.mode = option
            return previous
        if option == b"param":
            if not args:
                raise LuaRuntimeError("bad argument #2 to 'collectgarbage' (parameter expected)")
            name = self._as_bytes(args[0], "string")
            if name not in self.params:
                raise LuaRuntimeError("invalid garbage-collector parameter")
            previous = self.params[name]
            if len(args) > 1:
                value = args[1]
                if type(value) is not int or not 0 <= value <= 100000:
                    raise LuaRuntimeError("garbage-collector parameter out of range")
                self.params[name] = value
            return previous

        raise LuaRuntimeError("invalid option to 'collectgarbage'")
