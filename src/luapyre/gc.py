from __future__ import annotations

from dataclasses import dataclass
import gc as python_gc
import sys
import warnings

from .bytecode import Cell, Closure, Op, Proto
from .errors import LuaQuotaError, LuaRaisedError, LuaRuntimeError
from .opdispatch import OPCODE_HANDLERS
from .table import LuaTable
from .values import MultiValue
from .vm import Frame


_COLLECTABLE_TYPES = (LuaTable, Closure)
_VALID_WEAK_MODES = {b"k", b"v", b"kv"}

_BINARY_OPS = {
    Op.ADD, Op.ADD_I, Op.ADD_F,
    Op.SUB, Op.SUB_I, Op.SUB_F,
    Op.MUL, Op.MUL_I, Op.MUL_F,
    Op.DIV, Op.IDIV, Op.MOD, Op.POW,
    Op.BAND, Op.BOR, Op.BXOR, Op.SHL, Op.SHR,
    Op.CONCAT, Op.EQ, Op.LT, Op.LE,
}
_UNARY_OPS = {Op.LEN, Op.BNOT, Op.NEG, Op.NOT, Op.TOBOOL}
_CONDITIONAL_JUMPS = {Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL}
_CALL_OPS = {Op.CALL, Op.CALLV, Op.TAILCALL, Op.TAILCALLV}
_TERMINATORS = {Op.RETURN, Op.RETURNV, Op.HALT, Op.TAILCALL, Op.TAILCALLV}


def _rw_empty(_proto: Proto, _ins) -> tuple[set[int], set[int]]:
    return set(), set()


def _rw_write_a(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return set(), {ins.a}


def _rw_read_a(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.a}, set()


def _rw_read_b(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.b}, set()


def _rw_read_b_write_a(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.b}, {ins.a}


def _rw_setcell(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.a, ins.b}, {ins.a}


def _rw_closure(proto: Proto, ins) -> tuple[set[int], set[int]]:
    reads = {
        desc.index
        for desc in proto.children[ins.b].upvalues
        if desc.kind == "local"
    }
    return reads, {ins.a}


def _rw_gettable(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.b, ins.c}, {ins.a}


def _rw_settable(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.a, ins.b, ins.c}, set()


def _rw_setlistv(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.a, ins.c}, set()


def _rw_forprep(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    regs = {ins.a, ins.b, ins.c}
    return set(regs), regs


def _rw_forloop(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.a, ins.b, ins.c}, {ins.a}


def _rw_pforprep(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    regs = {ins.a, ins.a + 1, ins.a + 2}
    return set(regs), regs


def _rw_pforloop(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.a, ins.a + 1, ins.a + 2}, {ins.a, ins.a + 2}


def _rw_ptforprep(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    regs = {ins.a + 2, ins.a + 3}
    return set(regs), regs


def _rw_ptforloop(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.a + 3}, set()


def _rw_call(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    reads = {ins.b, *range(ins.c, ins.c + max(0, ins.d))}
    if ins.e == 0:
        writes = set()
    elif ins.e == -1:
        writes = {ins.a}
    else:
        writes = set(range(ins.a, ins.a + max(0, ins.e)))
    return reads, writes


def _rw_callv(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    reads = {ins.b, ins.e, *range(ins.c, ins.c + max(0, ins.d))}
    return reads, {ins.a}


def _rw_tailcall(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.b, *range(ins.c, ins.c + max(0, ins.d))}, set()


def _rw_tailcallv(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.b, ins.e, *range(ins.c, ins.c + max(0, ins.d))}, set()


def _rw_vararg(proto: Proto, ins) -> tuple[set[int], set[int]]:
    reads = {proto.vararg_name_reg} if proto.vararg_name_reg >= 0 else set()
    writes = {ins.a} if ins.b == -1 else set(range(ins.a, ins.a + max(0, ins.b)))
    return reads, writes


def _rw_pvararg(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    reads = {ins.c} if ins.c >= 0 else set()
    writes = {ins.a} if ins.b == -1 else set(range(ins.a, ins.a + max(0, ins.b)))
    return reads, writes


def _rw_pgetvarg(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.b}, {ins.a}


def _rw_unpack(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.b}, set(range(ins.a, ins.a + max(0, ins.c)))


def _rw_return(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return set(range(ins.a, ins.a + max(0, ins.b))), set()


def _rw_returnv(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    reads = set(range(ins.a, ins.a + max(0, ins.b)))
    reads.add(ins.c)
    return reads, set()


_RW_HANDLERS = {op: _rw_empty for op in Op}
_RW_HANDLERS.update({
    Op.LOADK: _rw_write_a,
    Op.MOVE: _rw_read_b_write_a,
    Op.LOCAL: _rw_read_b_write_a,
    Op.GETGLOBAL: _rw_write_a,
    Op.SETGLOBAL: _rw_read_a,
    Op.GETUPVAL: _rw_write_a,
    Op.SETUPVAL: _rw_read_b,
    Op.GETCELL: _rw_read_b_write_a,
    Op.SETCELL: _rw_setcell,
    Op.CLOSURE: _rw_closure,
    Op.NEWTABLE: _rw_write_a,
    Op.GETTABLE: _rw_gettable,
    Op.SETTABLE: _rw_settable,
    Op.SETLISTV: _rw_setlistv,
    Op.FORPREP: _rw_forprep,
    Op.FORLOOP: _rw_forloop,
    Op.PFORPREP: _rw_pforprep,
    Op.PFORLOOP: _rw_pforloop,
    Op.PTFORPREP: _rw_ptforprep,
    Op.PTFORLOOP: _rw_ptforloop,
    Op.CALL: _rw_call,
    Op.CALLV: _rw_callv,
    Op.TAILCALL: _rw_tailcall,
    Op.TAILCALLV: _rw_tailcallv,
    Op.VARARG: _rw_vararg,
    Op.PVARARG: _rw_pvararg,
    Op.PGETVARG: _rw_pgetvarg,
    Op.UNPACK: _rw_unpack,
    Op.TBC: _rw_read_a,
    Op.PTBC: _rw_read_a,
    Op.CHECKNIL: _rw_read_a,
    Op.RETURN: _rw_return,
    Op.RETURNV: _rw_returnv,
    Op.GUARD: _rw_read_a,
})
for _op in _BINARY_OPS:
    _RW_HANDLERS[_op] = _rw_gettable
for _op in _UNARY_OPS:
    _RW_HANDLERS[_op] = _rw_read_b_write_a
for _op in _CONDITIONAL_JUMPS:
    _RW_HANDLERS[_op] = _rw_read_b


def _succ_fallthrough(code, index: int, _ins) -> tuple[int, ...]:
    return (index + 1,) if index + 1 < len(code) else ()


def _succ_terminate(_code, _index: int, _ins) -> tuple[int, ...]:
    return ()


def _succ_jump(code, _index: int, ins) -> tuple[int, ...]:
    return (ins.a,) if 0 <= ins.a < len(code) else ()


def _succ_conditional(code, index: int, ins) -> tuple[int, ...]:
    out = []
    if index + 1 < len(code):
        out.append(index + 1)
    if 0 <= ins.a < len(code):
        out.append(ins.a)
    return tuple(out)


def _succ_loop(code, index: int, ins) -> tuple[int, ...]:
    out = []
    if index + 1 < len(code):
        out.append(index + 1)
    if 0 <= ins.d < len(code):
        out.append(ins.d)
    return tuple(out)


def _succ_ptforprep(code, _index: int, ins) -> tuple[int, ...]:
    return (ins.d,) if 0 <= ins.d < len(code) else ()


_SUCCESSOR_HANDLERS = {op: _succ_fallthrough for op in Op}
for _op in _TERMINATORS:
    _SUCCESSOR_HANDLERS[_op] = _succ_terminate
_SUCCESSOR_HANDLERS[Op.JMP] = _succ_jump
for _op in _CONDITIONAL_JUMPS:
    _SUCCESSOR_HANDLERS[_op] = _succ_conditional
for _op in (Op.FORPREP, Op.FORLOOP, Op.PFORPREP, Op.PFORLOOP, Op.PTFORLOOP):
    _SUCCESSOR_HANDLERS[_op] = _succ_loop
_SUCCESSOR_HANDLERS[Op.PTFORPREP] = _succ_ptforprep

if set(_RW_HANDLERS) != set(Op) or set(_SUCCESSOR_HANDLERS) != set(Op):
    raise RuntimeError("GC opcode analysis dispatch table mismatch")


@dataclass(slots=True)
class GCStats:
    cycles: int = 0
    reachable_objects: int = 0
    approximate_bytes: int = 0
    weak_entries_cleared: int = 0
    finalized_objects: int = 0


class LuaGC:
    """Lua-level tracing collector for GC-observable semantics.

    Python still owns physical memory. This collector models the part Lua code
    can observe: weak tables/ephemerons, table finalization, and the basic
    ``collectgarbage`` control surface. Roots are semantic Lua values, not every
    physical register slot, so stale temporaries cannot accidentally keep weak
    keys/values alive.
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
        # has run. Strong Python references here model that internal Lua list;
        # they are intentionally *not* ordinary tracing roots.
        self._finalizable: list[LuaTable] = []
        self._finalizable_ids: set[int] = set()
        self._finalized_ids: set[int] = set()
        self._liveness_cache: dict[int, tuple[Proto, tuple[frozenset[int], ...]]] = {}

    @staticmethod
    def _is_thread(value) -> bool:
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
        """Mark a table when its newly assigned metatable already has ``__gc``."""
        if not isinstance(table, LuaTable) or not isinstance(metatable, LuaTable):
            return
        if metatable.rawget(b"__gc") is None:
            return
        ident = id(table)
        if ident in self._finalizable_ids:
            return
        # Explicitly setting a __gc metatable after a previous finalization is
        # how Lua code can mark a resurrected object for finalization again.
        self._finalized_ids.discard(ident)
        self._finalizable_ids.add(ident)
        self._finalizable.append(table)

    @staticmethod
    def _call_result_regs(dest: int, want: int) -> set[int]:
        if want == 0:
            return set()
        if want == -1:
            return {dest}
        return set(range(dest, dest + max(0, want)))

    @staticmethod
    def _ins_reads_writes(proto: Proto, ins) -> tuple[set[int], set[int]]:
        return _RW_HANDLERS[ins.op](proto, ins)

    @staticmethod
    def _successors(code, index: int) -> tuple[int, ...]:
        ins = code[index]
        return _SUCCESSOR_HANDLERS[ins.op](code, index, ins)

    def _live_sets(self, proto: Proto) -> tuple[frozenset[int], ...]:
        ident = id(proto)
        cached = self._liveness_cache.get(ident)
        if cached is not None and cached[0] is proto:
            return cached[1]

        code = proto.code
        n = len(code)
        reads_writes = [self._ins_reads_writes(proto, ins) for ins in code]
        successors = [self._successors(code, i) for i in range(n)]
        live_in = [set() for _ in range(n + 1)]

        changed = True
        while changed:
            changed = False
            for i in range(n - 1, -1, -1):
                reads, writes = reads_writes[i]
                live_out: set[int] = set()
                for succ in successors[i]:
                    live_out.update(live_in[succ])
                new_live = reads | (live_out - writes)
                if new_live != live_in[i]:
                    live_in[i] = new_live
                    changed = True

        result = tuple(frozenset(values) for values in live_in)
        self._liveness_cache[ident] = (proto, result)
        return result

    def _live_regs(self, frame: Frame) -> frozenset[int]:
        live = self._live_sets(frame.proto)
        pc = max(0, min(frame.pc, len(live) - 1))
        return live[pc]

    def _trace(self, extra_roots=(), *, physical_regs: bool = False):
        marked: dict[int, object] = {}
        weak_tables: dict[int, tuple[LuaTable, bytes]] = {}
        ephemerons: list[tuple[object, object]] = []
        seen_frames: set[int] = set()

        def mark_frame(frame: Frame, excluded: set[int] | None = None) -> None:
            ident = id(frame)
            if ident in seen_frames:
                return
            seen_frames.add(ident)
            excluded = excluded or set()

            mark(frame.closure)
            regs_to_scan = range(len(frame.regs)) if physical_regs else self._live_regs(frame)
            for reg in regs_to_scan:
                if reg in excluded or reg < 0 or reg >= len(frame.regs):
                    continue
                cell = frame.cells.get(reg)
                mark(cell.value if cell is not None else frame.regs[reg])

            for value in frame.varargs:
                mark(value)
            for value in frame.close_stack:
                mark(value)
            for _reg, value in frame.puc_close_stack:
                mark(value)
            if isinstance(frame.pending_error, LuaRaisedError):
                mark(frame.pending_error.value)

        def mark_frame_stack(frames, active: bool = False) -> None:
            for index, frame in enumerate(frames):
                excluded: set[int] = set()
                if index + 1 < len(frames):
                    child = frames[index + 1]
                    if child.return_reg >= 0:
                        excluded.update(self._call_result_regs(child.return_reg, child.return_want))

                if active:
                    active_call = getattr(self.vm, "_active_call_result", None)
                    if active_call is not None and active_call[0] is frame:
                        _parent, dest, want, tail = active_call
                        if tail:
                            excluded.update(range(len(frame.regs)))
                        else:
                            excluded.update(self._call_result_regs(dest, want))
                mark_frame(frame, excluded)

        def mark(value) -> bool:
            if isinstance(value, MultiValue):
                changed = False
                for item in value.values:
                    changed = mark(item) or changed
                return changed
            if isinstance(value, Cell):
                return mark(value.value)
            if isinstance(value, Frame):
                mark_frame(value)
                return False
            if isinstance(value, LuaRaisedError):
                return mark(value.value)

            if not self._is_collectable(value):
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
                            mark(item)
                return True

            if isinstance(value, Closure):
                mark(value.env)
                for cell in value.upvalues:
                    mark(cell.value)
                return True

            # LuaThread
            mark(value.entry)
            mark_frame_stack(value.frames)
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

        active_frames = getattr(self.vm, "_active_frames", None)
        if active_frames:
            mark_frame_stack(active_frames, active=True)

        active_host_values = getattr(self.vm, "_active_host_values", None)
        if active_host_values:
            for value in active_host_values:
                mark(value)

        current = getattr(self.vm, "current_thread", None)
        if current is not None:
            mark(current)
        mark(self.vm.globals)
        main = getattr(self.vm, "main_thread", None)
        if main is not None and main is not current:
            mark(main)
        for value in extra_roots:
            mark(value)

        # Ephemeron convergence must reach a fixed point because marking a
        # value can make a key in another ephemeron reachable.
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
        keys: bool,
        values: bool,
        dead_value_ids: set[int] | None = None,
    ) -> int:
        marked_ids = set(marked)
        dead_value_ids = dead_value_ids or set()
        cleared = 0

        for table, mode in weak_tables.values():
            doomed = []
            for key, value in list(table.items()):
                dead_key = (
                    keys
                    and b"k" in mode
                    and self._is_collectable(key)
                    and id(key) not in marked_ids
                )
                dead_value = (
                    values
                    and b"v" in mode
                    and self._is_collectable(value)
                    and (id(value) not in marked_ids or id(value) in dead_value_ids)
                )
                if dead_key or dead_value:
                    doomed.append(key)
            for key in doomed:
                table.rawset(key, None)
                cleared += 1
        return cleared

    def _run_lua_callback(self, fn, args) -> None:
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
        previous_host_values = getattr(self.vm, "_active_host_values", None)
        previous_call_result = getattr(self.vm, "_active_call_result", None)
        self.in_finalizer = True
        try:
            if hasattr(self.vm, "main_thread"):
                self.vm.current_thread = self.vm.main_thread
            self.vm._active_frames = None
            self.vm._active_host_values = None
            self.vm._active_call_result = None
            self._run_lua_callback(fn, (table,))
        except LuaRuntimeError as exc:
            warnings.warn(
                f"error in __gc metamethod: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        finally:
            self.vm._active_frames = previous_frames
            self.vm._active_host_values = previous_host_values
            self.vm._active_call_result = previous_call_result
            if hasattr(self.vm, "current_thread"):
                self.vm.current_thread = previous_thread
            self.in_finalizer = False

    def collect(self) -> None:
        if self.in_finalizer:
            raise LuaRuntimeError("cannot run garbage collector from a finalizer")

        # Phase 1 mirrors Lua's atomic mark phase: find the strongly reachable
        # graph and clear weak values before objects are moved to finalization.
        marked, weak_tables = self._trace()
        cleared = self._clear_weak_tables(
            weak_tables,
            marked,
            keys=False,
            values=True,
        )

        dead = [table for table in self._finalizable if id(table) not in marked]
        dead_ids = {id(table) for table in dead}

        # Phase 2 resurrects objects selected for finalization and everything
        # reachable through them. This can discover weak tables that themselves
        # were unreachable in phase 1 (e.g. captured only by a __gc closure).
        resurrected_marked, resurrected_weak_tables = self._trace(dead)
        resurrected_only = set(resurrected_marked) - set(marked)

        # Lua treats the resurrected graph asymmetrically: it is alive for weak
        # keys, so finalizers can still look up associated metadata, but remains
        # dead for weak values during this collection cycle.
        cleared += self._clear_weak_tables(
            resurrected_weak_tables,
            resurrected_marked,
            keys=True,
            values=True,
            dead_value_ids=resurrected_only,
        )

        if dead_ids:
            self._finalizable = [t for t in self._finalizable if id(t) not in dead_ids]
            self._finalizable_ids.difference_update(dead_ids)
            self._finalized_ids.update(dead_ids)

        finalized = 0
        for table in reversed(dead):
            self._finalize(table)
            finalized += 1

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
        # Approximate allocator-visible memory, so include physical register
        # contents even when bytecode liveness says a slot is semantically dead.
        # Real collection continues to use semantic liveness in _trace().
        marked, _ = self._trace(physical_regs=True)
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
            # Explicit stepping is deterministic in 0.6: a step completes one
            # full observable cycle. Automatic incremental pacing is future work.
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