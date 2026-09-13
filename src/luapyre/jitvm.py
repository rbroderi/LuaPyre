from __future__ import annotations

from .bytecode import Closure, Op, Proto
from .diagnostics import capture_error
from .errors import LuaQuotaError, LuaRuntimeError
from .gcvm import GarbageCollectedVM
from .jit import CompiledLoop, DEOPT, PythonJIT
from .inline_cache import CacheState, InlineCacheFeedback
from .opdispatch import OPCODE_HANDLERS
from .threadvm import _CloseSelfSignal, _YieldSignal
from .trace_jit import TRACE_RETURN, TRACE_SIDE_EXIT, TraceJIT
from .vm import Frame


def _jit_forloop(vm, frames, frame, ins, regs, constants):
    """JFORLOOP handler with a low-overhead per-backedge JIT state cache."""
    if not vm._forloop(regs, ins):
        return
    backedge_pc = frame.pc - 1
    frame.pc = ins.d
    if not vm.jit.enabled:
        return
    key = id(ins)
    state = vm._jit_loop_states.get(key)
    if state is False:
        return
    vm._jit_backedge(frame, backedge_pc, key, state)


_JIT_OPCODE_HANDLERS = dict(OPCODE_HANDLERS)
_JIT_OPCODE_HANDLERS[Op.JFORLOOP] = _jit_forloop


def _jit_gettable(vm, frames, frame, ins, regs, constants):
    obj, key = regs[ins.b], regs[ins.c]
    if not isinstance(obj, vm._ic_table_type) or obj.metatable is not None:
        return vm._gettable(frames, frame, obj, key, ins.a)
    token = vm._ic_hash_key(key)
    if token is None:
        return vm._gettable(frames, frame, obj, key, ins.a)
    site = vm.inline_caches.table_site(frame.proto, frame.pc - 1, "get")
    before = site.state
    entry = site.match((obj, token))
    if entry is not None:
        if entry[2] == obj.version:
            regs[ins.a] = entry[3]
            return None
        site.invalidations += 1
        site.misses += 1
    value = obj.rawget(key)
    regs[ins.a] = value
    site.install((obj, token, obj.version, value), key_size=2)
    vm._note_megamorphic(before, site.state)
    return None


def _jit_settable(vm, frames, frame, ins, regs, constants):
    obj, key, value = regs[ins.a], regs[ins.b], regs[ins.c]
    if not isinstance(obj, vm._ic_table_type) or obj.metatable is not None:
        return vm._settable(frames, frame, obj, key, value)
    token = vm._ic_hash_key(key)
    if token is None:
        return vm._settable(frames, frame, obj, key, value)
    site = vm.inline_caches.table_site(frame.proto, frame.pc - 1, "set")
    before = site.state
    entry = site.match((obj, token))
    if entry is not None:
        obj.rawset(key, value)
        return None
    obj.rawset(key, value)
    site.install((obj, token), key_size=2)
    vm._note_megamorphic(before, site.state)
    return None


_JIT_OPCODE_HANDLERS[Op.GETTABLE] = _jit_gettable
_JIT_OPCODE_HANDLERS[Op.SETTABLE] = _jit_settable


def _jit_cfg_branch(vm, frames, frame, ins, regs, constants):
    source = frame.pc - 1
    result = OPCODE_HANDLERS[ins.op](vm, frames, frame, ins, regs, constants)
    if result is not None or not frame.proto.jit_fully_typed:
        return result
    return vm._trace_transition(frames, frame, source, frame.pc)


class TieredJITVM(GarbageCollectedVM):
    """GarbageCollectedVM with an optional guarded tier-2 Python JIT.

    The existing interpreter remains authoritative. Hot loop regions and hot
    straight-line leaf functions may run as generated Python; every unsupported
    or guard-miss path resumes the ordinary opcode interpreter.

    The source compiler quickens only structurally eligible numeric loops to
    JFORLOOP, so ordinary FORLOOP keeps the exact Tier-0 dispatch path. Once a
    quickened loop is dynamically proven unjittable, its instruction identity
    is negative-cached.
    """

    def __init__(
        self,
        globals=None,
        fuel=1_000_000,
        max_frames=1000,
        *,
        jit_enabled: bool = True,
        jit_threshold: int = 32,
    ):
        super().__init__(globals, fuel=fuel, max_frames=max_frames)
        self.jit = PythonJIT(threshold=jit_threshold, enabled=jit_enabled)
        self._jit_main_fuel = self.default_fuel
        self._jit_main_leaf_allowed = False
        self._jit_loop_states: dict[int, int | bool | CompiledLoop] = {}
        self.inline_caches = InlineCacheFeedback()
        self._call_site_arrays: dict[int, tuple[Proto, list[object | None]]] = {}
        self.trace_jit = TraceJIT(threshold=jit_threshold)
        from .table import LuaTable, _hash_key
        self._ic_table_type = LuaTable
        self._ic_hash_key = _hash_key

    def _note_megamorphic(self, before, after) -> None:
        if before is not CacheState.MEGAMORPHIC and after is CacheState.MEGAMORPHIC:
            self.jit.stats.megamorphic_sites += 1

    def _invoke_site(self, frames, parent, fn, args, dest, want, tail=False):
        pc = parent.pc - 1
        sites = parent.jit_call_sites
        if sites is None:
            proto = parent.proto
            cached_sites = self._call_site_arrays.get(id(proto))
            if cached_sites is None or cached_sites[0] is not proto:
                sites = [None] * len(proto.code)
                self._call_site_arrays[id(proto)] = (proto, sites)
            else:
                sites = cached_sites[1]
            parent.jit_call_sites = sites
        site = sites[pc]
        # Lua closures are distinct runtime objects on every CLOSURE execution,
        # but their executable call shape is the immutable Proto. Keying those
        # entries by Proto prevents repeated runs/recursive instantiations from
        # making an otherwise stable lexical site megamorphic. Host functions
        # retain exact object identity.
        call_target = fn.proto if isinstance(fn, Closure) else fn
        # Keep the stable monomorphic path to one dictionary lookup and one
        # identity guard. Polymorphic probing and transitions stay off this path.
        if (
            site is not None
            and not site.megamorphic
            and len(site.entries) == 1
            and site.entries[0][0] is call_target
        ):
            site.hits += 1
            entry = site.entries[0]
        else:
            if site is None:
                site = self.inline_caches.call_site(parent.proto, pc)
                sites[pc] = site
            entry = site.match((call_target,))
        before = site.state
        if entry is None:
            site.install((call_target, type(fn), None))
            self._note_megamorphic(before, site.state)
        else:
            compiled = entry[2]
            if (
                compiled is not None
                and not tail
                and not self._sync_frame_prefixes
                and self._jit_budget() >= compiled.instruction_cost
            ):
                if len(frames) >= self.max_frames:
                    raise LuaRuntimeError("stack overflow")
                leaf_frame = self._new_frame(fn, list(args), dest, want)
                values = compiled.runner(self, leaf_frame)
                if values is not DEOPT:
                    self._jit_consume(compiled.instruction_cost)
                    self._write_results(parent.regs, dest, want, values)
                    self.jit.stats.leaf_executions += 1
                    return None

        result = self._invoke(frames, parent, fn, args, dest, want, tail=tail)
        if isinstance(fn, Closure) and not tail and not site.megamorphic:
            cached = self.jit._leaf_cache.get(id(fn.proto))
            if cached is not None and cached[0] is fn.proto and cached[1] is not None:
                site.install((call_target, type(fn), cached[1]))
        return result

    def sync_inline_cache_stats(self) -> None:
        """Materialize observability counters off the execution hot path."""
        call_sites = self.inline_caches.calls.values()
        table_sites = self.inline_caches.tables.values()
        self.jit.stats.call_ic_hits = sum(site.hits for site in call_sites)
        self.jit.stats.call_ic_misses = sum(site.misses for site in call_sites)
        self.jit.stats.table_ic_hits = sum(site.hits for site in table_sites)
        self.jit.stats.table_ic_misses = sum(site.misses for site in table_sites)
        self.jit.stats.cache_invalidations = sum(
            site.invalidations for site in table_sites
        )

    def _trace_transition(self, frames, frame, source: int, target: int):
        entries = self.trace_jit.record_edge(frame.proto, source, target)
        if entries < self.trace_jit.threshold:
            return None
        key = (id(frame.proto), target)
        was_cached = key in self.trace_jit.cache
        compiled = self.trace_jit.maybe_trace(frame.proto, target)
        if not was_cached and compiled is not None:
            self.jit.stats.trace_compiles += 1
        if compiled is None:
            return None
        self.jit.stats.osr_entries += 1
        return self._execute_trace(frames, frame, compiled)

    def _execute_trace(self, frames, frame, compiled, depth: int = 0):
        used, status, values = compiled.runner(self, frame, self._jit_budget())
        if used:
            self._jit_consume(used)
            self.jit.stats.trace_executions += 1
        if status == TRACE_SIDE_EXIT:
            self.jit.stats.trace_side_exits += 1
            hot = self.trace_jit.record_side_exit(frame.proto, frame.pc)
            if hot >= self.trace_jit.threshold and depth < 4:
                key = (id(frame.proto), frame.pc)
                was_cached = key in self.trace_jit.cache
                successor = self.trace_jit.maybe_trace(frame.proto, frame.pc)
                if not was_cached and successor is not None:
                    self.jit.stats.trace_compiles += 1
                if successor is not None and successor is not compiled:
                    self.jit.stats.osr_entries += 1
                    return self._execute_trace(
                        frames, frame, successor, depth + 1
                    )
            return None
        if status == TRACE_RETURN:
            return self._return(frames, frame, values)
        return None

    def _jit_budget(self) -> int:
        thread = self.current_thread
        if thread is not None and not thread.is_main:
            return self._coroutine_fuel
        return self._jit_main_fuel

    def _jit_consume(self, amount: int) -> None:
        if amount <= 0:
            return
        thread = self.current_thread
        if thread is not None and not thread.is_main:
            self._coroutine_fuel -= amount
            if self._coroutine_fuel < 0:
                raise LuaQuotaError("execution quota exceeded")
            return
        self._jit_main_fuel -= amount
        if self._jit_main_fuel < 0:
            raise LuaQuotaError("execution quota exceeded")

    def _jit_backedge(self, frame, backedge_pc: int, key: int, state) -> None:
        compiled = state if isinstance(state, CompiledLoop) else None
        if compiled is None:
            hot = (state if type(state) is int else 0) + 1
            if hot < self.jit.threshold:
                self._jit_loop_states[key] = hot
                return
            compiled = self.jit._compile_loop(frame, frame.pc, backedge_pc)
            if compiled is None:
                self._jit_loop_states[key] = False
                self.jit.stats.compile_failures += 1
                return
            self._jit_loop_states[key] = compiled
            self.jit.stats.loop_compiles += 1

        used, progressed = compiled.runner(self, frame, self._jit_budget())
        if used:
            self._jit_consume(used)
            self.jit.stats.loop_executions += 1
            self.jit.stats.loop_iterations += used // compiled.cost_per_iteration
        if (not progressed and used == 0) or frame.pc not in (
            compiled.ir.start_pc,
            compiled.ir.exit_pc,
        ):
            self.jit.stats.deopts += 1
            reason = "entry_guard" if not progressed and used == 0 else "side_exit"
            if self.inline_caches.record_deopt(frame.proto, frame.pc, reason):
                self._jit_loop_states[key] = False
                self.jit.stats.retired_regions += 1

    def _invoke(self, frames, parent, fn, args, dest, want, tail=False):
        # Synchronous stdlib callbacks intentionally stay on the interpreter:
        # their local fuel accounting and visible-frame prefix are specialized
        # for callback semantics. On the main thread leaf compilation is entered
        # only from a direct CALL/CALLV wrapper whose local fuel was just synced.
        thread = self.current_thread
        leaf_budget_is_current = (
            thread is not None and not thread.is_main
        ) or self._jit_main_leaf_allowed
        if (
            isinstance(fn, Closure)
            and not tail
            and not self._sync_frame_prefixes
            and self.jit.enabled
            and leaf_budget_is_current
        ):
            if len(frames) >= self.max_frames:
                raise LuaRuntimeError("stack overflow")
            compiled = self.jit.maybe_leaf(fn.proto)
            if compiled is not None and self._jit_budget() >= compiled.instruction_cost:
                leaf_frame = self._new_frame(fn, list(args), dest, want)
                values = compiled.runner(self, leaf_frame)
                if values is not DEOPT:
                    self._jit_consume(compiled.instruction_cost)
                    self._write_results(parent.regs, dest, want, values)
                    self.jit.stats.leaf_executions += 1
                    return None
                self.jit.stats.deopts += 1
        return super()._invoke(frames, parent, fn, args, dest, want, tail=tail)

    def run(self, proto: Proto, fuel=None):
        previous_thread = self.current_thread
        self.current_thread = self.main_thread
        self.main_thread.status = "running"
        remaining = self.default_fuel if fuel is None else fuel
        self._jit_main_fuel = remaining
        previous_frames = self._active_frames
        try:
            root = Closure(proto, [], self.globals)
            self.gc.adopt(root)
            root_regs = [None] * max(1, proto.register_count)
            if proto.env_reg >= 0:
                root_regs[proto.env_reg] = self.globals
            frames = [Frame(root, root_regs)]
            self._active_frames = frames
            final_values = ()

            # Only these three handlers need to see/mutate the JIT-visible main
            # budget. Wrapping them once keeps the hot dispatch loop identical
            # to Tier 0: one opcode lookup and one handler call, with no per-op
            # membership test or fuel helper call.
            handlers = dict(_JIT_OPCODE_HANDLERS)

            def synced(handler, *, leaf_allowed=False):
                def wrapped(vm, active_frames, frame, ins, regs, constants):
                    nonlocal remaining
                    self._jit_main_fuel = remaining
                    self._jit_main_leaf_allowed = leaf_allowed
                    try:
                        return handler(
                            vm, active_frames, frame, ins, regs, constants
                        )
                    finally:
                        remaining = self._jit_main_fuel
                        self._jit_main_leaf_allowed = False
                return wrapped

            handlers[Op.CALL] = synced(OPCODE_HANDLERS[Op.CALL], leaf_allowed=True)
            handlers[Op.CALLV] = synced(OPCODE_HANDLERS[Op.CALLV], leaf_allowed=True)
            handlers[Op.JFORLOOP] = synced(_jit_forloop)
            if proto.jit_fully_typed:
                for branch_op in (Op.JMP, Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
                    handlers[branch_op] = synced(_jit_cfg_branch)

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
            self._jit_main_leaf_allowed = False
            self._active_frames = previous_frames
            self.current_thread = previous_thread

    def _execute_thread(self, thread, stop_depth: int | None = None):
        frames = thread.frames
        final_values = ()
        handlers = _JIT_OPCODE_HANDLERS

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
                if not frames or not any(frame.protected_name for frame in frames):
                    thread.error = exc
                    thread.status = "dead"
                    return "error", (self._error_value(exc),)
                frames[-1].pending_error = exc

        return "return", tuple(final_values)
