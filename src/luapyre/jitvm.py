from __future__ import annotations

from .bytecode import Closure, Op, Proto
from .diagnostics import capture_error
from .errors import LuaQuotaError, LuaRuntimeError
from .gcvm import GarbageCollectedVM
from .jit import CompiledLoop, DEOPT, PythonJIT
from .opdispatch import OPCODE_HANDLERS
from .threadvm import _CloseSelfSignal, _YieldSignal
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
                values = compiled.runner(leaf_frame)
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
        try:
            root = Closure(proto, [], self.globals)
            root_regs = [None] * max(1, proto.register_count)
            if proto.env_reg >= 0:
                root_regs[proto.env_reg] = self.globals
            frames = [Frame(root, root_regs)]
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

            if len(final_values) == 0:
                return None
            if len(final_values) == 1:
                return final_values[0]
            return final_values
        finally:
            self._jit_main_leaf_allowed = False
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
                thread.error = exc
                thread.status = "dead"
                return "error", (self._error_value(exc),)

        return "return", tuple(final_values)
