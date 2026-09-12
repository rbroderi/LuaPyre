from __future__ import annotations

from .bytecode import Closure, Op, Proto
from .diagnostics import capture_error
from .errors import LuaQuotaError, LuaRuntimeError
from .gcvm import GarbageCollectedVM
from .jit import DEOPT, PythonJIT
from .opdispatch import OPCODE_HANDLERS
from .threadvm import _CloseSelfSignal, _YieldSignal
from .vm import Frame


def _jit_forloop(vm, frames, frame, ins, regs, constants):
    """FORLOOP handler that hands only taken backedges to the JIT tier."""
    if vm._forloop(regs, ins):
        frame.pc = ins.d
        vm._jit_backedge(frame)


_JIT_OPCODE_HANDLERS = dict(OPCODE_HANDLERS)
_JIT_OPCODE_HANDLERS[Op.FORLOOP] = _jit_forloop


class TieredJITVM(GarbageCollectedVM):
    """GarbageCollectedVM with an optional guarded tier-2 Python JIT.

    The existing interpreter remains authoritative. Hot loop regions and hot
    straight-line leaf functions may run as generated Python; every unsupported
    or guard-miss path resumes the ordinary opcode interpreter.

    JIT loop probing happens only after a taken native FORLOOP backedge. Cold
    and unsupported straight-line code therefore does not pay a per-opcode JIT
    lookup tax.
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

    def _jit_backedge(self, frame) -> None:
        if not self.jit.enabled:
            return
        used, _handled = self.jit.try_loop(self, frame, self._jit_budget())
        if used:
            self._jit_consume(used)

    def _invoke(self, frames, parent, fn, args, dest, want, tail=False):
        # Synchronous stdlib callbacks intentionally stay on the interpreter:
        # their local fuel accounting and visible-frame prefix are specialized
        # for callback semantics. Tail calls also retain the normal frame swap.
        if (
            isinstance(fn, Closure)
            and not tail
            and not self._sync_frame_prefixes
            and self.jit.enabled
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
        self._jit_main_fuel = self.default_fuel if fuel is None else fuel
        try:
            root = Closure(proto, [], self.globals)
            root_regs = [None] * max(1, proto.register_count)
            if proto.env_reg >= 0:
                root_regs[proto.env_reg] = self.globals
            frames = [Frame(root, root_regs)]
            final_values = ()
            handlers = _JIT_OPCODE_HANDLERS

            while frames:
                try:
                    frame = frames[-1]
                    if self._drive_pending(frames, frame):
                        continue

                    self._jit_consume(1)
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
