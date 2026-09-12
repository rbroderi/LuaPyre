from __future__ import annotations

from .bytecode import Closure
from .errors import LuaRuntimeError
from .function_jit import _FUNC_RETURN
from .jitvm import TieredJITVM
from .super_jit import SuperPythonJIT


class OptimizingJITVM(TieredJITVM):
    """Tiered VM using the 0.15 AST super-JIT.

    Ordinary Lua retains the guarded 0.13/0.14 paths. Certified fully typed
    source additionally gets register-promoted super-regions and whole-function
    compilation with direct compiled recursion/calls.
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
        super().__init__(
            globals,
            fuel=fuel,
            max_frames=max_frames,
            jit_enabled=jit_enabled,
            jit_threshold=jit_threshold,
        )
        self.jit = SuperPythonJIT(
            threshold=jit_threshold,
            enabled=jit_enabled,
        )

    def _invoke(self, frames, parent, fn, args, dest, want, tail=False):
        # The base TieredJITVM exposes a synchronized main-thread budget only
        # while executing a direct CALL/CALLV handler. Coroutines already keep
        # their live budget on the VM object. Whole-function compilation obeys
        # the same entry rule as the leaf tier.
        thread = self.current_thread
        budget_is_current = (
            thread is not None and not thread.is_main
        ) or self._jit_main_leaf_allowed

        if (
            isinstance(fn, Closure)
            and fn.proto.jit_fully_typed
            and not tail
            and not self._sync_frame_prefixes
            and self.jit.enabled
            and budget_is_current
        ):
            compiled = self.jit.maybe_function(fn)
            if compiled is not None:
                if len(frames) >= self.max_frames:
                    raise LuaRuntimeError("stack overflow")
                child = self._new_frame(fn, list(args), dest, want)
                frames.append(child)
                meter = [0]
                try:
                    status, values = compiled.runner(
                        self,
                        frames,
                        child,
                        self._jit_budget(),
                        meter,
                    )
                except BaseException:
                    if meter[0]:
                        self._jit_consume(meter[0])
                    raise

                if meter[0]:
                    self._jit_consume(meter[0])

                if status == _FUNC_RETURN:
                    if not frames or frames[-1] is not child:
                        raise RuntimeError("compiled function stack mismatch")
                    frames.pop()
                    self._write_results(parent.regs, dest, want, values)
                    self.jit.function_executions += 1
                else:
                    # Guard/budget suspension deliberately leaves the compiled
                    # frame (and possibly a nested child) on the real VM stack.
                    # Tier 0 resumes from the exact spilled PC on the next turn.
                    self.jit.function_suspends += 1
                    if self._jit_budget() > 0 and frames:
                        active = frames[-1]
                        self.trace_jit.record_side_exit(active.proto, active.pc)
                return None

        return super()._invoke(frames, parent, fn, args, dest, want, tail=tail)
