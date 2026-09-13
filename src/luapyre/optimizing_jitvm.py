from __future__ import annotations

from .bytecode import Closure
from .errors import LuaRuntimeError
from .function_jit import _FUNC_RETURN
from .jitvm import TieredJITVM
from .super_jit import SuperPythonJIT
from .values import static_value_type, type_matches


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
        debug_hooks_enabled: bool = False,
    ):
        super().__init__(
            globals,
            fuel=fuel,
            max_frames=max_frames,
            jit_enabled=jit_enabled,
            jit_threshold=jit_threshold,
            debug_hooks_enabled=debug_hooks_enabled,
        )
        self.jit = SuperPythonJIT(
            threshold=jit_threshold,
            enabled=jit_enabled,
        )

    def _acquire_compiled_frame(
        self, compiled, closure, args, return_reg, return_want, *, validate_args=True
    ):
        """Reuse an inactive exact Frame for a compiled call when available."""
        proto = closure.proto
        pool = compiled.frame_pool
        if pool:
            frame = pool.pop()
            regs = frame.regs
            for index in range(proto.param_count):
                arg = args[index] if index < len(args) else None
                if validate_args:
                    expected = proto.param_types[index].name
                    if not type_matches(expected, arg):
                        raise LuaRuntimeError(
                            f"argument {index + 1}: expected {expected}, "
                            f"got {static_value_type(arg).name}"
                        )
                regs[index] = arg
            if proto.env_reg >= 0:
                regs[proto.env_reg] = closure.env
            frame.closure = closure
            frame.pc = 0
            frame.return_reg = return_reg
            frame.return_want = return_want
            if self.debug_hooks_enabled:
                frame.hook_call_values = tuple(args[:proto.param_count])
            return frame

        self.jit.stats.compiled_frame_allocations += 1
        if validate_args:
            return self._new_frame(closure, list(args), return_reg, return_want)
        regs = [None] * max(1, proto.register_count)
        for index in range(proto.param_count):
            regs[index] = args[index] if index < len(args) else None
        if proto.env_reg >= 0:
            regs[proto.env_reg] = closure.env
        from .vm import Frame

        return Frame(closure, regs, 0, return_reg, return_want)

    def _release_compiled_frame(self, compiled, frame) -> None:
        pool = compiled.frame_pool
        if len(pool) >= min(128, self.max_frames):
            return
        frame.cells.clear()
        frame.close_stack.clear()
        frame.puc_close_stack.clear()
        frame.pending_error = None
        frame.return_prefix = ()
        frame.return_limit = -1
        frame.protected_handler = None
        frame.protected_name = None
        frame.protected_error = None
        frame.protected_error_depth = 0
        frame.trace_name = None
        frame.call_name = None
        frame.call_namewhat = ""
        pool.append(frame)

    def _invoke(self, frames, parent, fn, args, dest, want, tail=False):
        # The base TieredJITVM exposes a synchronized main-thread budget only
        # while executing a direct CALL/CALLV handler. Coroutines already keep
        # their live budget on the VM object. Whole-function compilation obeys
        # the same entry rule as the leaf tier.
        thread = self.current_thread
        if thread is not None and not thread.is_main:
            return super()._invoke(frames, parent, fn, args, dest, want, tail=tail)
        budget_is_current = (
            self._jit_main_leaf_allowed
        )

        if (
            isinstance(fn, Closure)
            and fn.proto.jit_fully_typed
            and not tail
            and not self._sync_frame_prefixes
            and not self.hooks_active()
            and self.jit.enabled
            and budget_is_current
        ):
            compiled = self.jit.maybe_function(fn)
            if compiled is not None:
                if len(frames) >= self.max_frames:
                    raise LuaRuntimeError("stack overflow")
                child = self._acquire_compiled_frame(compiled, fn, args, dest, want)
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
                    self._release_compiled_frame(compiled, child)
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
