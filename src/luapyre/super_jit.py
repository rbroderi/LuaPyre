from __future__ import annotations

from .ast_jit import AstPythonJIT
from .bytecode import Op
from .cfg_loop_opt_jit import CFGLoopOptimizationJITMixin
from .cfg_value_ir_function_jit import CFGValueIRFunctionJITMixin
from .dense_jit import DenseEmitterJITMixin
from .direct_call_ir_function_jit import DirectCallIRFunctionJITMixin
from .function_jit import TypedFunctionJITMixin
from .region_jit import RegionPythonJIT
from .structured_jit import StructuredTypedLoopJITMixin
from .typed_ir_function_jit import TypedIRFunctionJITMixin
from .typed_ir_jit import TypedIRLoopJITMixin
from .value_ir_function_jit import ValueIRFunctionJITMixin


class SuperPythonJIT(
    CFGLoopOptimizationJITMixin,
    CFGValueIRFunctionJITMixin,
    ValueIRFunctionJITMixin,
    DirectCallIRFunctionJITMixin,
    TypedIRFunctionJITMixin,
    TypedFunctionJITMixin,
    StructuredTypedLoopJITMixin,
    DenseEmitterJITMixin,
    TypedIRLoopJITMixin,
    AstPythonJIT,
):
    """Typed compiler pipeline ending in the optimized Python-AST backend.

    0.25 shapes emitted functions for CPython specialization. Dense constant
    register accesses become fast locals, backend-neutral integer intervals
    remove proven-unnecessary signed-64 wrapping, and proven integer loop
    backedges omit repeated numeric representation dispatch.

    0.22 adds a profile-guided trace tier around this function/region compiler.
    Typed interpreter branches can OSR the live Frame directly into a cyclic
    scalar CFG trace, and alternate successors become exact hot side exits.

    0.21 layers adaptive call/table inline caches and deoptimization feedback
    around the 0.20 backend-neutral loop optimizer. Stable bytecode sites grow
    from monomorphic to bounded polymorphic caches; megamorphic sites and
    repeatedly failing compiled regions fall back permanently to Tier 0.

    0.20 layers backend-neutral loop optimization on the 0.19 cyclic CFG/value
    graph. Dominance/natural-loop facts now prove loop-invariant scalar values
    and canonical integer induction variables. The Python backend materializes
    invariants once and can update a loop-carried phi directly when its temporary
    backedge expression has no other consumer. Lua bytecode fuel remains charged
    at the original dynamic instruction sites.

    Environment/table-heavy loop regions continue through TypedIR first, where
    invariant global reads already hoist their raw-load and table/metatable shape
    guard ahead of the generated loop. That speculative guard fails closed to
    Tier 0 before any Lua instruction is consumed. This keeps guard specialization
    in backend-neutral IR while CFG Value IR grows scalar LICM/induction support.

    The 0.17 value tier still owns straight-line pure functions and lexical-call
    inlining. The 0.23 escape plan lets CALL IR virtualize eligible branchy leaf
    Frames and open-result MultiValues, sinking real objects to suspension/error
    boundaries. Optimization decisions live in typed/value/CALL/CFG/loop IR.
    Python AST is a backend, and every unsupported shape fails closed to an
    earlier exact tier or the interpreter.
    """

    _IR_PRIMARY_OPS = frozenset(
        {Op.GETUPVAL, Op.GETTABLE, Op.SETTABLE, Op.CONCAT}
    )

    def _compile_loop(self, frame, start_pc: int, backedge_pc: int):
        if frame.proto.jit_fully_typed:
            compiled = self._compile_structured_typed_loop(
                frame, start_pc, backedge_pc
            )
            if compiled is not None:
                return compiled
        # Table/environment operations are precisely where TypedIR carries alias,
        # constant-key, cache and guard-hoisting facts, so give that optimizer
        # first refusal. Unsupported IR shapes return None and immediately fall
        # through to the proven earlier tiers.
        if frame.proto.jit_fully_typed:
            body = frame.proto.code[start_pc:backedge_pc]
            if any(ins.op in self._IR_PRIMARY_OPS for ins in body):
                compiled = TypedIRLoopJITMixin._compile_ast_loop(
                    self, frame, start_pc, backedge_pc
                )
                if compiled is not None:
                    return compiled
        return super()._compile_loop(frame, start_pc, backedge_pc)

    def _emit_instruction(self, *args, **kwargs):
        """Route the two deliberately different region-emitter protocols.

        ``RegionPythonJIT`` owns the proven 0.14 source-string emitter and calls
        ``self._emit_instruction(..., offset=..., trusted=...)``. The typed AST
        super-region emitter carries frame/register/call metadata instead. Keep
        both implementations independent so ordinary Lua never changes behavior
        merely because the typed backend is installed.
        """

        if "offset" in kwargs or "trusted" in kwargs:
            return RegionPythonJIT._emit_instruction(self, *args, **kwargs)
        return AstPythonJIT._emit_instruction(self, *args, **kwargs)
