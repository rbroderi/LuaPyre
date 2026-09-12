from __future__ import annotations

from .ast_jit import AstPythonJIT
from .bytecode import Op
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

    0.19 gives dominance-aware reducible scalar CFGs first refusal. Forward
    branches retain the 0.18 phi/side-exit contract, while natural backedges now
    introduce loop-carried phi values in the same backend-neutral value graph.
    Simple whole-function natural loops lower directly to Python ``while``;
    branchy/nested reducible loops use the generic predecessor-tracked CFG
    backend. The older structured hot-loop tier remains as a region/top-level
    fallback for shapes that do not enter whole-function CFG Value IR.

    The 0.17 value tier still owns straight-line pure functions and lexical-call
    inlining; larger static calls retain real Lua closures/frames through CALL
    IR. Optimization decisions live in typed/value/CALL/CFG IR. Python AST is a
    backend, and every unsupported shape fails closed to an earlier exact tier or
    the interpreter.
    """

    _IR_PRIMARY_OPS = frozenset({Op.GETUPVAL, Op.GETTABLE, Op.SETTABLE})

    def _compile_loop(self, frame, start_pc: int, backedge_pc: int):
        # Compiler-tier routing belongs here rather than inside Python AST
        # peepholes. Table/environment operations are precisely where the 0.16
        # typed IR carries alias, constant-key, cache and loop-invariance facts,
        # so give that optimizer first refusal. Unsupported IR shapes return
        # None and immediately fall through to the proven earlier tiers.
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
