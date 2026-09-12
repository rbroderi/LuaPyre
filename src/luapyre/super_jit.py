from __future__ import annotations

from .ast_jit import AstPythonJIT
from .bytecode import Op
from .dense_jit import DenseEmitterJITMixin
from .direct_call_ir_function_jit import DirectCallIRFunctionJITMixin
from .function_jit import TypedFunctionJITMixin
from .region_jit import RegionPythonJIT
from .structured_jit import StructuredTypedLoopJITMixin
from .typed_ir_function_jit import TypedIRFunctionJITMixin
from .typed_ir_jit import TypedIRLoopJITMixin
from .value_ir_function_jit import ValueIRFunctionJITMixin


class SuperPythonJIT(
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

    0.17 layers SSA-like value numbering and backend-neutral CALL facts over the
    0.16 typed IR. Tiny pure lexical calls may disappear into the value graph;
    larger statically resolved lexical calls retain real Lua closures/frames and
    run through the same shared-meter direct-call machinery used by the proven
    function compiler. Table/global functions retain the 0.16 typed-IR backend,
    while specialized numeric regions keep the faster structured tiers.

    The important boundary is architectural: optimization decisions live in the
    typed/value/CALL IR. Python AST is a backend, and every unsupported shape
    fails closed to an earlier exact tier or the interpreter.
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
