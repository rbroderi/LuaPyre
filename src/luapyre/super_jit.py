from __future__ import annotations

from .ast_jit import AstPythonJIT
from .bytecode import Op
from .dense_jit import DenseEmitterJITMixin
from .function_jit import TypedFunctionJITMixin
from .region_jit import RegionPythonJIT
from .structured_jit import StructuredTypedLoopJITMixin
from .typed_ir_function_jit import TypedIRFunctionJITMixin
from .typed_ir_jit import TypedIRLoopJITMixin


class SuperPythonJIT(
    TypedIRFunctionJITMixin,
    TypedFunctionJITMixin,
    StructuredTypedLoopJITMixin,
    DenseEmitterJITMixin,
    TypedIRLoopJITMixin,
    AstPythonJIT,
):
    """Typed optimizer pipeline ending in the Python-AST backend.

    0.16 makes the architectural boundary explicit: fully typed regions and
    whole functions with IR-relevant operations are lowered through a small
    backend-neutral IR before Python-specific AST specialization. Pure numeric
    straight-line loops and call-heavy/recursive functions retain the proven
    0.15 backends until their IR lowering is at least as exact and fast. Earlier
    tiers remain fail-closed fallbacks for unsupported shapes and ordinary Lua.
    """

    _IR_PRIMARY_OPS = frozenset({Op.GETUPVAL, Op.GETTABLE, Op.SETTABLE})

    def _compile_loop(self, frame, start_pc: int, backedge_pc: int):
        # Compiler-tier routing belongs here rather than inside Python AST
        # peepholes. Table/environment operations are precisely where the new
        # typed IR carries alias, constant-key, cache and loop-invariance facts,
        # so give that optimizer first refusal. Unsupported IR shapes return
        # None and immediately fall through to the proven 0.15 tiers.
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
