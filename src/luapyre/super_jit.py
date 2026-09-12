from __future__ import annotations

from .ast_jit import AstPythonJIT
from .dense_jit import DenseEmitterJITMixin
from .function_jit import TypedFunctionJITMixin
from .region_jit import RegionPythonJIT
from .structured_jit import StructuredTypedLoopJITMixin
from .typed_ir_jit import TypedIRLoopJITMixin


class SuperPythonJIT(
    TypedFunctionJITMixin,
    StructuredTypedLoopJITMixin,
    DenseEmitterJITMixin,
    TypedIRLoopJITMixin,
    AstPythonJIT,
):
    """Typed optimizer pipeline ending in the Python-AST backend.

    0.16 makes the architectural boundary explicit: fully typed regions are
    lowered through a small backend-neutral IR before Python-specific AST
    specialization. Earlier proven tiers remain in the MRO as fail-closed
    fallbacks for unsupported shapes and ordinary Lua.
    """

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
