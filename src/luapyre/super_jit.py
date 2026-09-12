from __future__ import annotations

from .ast_jit import AstPythonJIT
from .function_jit import TypedFunctionJITMixin
from .region_jit import RegionPythonJIT


class SuperPythonJIT(TypedFunctionJITMixin, AstPythonJIT):
    """0.15 Python backend: super-regions plus whole typed functions."""

    def _emit_instruction(self, *args, **kwargs):
        """Route the two deliberately different region-emitter protocols.

        ``RegionPythonJIT`` owns the proven 0.14 source-string emitter and calls
        ``self._emit_instruction(..., offset=..., trusted=...)``. The 0.15 AST
        super-region emitter carries frame/register/call metadata instead. Keep
        both implementations independent so ordinary Lua never changes behavior
        merely because the typed backend is installed.
        """

        if "offset" in kwargs or "trusted" in kwargs:
            return RegionPythonJIT._emit_instruction(self, *args, **kwargs)
        return AstPythonJIT._emit_instruction(self, *args, **kwargs)
