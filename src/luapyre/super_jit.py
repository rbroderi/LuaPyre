from __future__ import annotations

from .ast_jit import AstPythonJIT
from .function_jit import TypedFunctionJITMixin


class SuperPythonJIT(TypedFunctionJITMixin, AstPythonJIT):
    """0.15 Python backend: super-regions plus whole typed functions."""

    pass
