from __future__ import annotations

from .ast_jit import AstPythonJIT
from .jitvm import TieredJITVM


class OptimizingJITVM(TieredJITVM):
    """Tiered VM using the typed 0.15 AST super-region backend.

    ``AstPythonJIT`` inherits the 0.14 branch-region backend, so ordinary Lua
    retains the proven guarded path while fully typed source can use register
    promotion, local jump-list lowering/AST inlining, and wider CFG regions.
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
        self.jit = AstPythonJIT(
            threshold=jit_threshold,
            enabled=jit_enabled,
        )
