from __future__ import annotations

from .jitvm import TieredJITVM
from .region_jit import RegionPythonJIT


class OptimizingJITVM(TieredJITVM):
    """Tiered VM using the 0.14 internal-branch region backend."""

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
        self.jit = RegionPythonJIT(
            threshold=jit_threshold,
            enabled=jit_enabled,
        )
