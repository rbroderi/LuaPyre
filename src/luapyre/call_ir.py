from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StaticClosureRef:
    """Compile-time identity of a non-escaping lexical child closure."""

    child_index: int


@dataclass(frozen=True, slots=True)
class StaticCallSite:
    """Backend-neutral description of one statically resolved Lua CALL.

    The call remains charged as a Lua CALL plus the complete reachable callee
    bytecode. A backend may inline ``child_index`` only when its optimizer has
    separately proved that the child is a pure fully typed leaf and that the
    closure value cannot escape.
    """

    pc: int
    child_index: int
    arg_count: int
    result_count: int
    callee_instruction_count: int

    @property
    def total_inline_cost(self) -> int:
        """Extra cost beyond the caller's already-counted CALL instruction."""

        return self.callee_instruction_count
