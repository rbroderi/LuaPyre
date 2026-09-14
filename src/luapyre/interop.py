from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NewType

if TYPE_CHECKING:
    from .runtime import LuaRuntime


LuaInt = NewType("LuaInt", int)
"""A Python type-hint marker for LuaPyre's non-overflowing typed integer."""


@dataclass(frozen=True, slots=True)
class LuaFunction:
    """A Lua function bound to the runtime that owns it.

    Instances are returned by the Python-facing conversion APIs and can be
    called like ordinary Python functions. Arguments cross the same recursive
    conversion boundary as :meth:`LuaRuntime.call`.
    """

    _runtime: LuaRuntime
    _value: object
    name: str = "?"
    _leaf_cache: list[object] = field(
        default_factory=lambda: [None], compare=False, repr=False
    )
    _adapter_cache: list[object] = field(
        default_factory=lambda: [None], compare=False, repr=False
    )

    def __call__(
        self,
        *args: object,
        return_type: object = None,
        fuel: int | None = None,
    ) -> Any:
        adapter = self._adapter_cache[0]
        if adapter is None:
            adapter = self._runtime._make_bound_adapter(
                self._value, self._leaf_cache
            )
            if adapter is None:
                adapter = False
            self._adapter_cache[0] = adapter
        if adapter is not False:
            return adapter(args, return_type, fuel)
        return self._runtime._call_bound(
            self._value,
            args,
            return_type=return_type,
            fuel=fuel,
            leaf_cache=self._leaf_cache,
        )

    @property
    def raw(self) -> object:
        """Return the low-level LuaPyre function value."""
        return self._value
