from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runtime import LuaRuntime


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

    def __call__(
        self,
        *args: object,
        return_type: object = None,
        fuel: int | None = None,
    ) -> Any:
        return self._runtime.call(
            self,
            *args,
            return_type=return_type,
            fuel=fuel,
        )

    @property
    def raw(self) -> object:
        """Return the low-level LuaPyre function value."""
        return self._value
