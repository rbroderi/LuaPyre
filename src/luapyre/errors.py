from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LuaTraceFrame:
    """One Lua frame captured while propagating a runtime error."""

    source: str | bytes | None
    line: int
    name: str
    tailcall: bool = False


class LuaPyreError(Exception):
    """Base exception for LuaPyre."""


class LuaSyntaxError(LuaPyreError):
    def __init__(self, message, *, line: int | None = None, column: int | None = None):
        self.line = line
        self.column = column
        super().__init__(message)


class LuaTypeError(LuaPyreError):
    pass


class LuaRuntimeError(LuaPyreError):
    """Runtime failure with a Lua error object and structured Lua traceback."""

    def __init__(
        self,
        message,
        *,
        value=None,
        trace: list[LuaTraceFrame] | None = None,
        located: bool = False,
    ):
        self.value = value
        self.trace = list(trace or ())
        self.located = located
        super().__init__(message)

    def add_trace_frame(self, frame: LuaTraceFrame) -> None:
        if not self.trace or self.trace[-1] != frame:
            self.trace.append(frame)


class LuaRaisedError(LuaRuntimeError):
    """A Lua-level error that retains its exact Lua error object."""

    def __init__(
        self,
        value,
        *,
        trace: list[LuaTraceFrame] | None = None,
        located: bool = False,
    ):
        self.value = value
        if isinstance(value, bytes):
            message = value.decode("utf-8", "replace")
        else:
            message = str(value)
        super().__init__(message, value=value, trace=trace, located=located)


class LuaQuotaError(LuaRuntimeError):
    pass
