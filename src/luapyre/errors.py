class LuaPyreError(Exception):
    """Base exception for LuaPyre."""


class LuaSyntaxError(LuaPyreError):
    pass


class LuaTypeError(LuaPyreError):
    pass


class LuaRuntimeError(LuaPyreError):
    pass


class LuaRaisedError(LuaRuntimeError):
    """A Lua-level error that retains its Lua error object."""

    def __init__(self, value):
        self.value = value
        if isinstance(value, bytes):
            message = value.decode("utf-8", "replace")
        else:
            message = str(value)
        super().__init__(message)


class LuaQuotaError(LuaRuntimeError):
    pass
