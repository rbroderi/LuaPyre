class LuaPyreError(Exception):
    """Base exception for LuaPyre."""


class LuaSyntaxError(LuaPyreError):
    pass


class LuaTypeError(LuaPyreError):
    pass


class LuaRuntimeError(LuaPyreError):
    pass


class LuaQuotaError(LuaRuntimeError):
    pass
