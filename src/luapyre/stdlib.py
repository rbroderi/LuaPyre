from __future__ import annotations

from .errors import LuaRuntimeError
from .table import LuaTable
from .values import MultiValue, i64, lua_equal, lua_type_name, truthy
from .vm import HostFunction


def _tostring(value):
    if value is None:
        return b"nil"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, bytes):
        return value
    if type(value) in (int, float):
        return str(value).encode("ascii")
    return repr(value).encode("utf-8")


def install_safe_stdlib(globals_table: LuaTable):
    def put(name, fn):
        globals_table.rawset(name.encode(), HostFunction(fn, name))

    globals_table.rawset(b"_G", globals_table)
    globals_table.rawset(b"_VERSION", b"Lua 5.5")

    put("type", lambda value: lua_type_name(value).encode())
    put("tostring", _tostring)

    def tonumber(value):
        if type(value) in (int, float):
            return value
        if not isinstance(value, bytes):
            return None
        try:
            s = value.decode("ascii").strip()
            if any(c in s for c in ".eE"):
                return float(s)
            return i64(int(s, 0))
        except (ValueError, UnicodeDecodeError):
            return None
    put("tonumber", tonumber)

    def lua_assert(*args):
        value = args[0] if args else None
        if not truthy(value):
            message = args[1] if len(args) > 1 else b"assertion failed!"
            if isinstance(message, bytes):
                message = message.decode("utf-8", "replace")
            raise LuaRuntimeError(str(message))
        return MultiValue(tuple(args))
    put("assert", lua_assert)

    put("rawequal", lua_equal)

    def rawget(table, key):
        if not isinstance(table, LuaTable):
            raise LuaRuntimeError("bad argument #1 to 'rawget' (table expected)")
        return table.rawget(key)
    put("rawget", rawget)

    def rawset(table, key, value):
        if not isinstance(table, LuaTable):
            raise LuaRuntimeError("bad argument #1 to 'rawset' (table expected)")
        table.rawset(key, value)
        return table
    put("rawset", rawset)

    def rawlen(value):
        if isinstance(value, LuaTable):
            return value.rawlen()
        if isinstance(value, bytes):
            return len(value)
        raise LuaRuntimeError("bad argument #1 to 'rawlen' (table or string expected)")
    put("rawlen", rawlen)
