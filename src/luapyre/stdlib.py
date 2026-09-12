from __future__ import annotations

from .errors import LuaRuntimeError, LuaRaisedError
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


def install_safe_stdlib(globals_table: LuaTable, vm=None):
    def put(name, fn):
        host = HostFunction(fn, name)
        globals_table.rawset(name.encode(), host)
        return host

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
            raise LuaRaisedError(message)
        return MultiValue(tuple(args))
    put("assert", lua_assert)

    def lua_error(value=None, _level=1):
        if value is None:
            value = b"error object is nil"
        raise LuaRaisedError(value)
    put("error", lua_error)

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

    def getmetatable(value):
        if not isinstance(value, LuaTable):
            return None
        mt = value.metatable
        if mt is None:
            return None
        protected = mt.rawget(b"__metatable")
        return protected if protected is not None else mt
    put("getmetatable", getmetatable)

    def setmetatable(table, mt):
        if not isinstance(table, LuaTable):
            raise LuaRuntimeError("bad argument #1 to 'setmetatable' (table expected)")
        if mt is not None and not isinstance(mt, LuaTable):
            raise LuaRuntimeError("bad argument #2 to 'setmetatable' (nil or table expected)")
        if table.metatable is not None and table.metatable.rawget(b"__metatable") is not None:
            raise LuaRuntimeError("cannot change a protected metatable")
        table.metatable = mt
        table.version += 1
        if vm is not None and hasattr(vm, "gc"):
            vm.gc.mark_finalizable(table, mt)
        return table
    put("setmetatable", setmetatable)

    def next_fn(table, key=None):
        if not isinstance(table, LuaTable):
            raise LuaRuntimeError("bad argument #1 to 'next' (table expected)")
        items = list(table.items())
        if key is None:
            return MultiValue(items[0]) if items else None
        for i, (current, value) in enumerate(items):
            if lua_equal(current, key):
                return MultiValue(items[i + 1]) if i + 1 < len(items) else None
        raise LuaRuntimeError("invalid key to 'next'")

    next_host = put("next", next_fn)

    def pairs(table):
        if not isinstance(table, LuaTable):
            raise LuaRuntimeError("bad argument #1 to 'pairs' (table expected)")
        return MultiValue((next_host, table, None))
    put("pairs", pairs)

    def ipairs_iter(table, index):
        index = i64(index + 1)
        value = table.rawget(index)
        return None if value is None else MultiValue((index, value))

    ipairs_host = HostFunction(ipairs_iter, "ipairsaux")

    def ipairs(table):
        if not isinstance(table, LuaTable):
            raise LuaRuntimeError("bad argument #1 to 'ipairs' (table expected)")
        return MultiValue((ipairs_host, table, 0))
    put("ipairs", ipairs)

    if vm is not None:
        if hasattr(vm, "gc"):
            put("collectgarbage", vm.gc.command)

        coroutine = LuaTable()
        coroutine.rawset(b"create", HostFunction(vm.create_thread, "coroutine.create"))
        coroutine.rawset(b"resume", HostFunction(vm.resume_thread, "coroutine.resume"))
        coroutine.rawset(b"yield", HostFunction(vm.yield_current, "coroutine.yield"))
        coroutine.rawset(b"status", HostFunction(vm.coroutine_status, "coroutine.status"))
        coroutine.rawset(b"running", HostFunction(vm.running_thread, "coroutine.running"))
        coroutine.rawset(b"isyieldable", HostFunction(vm.is_yieldable, "coroutine.isyieldable"))
        coroutine.rawset(b"close", HostFunction(vm.close_thread, "coroutine.close"))
        coroutine.rawset(b"wrap", HostFunction(vm.wrap_thread, "coroutine.wrap"))
        globals_table.rawset(b"coroutine", coroutine)
