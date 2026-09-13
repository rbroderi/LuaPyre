from __future__ import annotations

import warnings

from .binary_chunks import (
    BinaryChunkError,
    NATIVE_MAGIC,
    PUC_MAGIC,
    dump_native_chunk,
    fresh_loaded_closure,
    load_native_chunk,
)
from .puc55 import load_puc55_chunk
from .bytecode import Closure
from .compiler import Compiler
from .errors import LuaPyreError, LuaQuotaError, LuaRaisedError, LuaRuntimeError
from .parser import Parser
from .table import LuaTable
from .values import MultiValue, i64, lua_equal, lua_type_name, parse_lua_number, truthy
from .vm import HostFunction
from .stdlib_math import install_math_library
from .stdlib_string import install_string_library
from .stdlib_support import (
    INT_MAX,
    INT_MIN,
    lua_c_function,
    need_integer,
    tostring_value,
)
from .stdlib_table import install_table_library
from .stdlib_utf8 import install_utf8_library


_MISSING = object()


def install_safe_stdlib(globals_table: LuaTable, vm=None):
    def put(name, fn):
        host = lua_c_function(fn, name)
        globals_table.rawset(name.encode("ascii"), host)
        return host

    globals_table.rawset(b"_G", globals_table)
    globals_table.rawset(b"_VERSION", b"Lua 5.5")

    put("type", lambda value: lua_type_name(value).encode("ascii"))
    put("tostring", lambda value: tostring_value(vm, value))

    def tonumber(value, base=None):
        if base is None:
            return parse_lua_number(value)

        if not isinstance(value, bytes):
            raise LuaRuntimeError("bad argument #1 to 'tonumber' (string expected)")
        base = need_integer(base, 2, "tonumber")
        if base < 2 or base > 36:
            raise LuaRuntimeError("bad argument #2 to 'tonumber' (base out of range)")
        try:
            text = value.decode("ascii").strip()
            integer = int(text, base)
        except (UnicodeDecodeError, ValueError):
            return None
        if not INT_MIN <= integer <= INT_MAX:
            return None
        return integer

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
        mt = vm.metatable_for(value) if vm is not None else (value.metatable if isinstance(value, LuaTable) else None)
        if mt is None:
            return None
        protected = mt.rawget(b"__metatable")
        return protected if protected is not None else mt

    put("getmetatable", getmetatable)

    def setmetatable(table, mt=_MISSING, *_ignored):
        if not isinstance(table, LuaTable):
            raise LuaRuntimeError("bad argument #1 to 'setmetatable' (table expected)")
        if mt is _MISSING:
            raise LuaRuntimeError("bad argument #2 to 'setmetatable' (value expected)")
        if mt is not None and not isinstance(mt, LuaTable):
            raise LuaRuntimeError("bad argument #2 to 'setmetatable' (nil or table expected)")
        if table.metatable is not None and table.metatable.rawget(b"__metatable") is not None:
            raise LuaRuntimeError("cannot change a protected metatable")
        table.metatable = mt
        table.version += 1
        if vm is not None and hasattr(vm, "gc"):
            vm.gc.adopt(mt)
            vm.gc.write_barrier(table, mt)
            vm.gc.observe_weak_table(table)
            vm.gc.mark_finalizable(table, mt)
        return table

    put("setmetatable", setmetatable)

    def next_fn(table, key=None, *_ignored):
        if not isinstance(table, LuaTable):
            raise LuaRuntimeError("bad argument #1 to 'next' (table expected)")
        items = list(table.items())
        if key is None:
            return MultiValue(items[0]) if items else None
        for index, (current, value) in enumerate(items):
            if lua_equal(current, key):
                return MultiValue(items[index + 1]) if index + 1 < len(items) else None
        known_deleted, successor = table.successor_after_deleted(key)
        if known_deleted:
            return None if successor is None else MultiValue((successor, table.rawget(successor)))
        raise LuaRuntimeError("invalid key to 'next'")

    next_host = put("next", next_fn)

    def pairs_fallback(value=None):
        if value is None:
            raise LuaRuntimeError("bad argument #1 to 'pairs' (table expected)")
        if not isinstance(value, LuaTable):
            raise LuaRuntimeError("bad argument #1 to 'pairs' (table expected)")
        return MultiValue((next_host, value, None, None))

    pairs_fallback_host = lua_c_function(pairs_fallback, "pairs")

    def pairs_metamethod(value=None):
        return vm._tm(value, b"__pairs") if vm is not None else None

    pairs_metamethod_host = lua_c_function(pairs_metamethod, "pairs")
    if vm is None:
        globals_table.rawset(b"pairs", pairs_fallback_host)
    else:
        # Keep the metamethod call in Lua so __pairs can yield.  A synchronous
        # host callback cannot preserve the suspended Lua continuation.
        factory = Compiler().compile(Parser("""
            local getmetamethod, fallback = ...
            return function (value)
              local metamethod = getmetamethod(value)
              if metamethod ~= nil then return metamethod(value) end
              return fallback(value)
            end
        """).parse())
        wrapper = vm.call_sync(
            Closure(factory, [], globals_table),
            (pairs_metamethod_host, pairs_fallback_host),
            fuel=1_000,
        )[0]
        globals_table.rawset(b"pairs", wrapper)

    def ipairs_iter(value, index):
        index = i64(need_integer(index, 2, "ipairsaux") + 1)
        item = vm.index_sync(value, index) if vm is not None else value.rawget(index)
        return None if item is None else MultiValue((index, item))

    ipairs_host = HostFunction(ipairs_iter, "ipairsaux")

    def ipairs(value=None):
        if value is None:
            raise LuaRuntimeError("bad argument #1 to 'ipairs' (table expected)")
        return MultiValue((ipairs_host, value, 0))

    put("ipairs", ipairs)

    def select(index, *values):
        if isinstance(index, bytes) and index.startswith(b"#"):
            return len(values)
        position = need_integer(index, 1, "select")
        count = len(values)
        if position < 0:
            position = count + position + 1
        if position == 0 or position < 1:
            raise LuaRuntimeError("bad argument #1 to 'select' (index out of range)")
        if position > count:
            return MultiValue(())
        return MultiValue(tuple(values[position - 1:]))

    put("select", select)

    def _error_value(error):
        if isinstance(error, LuaRaisedError):
            return error.value
        return str(error).encode("utf-8", "replace")

    if vm is not None:
        globals_table.rawset(
            b"pcall", HostFunction(lambda *_args: None, "pcall", protected_mode="pcall")
        )
        globals_table.rawset(
            b"xpcall", HostFunction(lambda *_args: None, "xpcall", protected_mode="xpcall")
        )

        def load(chunk, chunkname=None, mode=b"bt", *env_args):
            if mode is None:
                mode = b"bt"
            if not isinstance(mode, bytes):
                raise LuaRuntimeError("bad argument #3 to 'load' (string expected)")
            if any(char not in b"bt" for char in mode) or not mode:
                raise LuaRuntimeError("bad argument #3 to 'load' (invalid mode)")

            if isinstance(chunk, bytes):
                source = chunk
            else:
                pieces = []
                total = 0
                while True:
                    results = vm.call_sync(chunk, ())
                    piece = results[0] if results else None
                    if piece is None or piece == b"":
                        break
                    if not isinstance(piece, bytes):
                        return MultiValue((None, b"reader function must return a string"))
                    total += len(piece)
                    if total > 16 * 1024 * 1024:
                        return MultiValue((None, b"chunk too large"))
                    pieces.append(piece)
                source = b"".join(pieces)

            binary = source.startswith(b"\x1b")
            if binary and b"b" not in mode:
                return MultiValue((None, b"attempt to load a binary chunk (mode is 't')"))
            if not binary and b"t" not in mode:
                return MultiValue((None, b"attempt to load a text chunk (mode is 'b')"))

            environment = globals_table if not env_args else env_args[0]
            try:
                if source.startswith(NATIVE_MAGIC):
                    proto = load_native_chunk(source)
                elif source.startswith(PUC_MAGIC) or binary:
                    proto = load_puc55_chunk(source)
                else:
                    text = source.decode("utf-8")
                    proto = Compiler().compile(Parser(text).parse())
            except (UnicodeDecodeError, LuaPyreError) as error:
                return MultiValue((None, str(error).encode("utf-8", "replace")))
            return fresh_loaded_closure(proto, environment)

        put("load", load)

    warning_enabled = True

    def warn(*messages):
        nonlocal warning_enabled
        if not messages:
            raise LuaRuntimeError("bad argument #1 to 'warn' (string expected)")
        for index, message in enumerate(messages, 1):
            if not isinstance(message, bytes):
                raise LuaRuntimeError(f"bad argument #{index} to 'warn' (string expected)")
        if len(messages) == 1 and messages[0].startswith(b"@"):
            if messages[0] == b"@off":
                warning_enabled = False
            elif messages[0] == b"@on":
                warning_enabled = True
            return None
        if warning_enabled:
            warnings.warn(b"".join(messages).decode("utf-8", "replace"), RuntimeWarning, stacklevel=2)

    put("warn", warn)

    if vm is not None:
        if hasattr(vm, "gc"):
            put("collectgarbage", vm.gc.command)

        stringlib = install_string_library(globals_table, vm)

        def string_dump(fn, strip=False):
            if not isinstance(fn, Closure):
                raise LuaRuntimeError("bad argument #1 to 'dump' (function expected)")
            return dump_native_chunk(fn.proto, strip=truthy(strip))

        stringlib.rawset(b"dump", HostFunction(string_dump, "string.dump"))

        install_utf8_library(globals_table, vm)
        install_table_library(globals_table, vm)
        install_math_library(globals_table, vm)

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
