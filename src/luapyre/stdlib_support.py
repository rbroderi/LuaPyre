from __future__ import annotations

import inspect
import math

from .errors import LuaRuntimeError
from .table import LuaTable
from .values import i64, lua_type_name, parse_lua_number
from .vm import HostFunction


INT_MIN = -(1 << 63)
INT_MAX = (1 << 63) - 1
UINT_MASK = (1 << 64) - 1


def lua_c_function(fn, name: str) -> HostFunction:
    """Wrap a stdlib function so surplus Lua arguments are ignored."""
    parameters = tuple(inspect.signature(fn).parameters.values())
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return HostFunction(fn, name)
    positional = sum(
        parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in parameters
    )

    return HostFunction(fn, name, positional)


def put(table: LuaTable, name: str, fn):
    host = lua_c_function(fn, name)
    table.rawset(name.encode("ascii"), host)
    return host


def need_table(value, arg=1, name="function") -> LuaTable:
    if not isinstance(value, LuaTable):
        raise LuaRuntimeError(f"bad argument #{arg} to '{name}' (table expected)")
    return value


def need_bytes(value, arg=1, name="function") -> bytes:
    if not isinstance(value, bytes):
        raise LuaRuntimeError(f"bad argument #{arg} to '{name}' (string expected)")
    return value


def need_number(value, arg=1, name="function"):
    if type(value) not in (int, float):
        metatable = getattr(value, "metatable", None)
        custom = metatable.rawget(b"__name") if isinstance(metatable, LuaTable) else None
        actual = custom.decode("utf-8", "replace") if isinstance(custom, bytes) else lua_type_name(value)
        raise LuaRuntimeError(
            f"bad argument #{arg} to '{name}' (number expected, got {actual})"
        )
    return value


def to_integer(value):
    number = parse_lua_number(value)
    if type(number) is int:
        return i64(number)
    if type(number) is float and math.isfinite(number) and number.is_integer():
        integer = int(number)
        if INT_MIN <= integer <= INT_MAX:
            return integer
    return None


def need_integer(value, arg=1, name="function") -> int:
    integer = to_integer(value)
    if integer is None:
        if type(value) is float:
            raise LuaRuntimeError(
                f"bad argument #{arg} to '{name}' "
                "(number has no integer representation)"
            )
        raise LuaRuntimeError(f"bad argument #{arg} to '{name}' (integer expected)")
    return integer


def normalize_index(index: int, length: int) -> int:
    """Translate a Lua string position into a 1-based positive position."""
    return index if index >= 0 else length + index + 1


def slice_bounds(i: int, j: int, length: int) -> tuple[int, int]:
    i = normalize_index(i, length)
    j = normalize_index(j, length)
    if i < 1:
        i = 1
    if j > length:
        j = length
    return i, j


def number_to_bytes(value) -> bytes:
    if type(value) is int:
        return str(value).encode("ascii")
    if type(value) is float:
        return repr(value).encode("ascii")
    raise LuaRuntimeError("number expected")


def tostring_value(vm, value) -> bytes:
    metatable = vm.metatable_for(value) if vm is not None else None
    tm = metatable.rawget(b"__tostring") if isinstance(metatable, LuaTable) else None
    if tm is not None:
        results = vm.call_sync(tm, (value,))
        rendered = results[0] if results else None
        if not isinstance(rendered, bytes):
            raise LuaRuntimeError("'__tostring' must return a string")
        return rendered
    if value is None:
        return b"nil"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, bytes):
        return value
    if type(value) in (int, float):
        return number_to_bytes(value)
    name = metatable.rawget(b"__name") if isinstance(metatable, LuaTable) else None
    if isinstance(name, bytes):
        return name + b": " + f"0x{id(value):x}".encode("ascii")
    # Use Lua's conventional opaque function rendering and do not expose
    # Python implementation details for host-backed functions.
    from .bytecode import Closure
    from .vm import HostFunction
    if isinstance(value, (Closure, HostFunction)):
        return f"function: 0x{id(value):x}".encode("ascii")
    return repr(value).encode("utf-8", "replace")
