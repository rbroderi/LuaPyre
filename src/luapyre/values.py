from __future__ import annotations

from dataclasses import dataclass
import math

from .table import LuaTable
from .typesys import ANY, BOOLEAN, FLOAT, FUNCTION, INTEGER, NIL, STRING, TABLE, LuaType

MASK64 = (1 << 64) - 1
SIGN64 = 1 << 63


def i64(value: int) -> int:
    value &= MASK64
    return value - (1 << 64) if value & SIGN64 else value


def truthy(value) -> bool:
    return value is not None and value is not False


def lua_equal(a, b) -> bool:
    if type(a) is bool or type(b) is bool:
        return type(a) is bool and type(b) is bool and a is b
    if type(a) in (int, float) and type(b) in (int, float):
        return a == b
    if isinstance(a, (bytes, type(None))) or isinstance(b, (bytes, type(None))):
        return type(a) is type(b) and a == b
    if isinstance(a, LuaTable) or isinstance(b, LuaTable):
        return a is b
    return a is b


def lua_type_name(value) -> str:
    if value is None:
        return "nil"
    if type(value) is bool:
        return "boolean"
    if type(value) in (int, float):
        return "number"
    if isinstance(value, bytes):
        return "string"
    if isinstance(value, LuaTable):
        return "table"
    from .bytecode import Closure
    from .vm import HostFunction
    if isinstance(value, (Closure, HostFunction)):
        return "function"
    return "userdata"


def static_value_type(value) -> LuaType:
    if value is None:
        return NIL
    if type(value) is bool:
        return BOOLEAN
    if type(value) is int:
        return INTEGER
    if type(value) is float:
        return FLOAT
    if isinstance(value, bytes):
        return STRING
    if isinstance(value, LuaTable):
        return TABLE
    from .bytecode import Closure
    from .vm import HostFunction
    if isinstance(value, (Closure, HostFunction)):
        return FUNCTION
    return ANY


def type_matches(type_name: str, value) -> bool:
    actual = static_value_type(value).name
    if type_name == "Any":
        return True
    if type_name == "number":
        return actual in ("integer", "float")
    if " | " in type_name:
        return any(type_matches(part, value) for part in type_name.split(" | "))
    return actual == type_name


@dataclass(frozen=True, slots=True)
class MultiValue:
    values: tuple[object, ...]

    def first(self):
        return self.values[0] if self.values else None
