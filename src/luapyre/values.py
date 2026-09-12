from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .table import LuaTable
from .typesys import ANY, BOOLEAN, FLOAT, FUNCTION, INTEGER, NIL, STRING, TABLE, THREAD, LuaType

MASK64 = (1 << 64) - 1
SIGN64 = 1 << 63
INT_MIN = -(1 << 63)
INT_MAX = (1 << 63) - 1

_DECIMAL_NUMBER = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)
_HEX_NUMBER = re.compile(
    r"^[+-]?0[xX](?:(?:[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?)|(?:\.[0-9a-fA-F]+))(?:[pP][+-]?\d+)?$"
)


def i64(value: int) -> int:
    value &= MASK64
    return value - (1 << 64) if value & SIGN64 else value


def parse_lua_number(value):
    """Return Lua's numeric interpretation of a number/string, or ``None``.

    Lua arithmetic coercion accepts complete numeral strings with surrounding
    ASCII whitespace.  Integral hexadecimal strings use Lua's unsigned parsing
    and therefore wrap into the signed 64-bit ``lua_Integer`` domain, while an
    overflowing decimal integer becomes a float when representable.
    """
    if type(value) in (int, float):
        return value
    if not isinstance(value, bytes):
        return None
    try:
        text = value.decode("ascii").strip()
    except UnicodeDecodeError:
        return None
    if not text:
        return None

    try:
        if _HEX_NUMBER.fullmatch(text):
            lower = text.lower()
            if "." in lower or "p" in lower:
                return float.fromhex(text)
            sign = -1 if text.startswith("-") else 1
            digits = text[1:] if text[:1] in "+-" else text
            integer = sign * int(digits[2:], 16)
            return i64(integer)

        if not _DECIMAL_NUMBER.fullmatch(text):
            return None
        if not any(char in text for char in ".eE"):
            integer = int(text, 10)
            if INT_MIN <= integer <= INT_MAX:
                return integer
        return float(text)
    except (ValueError, OverflowError):
        return None


def coerce_lua_integer(value):
    number = parse_lua_number(value)
    if type(number) is int:
        return i64(number)
    if type(number) is float and math.isfinite(number) and number.is_integer():
        integer = int(number)
        if INT_MIN <= integer <= INT_MAX:
            return integer
    return None


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


def _is_thread(value) -> bool:
    try:
        from .threadvm import LuaThread
    except ImportError:
        return False
    return isinstance(value, LuaThread)


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
    if _is_thread(value):
        return "thread"
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
    if _is_thread(value):
        return THREAD
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
