from __future__ import annotations

from .errors import LuaRuntimeError
from .table import LuaTable
from .values import MultiValue
from .vm import HostFunction
from .stdlib_support import need_bytes, need_integer, normalize_index


CHARPATTERN = b"[\x00-\x7f\xc2-\xfd][\x80-\xbf]*"


def _encode(codepoint: int) -> bytes:
    if codepoint < 0 or codepoint > 0x7FFFFFFF:
        raise LuaRuntimeError("value out of range")
    if codepoint <= 0x7F:
        return bytes((codepoint,))
    if codepoint <= 0x7FF:
        count, lead = 2, 0xC0
    elif codepoint <= 0xFFFF:
        count, lead = 3, 0xE0
    elif codepoint <= 0x1FFFFF:
        count, lead = 4, 0xF0
    elif codepoint <= 0x3FFFFFF:
        count, lead = 5, 0xF8
    else:
        count, lead = 6, 0xFC
    out = [0] * count
    value = codepoint
    for index in range(count - 1, 0, -1):
        out[index] = 0x80 | (value & 0x3F)
        value >>= 6
    masks = {2: 0x1F, 3: 0x0F, 4: 0x07, 5: 0x03, 6: 0x01}
    out[0] = lead | (value & masks[count])
    return bytes(out)


def _decode(data: bytes, start: int, lax: bool = False) -> tuple[int, int]:
    """Decode one character at zero-based ``start``; return (codepoint, end)."""
    if start < 0 or start >= len(data):
        raise LuaRuntimeError("invalid UTF-8 code")
    first = data[start]
    if first < 0x80:
        return first, start + 1
    if 0xC2 <= first <= 0xDF:
        count, value = 2, first & 0x1F
    elif 0xE0 <= first <= 0xEF:
        count, value = 3, first & 0x0F
    elif 0xF0 <= first <= 0xF7:
        count, value = 4, first & 0x07
    elif 0xF8 <= first <= 0xFB:
        count, value = 5, first & 0x03
    elif 0xFC <= first <= 0xFD:
        count, value = 6, first & 0x01
    else:
        raise LuaRuntimeError("invalid UTF-8 code")
    if start + count > len(data):
        raise LuaRuntimeError("invalid UTF-8 code")
    for pos in range(start + 1, start + count):
        byte = data[pos]
        if byte < 0x80 or byte > 0xBF:
            raise LuaRuntimeError("invalid UTF-8 code")
        value = (value << 6) | (byte & 0x3F)
    minimum = (0, 0, 0x80, 0x800, 0x10000, 0x200000, 0x4000000)[count]
    if value < minimum:
        raise LuaRuntimeError("invalid UTF-8 code")
    if not lax and (value > 0x10FFFF or 0xD800 <= value <= 0xDFFF):
        raise LuaRuntimeError("invalid UTF-8 code")
    return value, start + count


def _position(value: int, length: int, *, allow_end=True) -> int:
    value = normalize_index(value, length)
    maximum = length + 1 if allow_end else length
    if value < 1 or value > maximum:
        raise LuaRuntimeError("position out of bounds")
    return value


def install_utf8_library(globals_table: LuaTable, vm) -> LuaTable:
    lib = LuaTable()

    def register(name, fn):
        lib.rawset(name.encode("ascii"), HostFunction(fn, f"utf8.{name}"))

    def char(*values):
        return b"".join(_encode(need_integer(v, i + 1, "char")) for i, v in enumerate(values))

    def codepoint(s, i=1, j=None, lax=False):
        s = need_bytes(s, 1, "codepoint")
        i = _position(need_integer(i, 2, "codepoint"), len(s), allow_end=True)
        j = i if j is None else _position(need_integer(j, 3, "codepoint"), len(s), allow_end=True)
        if i > j:
            return MultiValue(())
        values = []
        pos = i - 1
        limit = j - 1
        while pos <= limit and pos < len(s):
            cp, end = _decode(s, pos, bool(lax))
            values.append(cp)
            pos = end
        return MultiValue(tuple(values))

    def length_fn(s, i=1, j=-1, lax=False):
        s = need_bytes(s, 1, "len")
        try:
            i = _position(need_integer(i, 2, "len"), len(s), allow_end=True)
            j = _position(need_integer(j, 3, "len"), len(s), allow_end=True)
        except LuaRuntimeError:
            raise
        if i > j:
            return 0
        pos = i - 1
        limit = j - 1
        count = 0
        while pos <= limit and pos < len(s):
            try:
                _cp, end = _decode(s, pos, bool(lax))
            except LuaRuntimeError:
                return MultiValue((None, pos + 1))
            count += 1
            pos = end
        return count

    def offset(s, n, i=None):
        s = need_bytes(s, 1, "offset")
        n = need_integer(n, 2, "offset")
        if i is None:
            i = 1 if n >= 0 else len(s) + 1
        else:
            i = _position(need_integer(i, 3, "offset"), len(s), allow_end=True)

        if n == 0:
            if i == len(s) + 1:
                return MultiValue((i, i))
            pos = i - 1
            while pos > 0 and 0x80 <= s[pos] <= 0xBF:
                pos -= 1
            _cp, end = _decode(s, pos, True)
            if i - 1 >= end:
                raise LuaRuntimeError("initial position is a continuation byte")
            return MultiValue((pos + 1, end))

        if i <= len(s) and 0x80 <= s[i - 1] <= 0xBF:
            raise LuaRuntimeError("initial position is a continuation byte")

        if n > 0:
            pos = i - 1
            remaining = n
            while remaining > 1:
                if pos >= len(s):
                    return None
                _cp, pos = _decode(s, pos, True)
                remaining -= 1
            if pos == len(s):
                return MultiValue((len(s) + 1, len(s) + 1))
            if pos > len(s):
                return None
            _cp, end = _decode(s, pos, True)
            return MultiValue((pos + 1, end))

        pos = i - 1
        remaining = -n
        while remaining > 0:
            if pos <= 0:
                return None
            pos -= 1
            while pos > 0 and 0x80 <= s[pos] <= 0xBF:
                pos -= 1
            remaining -= 1
        _cp, end = _decode(s, pos, True)
        return MultiValue((pos + 1, end))

    def codes(s, lax=False):
        s = need_bytes(s, 1, "codes")
        lax = bool(lax)

        def iterator(state, control):
            control = need_integer(control, 2, "codes iterator")
            pos = 0 if control == 0 else control
            if pos >= len(state):
                return None
            cp, end = _decode(state, pos, lax)
            return MultiValue((pos + 1, cp))

        return MultiValue((HostFunction(iterator, "utf8.codes iterator"), s, 0))

    register("char", char)
    register("codes", codes)
    register("codepoint", codepoint)
    register("len", length_fn)
    register("offset", offset)
    lib.rawset(b"charpattern", CHARPATTERN)
    globals_table.rawset(b"utf8", lib)
    return lib
