from __future__ import annotations

from .errors import LuaRuntimeError
from .table import LuaTable
from .values import MultiValue
from .vm import HostFunction
from .stdlib_support import need_bytes, need_integer


CHARPATTERN = b"[\x00-\x7f\xc2-\xfd][\x80-\xbf]*"


def _is_cont(byte: int) -> bool:
    return (byte & 0xC0) == 0x80


def _byte_at(data: bytes, index: int) -> int:
    """Return Lua's implicit trailing NUL when indexing at string end."""
    return data[index] if 0 <= index < len(data) else 0


def _u_posrelat(value: int, length: int) -> int:
    """Mirror Lua 5.5's utf8-library relative-position helper."""
    if value >= 0:
        return value
    if -value > length:
        return 0
    return length + value + 1


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
        if not _is_cont(byte):
            raise LuaRuntimeError("invalid UTF-8 code")
        value = (value << 6) | (byte & 0x3F)
    minimum = (0, 0, 0x80, 0x800, 0x10000, 0x200000, 0x4000000)[count]
    if value < minimum:
        raise LuaRuntimeError("invalid UTF-8 code")
    if not lax and (value > 0x10FFFF or 0xD800 <= value <= 0xDFFF):
        raise LuaRuntimeError("invalid UTF-8 code")
    return value, start + count


def install_utf8_library(globals_table: LuaTable, vm) -> LuaTable:
    lib = LuaTable()

    def register(name, fn):
        lib.rawset(name.encode("ascii"), HostFunction(fn, f"utf8.{name}"))

    def char(*values):
        return b"".join(_encode(need_integer(v, i + 1, "char")) for i, v in enumerate(values))

    def codepoint(s, i=1, j=None, lax=False):
        s = need_bytes(s, 1, "codepoint")
        length = len(s)
        posi = _u_posrelat(need_integer(i, 2, "codepoint"), length)
        raw_j = posi if j is None else need_integer(j, 3, "codepoint")
        pose = _u_posrelat(raw_j, length)
        if posi < 1:
            raise LuaRuntimeError("initial position out of bounds")
        if pose > length:
            raise LuaRuntimeError("final position out of bounds")
        if posi > pose:
            return MultiValue(())
        values = []
        pos = posi - 1
        limit = pose
        while pos < limit:
            cp, pos = _decode(s, pos, bool(lax))
            values.append(cp)
        return MultiValue(tuple(values))

    def length_fn(s, i=1, j=-1, lax=False):
        s = need_bytes(s, 1, "len")
        length = len(s)
        posi = _u_posrelat(need_integer(i, 2, "len"), length)
        posj = _u_posrelat(need_integer(j, 3, "len"), length)
        if posi < 1:
            raise LuaRuntimeError("initial position out of bounds")
        posi -= 1
        if posi > length:
            raise LuaRuntimeError("initial position out of bounds")
        posj -= 1
        if posj >= length:
            raise LuaRuntimeError("final position out of bounds")
        count = 0
        while posi <= posj:
            try:
                _cp, posi = _decode(s, posi, bool(lax))
            except LuaRuntimeError:
                return MultiValue((None, posi + 1))
            count += 1
        return count

    def offset(s, n, i=None):
        s = need_bytes(s, 1, "offset")
        n = need_integer(n, 2, "offset")
        length = len(s)
        default_i = 1 if n >= 0 else length + 1
        raw_i = default_i if i is None else need_integer(i, 3, "offset")
        posi = _u_posrelat(raw_i, length)
        if posi < 1:
            raise LuaRuntimeError("position out of bounds")
        pos = posi - 1
        if pos > length:
            raise LuaRuntimeError("position out of bounds")

        if n == 0:
            while pos > 0 and _is_cont(_byte_at(s, pos)):
                pos -= 1
        else:
            if _is_cont(_byte_at(s, pos)):
                raise LuaRuntimeError("initial position is a continuation byte")
            if n < 0:
                while n < 0 and pos > 0:
                    pos -= 1
                    while pos > 0 and _is_cont(_byte_at(s, pos)):
                        pos -= 1
                    n += 1
            else:
                n -= 1
                while n > 0 and pos < length:
                    pos += 1
                    while _is_cont(_byte_at(s, pos)):
                        pos += 1
                    n -= 1

        if n != 0:
            return None

        initial = pos + 1
        byte = _byte_at(s, pos)
        if byte & 0x80:
            if _is_cont(byte):
                raise LuaRuntimeError("initial position is a continuation byte")
            while _is_cont(_byte_at(s, pos + 1)):
                pos += 1
        return MultiValue((initial, pos + 1))

    def codes(s, lax=False):
        s = need_bytes(s, 1, "codes")
        lax = bool(lax)
        if s and _is_cont(s[0]):
            raise LuaRuntimeError("invalid UTF-8 code")

        def iterator(state, control):
            control = need_integer(control, 2, "codes iterator")
            if control < 0:
                return None
            pos = control
            if pos < len(state):
                while pos < len(state) and _is_cont(state[pos]):
                    pos += 1
            if pos >= len(state):
                return None
            cp, end = _decode(state, pos, lax)
            if end < len(state) and _is_cont(state[end]):
                raise LuaRuntimeError("invalid UTF-8 code")
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
