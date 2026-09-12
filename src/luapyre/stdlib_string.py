from __future__ import annotations

import ctypes
import math
import struct
import sys

from .errors import LuaRuntimeError
from .lua_pattern import LuaPattern, PatternMatch
from .table import LuaTable
from .values import MultiValue, lua_type_name
from .vm import HostFunction
from .stdlib_support import (
    INT_MAX,
    INT_MIN,
    UINT_MASK,
    need_bytes,
    need_integer,
    normalize_index,
    number_to_bytes,
    slice_bounds,
    tostring_value,
)


def _start_index(value, length: int) -> int:
    value = normalize_index(value, length)
    if value < 1:
        value = 1
    return value - 1


def _match_values(match: PatternMatch, subject: bytes, *, whole_if_empty=True):
    captures = match.capture_values(subject)
    if captures:
        return captures
    return (subject[match.start:match.end],) if whole_if_empty else ()


def _replacement_bytes(value):
    if isinstance(value, bytes):
        return value
    if type(value) in (int, float):
        return number_to_bytes(value)
    return None


def _quote_float(value: float) -> bytes:
    if math.isnan(value):
        return b"(0/0)"
    if math.isinf(value):
        return b"1e9999" if value > 0 else b"-1e9999"
    if value == 0.0:
        return b"-0x0p+0" if math.copysign(1.0, value) < 0 else b"0x0p+0"
    text = value.hex()
    mantissa, exponent = text.split("p", 1)
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}p{exponent}".encode("ascii")


def _quote_lua(value) -> bytes:
    """Serialize one value using Lua 5.5's exact ``%q`` literal rules."""
    if value is None:
        return b"nil"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if type(value) is int:
        if value == INT_MIN:
            return f"0x{value & UINT_MASK:x}".encode("ascii")
        return str(value).encode("ascii")
    if type(value) is float:
        return _quote_float(value)
    if not isinstance(value, bytes):
        raise LuaRuntimeError("bad argument to 'format' for %q")

    out = bytearray(b'"')
    for index, byte in enumerate(value):
        if byte in (ord('"'), ord('\\'), ord('\n')):
            out.append(ord('\\'))
            out.append(byte)
        elif byte < 32 or byte == 127:
            next_byte = value[index + 1] if index + 1 < len(value) else None
            if next_byte is not None and ord('0') <= next_byte <= ord('9'):
                out.extend(f"\\{byte:03d}".encode("ascii"))
            else:
                out.extend(f"\\{byte}".encode("ascii"))
        else:
            out.append(byte)
    out.append(ord('"'))
    return bytes(out)


class _PackFormat:
    def __init__(self, fmt: bytes):
        try:
            self.text = fmt.decode("ascii")
        except UnicodeDecodeError:
            raise LuaRuntimeError("invalid format option") from None
        self.index = 0
        self.endian = sys.byteorder
        self.maxalign = 1

    @staticmethod
    def _number(text: str, index: int):
        start = index
        while index < len(text) and text[index].isdigit():
            index += 1
        if start == index:
            return None, index
        return int(text[start:index]), index

    def _raw_option(self):
        text = self.text
        while self.index < len(text) and text[self.index] == " ":
            self.index += 1
        if self.index >= len(text):
            return None
        op = text[self.index]
        self.index += 1
        if op in "<>=":
            self.endian = "little" if op == "<" else "big" if op == ">" else sys.byteorder
            return ("config", op, 0, 1)
        if op == "!":
            n, self.index = self._number(text, self.index)
            n = ctypes.sizeof(ctypes.c_long) if n is None else n
            if n <= 0 or n > 16:
                raise LuaRuntimeError(f"integral size ({n}) out of limits [1,16]")
            if n & (n - 1):
                raise LuaRuntimeError("format asks for alignment not power of 2")
            self.maxalign = n
            return ("config", op, 0, 1)
        sizes = {
            "b": (1, True), "B": (1, False),
            "h": (ctypes.sizeof(ctypes.c_short), True),
            "H": (ctypes.sizeof(ctypes.c_ushort), False),
            "l": (ctypes.sizeof(ctypes.c_long), True),
            "L": (ctypes.sizeof(ctypes.c_ulong), False),
            "j": (8, True), "J": (8, False),
            "T": (ctypes.sizeof(ctypes.c_size_t), False),
        }
        if op in sizes:
            size, signed = sizes[op]
            return ("int", signed, size, min(size, self.maxalign))
        if op in "iI":
            size, self.index = self._number(text, self.index)
            size = ctypes.sizeof(ctypes.c_int) if size is None else size
            if not 1 <= size <= 16:
                raise LuaRuntimeError(f"integral size ({size}) out of limits [1,16]")
            alignment = min(size, self.maxalign)
            if alignment & (alignment - 1):
                raise LuaRuntimeError("format asks for alignment not power of 2")
            return ("int", op == "i", size, alignment)
        if op in "fdn":
            size = 4 if op == "f" else 8
            return ("float", op, size, min(size, self.maxalign))
        if op == "c":
            size, self.index = self._number(text, self.index)
            if size is None:
                raise LuaRuntimeError("missing size for format option 'c'")
            return ("fixed", op, size, 1)
        if op == "z":
            return ("zero", op, 0, 1)
        if op == "s":
            size, self.index = self._number(text, self.index)
            size = ctypes.sizeof(ctypes.c_size_t) if size is None else size
            if not 1 <= size <= 16:
                raise LuaRuntimeError(f"integral size ({size}) out of limits [1,16]")
            alignment = min(size, self.maxalign)
            if alignment & (alignment - 1):
                raise LuaRuntimeError("format asks for alignment not power of 2")
            return ("sized", op, size, alignment)
        if op == "x":
            return ("padding", op, 1, 1)
        if op == "X":
            following = self._raw_option()
            while following is not None and following[0] == "config":
                following = self._raw_option()
            if following is None or following[0] in ("zero", "padding"):
                raise LuaRuntimeError("invalid next option for option 'X'")
            return ("align", op, 0, following[3])
        raise LuaRuntimeError(f"invalid format option '{op}'")

    def options(self):
        while True:
            option = self._raw_option()
            if option is None:
                return
            if option[0] != "config":
                yield option, self.endian


def _padding(offset: int, alignment: int) -> int:
    return (-offset) % max(1, alignment)


def _pack(fmt: bytes, values):
    parser = _PackFormat(fmt)
    out = bytearray()
    value_index = 0
    for option, endian in parser.options():
        kind, detail, size, alignment = option
        pad = _padding(len(out), alignment)
        out.extend(b"\0" * pad)
        if kind == "padding":
            out.append(0)
            continue
        if kind == "align":
            continue
        if value_index >= len(values):
            raise LuaRuntimeError("bad argument to 'pack' (value expected)")
        value = values[value_index]
        value_index += 1
        if kind == "int":
            integer = need_integer(value, value_index + 1, "pack")
            signed = bool(detail)
            if signed:
                minimum = -(1 << (size * 8 - 1))
                maximum = (1 << (size * 8 - 1)) - 1
                if not minimum <= integer <= maximum:
                    raise LuaRuntimeError("integer overflow")
                out.extend(int(integer).to_bytes(size, endian, signed=True))
            else:
                unsigned = integer & UINT_MASK
                if unsigned >= (1 << (size * 8)):
                    raise LuaRuntimeError("unsigned overflow")
                out.extend(unsigned.to_bytes(size, endian, signed=False))
        elif kind == "float":
            if type(value) not in (int, float):
                raise LuaRuntimeError("number expected")
            prefix = "<" if endian == "little" else ">"
            code = "f" if size == 4 else "d"
            out.extend(struct.pack(prefix + code, float(value)))
        elif kind == "fixed":
            data = need_bytes(value, value_index + 1, "pack")
            if len(data) > size:
                raise LuaRuntimeError("string longer than given size")
            out.extend(data)
            out.extend(b"\0" * (size - len(data)))
        elif kind == "zero":
            data = need_bytes(value, value_index + 1, "pack")
            if b"\0" in data:
                raise LuaRuntimeError("string contains zeros")
            out.extend(data)
            out.append(0)
        elif kind == "sized":
            data = need_bytes(value, value_index + 1, "pack")
            if len(data) >= (1 << (size * 8)):
                raise LuaRuntimeError("string length does not fit in given size")
            out.extend(len(data).to_bytes(size, endian, signed=False))
            out.extend(data)
    return bytes(out)


def _unpack_integer(data: bytes, offset: int, size: int, endian: str, signed: bool) -> int:
    """Mirror Lua 5.5 ``unpackint`` using a 64-bit lua_Unsigned accumulator."""
    end = offset + size
    if end > len(data):
        raise LuaRuntimeError("data string too short")

    if size <= 8:
        raw = data[offset:end]
        unsigned = int.from_bytes(raw, endian, signed=False)
        if signed and size < 8 and unsigned & (1 << (size * 8 - 1)):
            unsigned |= UINT_MASK ^ ((1 << (size * 8)) - 1)
    else:
        if endian == "little":
            low = data[offset:offset + 8]
            extra = data[offset + 8:end]
        else:
            low = data[end - 8:end]
            extra = data[offset:end - 8]
        unsigned = int.from_bytes(low, endian, signed=False)
        signed_value = unsigned if unsigned <= INT_MAX else unsigned - (1 << 64)
        extension = 0xFF if signed and signed_value < 0 else 0x00
        if any(byte != extension for byte in extra):
            raise LuaRuntimeError(f"{size}-byte integer does not fit into Lua Integer")

    unsigned &= UINT_MASK
    return unsigned if unsigned <= INT_MAX else unsigned - (1 << 64)


def _unpack(fmt: bytes, data: bytes, position: int):
    parser = _PackFormat(fmt)
    offset = position
    values = []
    for option, endian in parser.options():
        kind, detail, size, alignment = option
        offset += _padding(offset, alignment)
        if kind == "padding":
            offset += 1
            if offset > len(data):
                raise LuaRuntimeError("data string too short")
            continue
        if kind == "align":
            continue
        if kind == "int":
            values.append(_unpack_integer(data, offset, size, endian, bool(detail)))
            offset += size
        elif kind == "float":
            end = offset + size
            if end > len(data):
                raise LuaRuntimeError("data string too short")
            prefix = "<" if endian == "little" else ">"
            code = "f" if size == 4 else "d"
            values.append(struct.unpack(prefix + code, data[offset:end])[0])
            offset = end
        elif kind == "fixed":
            end = offset + size
            if end > len(data):
                raise LuaRuntimeError("data string too short")
            values.append(data[offset:end])
            offset = end
        elif kind == "zero":
            end = data.find(b"\0", offset)
            if end < 0:
                raise LuaRuntimeError("unfinished string for format 'z'")
            values.append(data[offset:end])
            offset = end + 1
        elif kind == "sized":
            prefix_end = offset + size
            if prefix_end > len(data):
                raise LuaRuntimeError("data string too short")
            length = _unpack_integer(data, offset, size, endian, False) & UINT_MASK
            if length > len(data) - prefix_end:
                raise LuaRuntimeError("data string too short")
            end = prefix_end + length
            values.append(data[prefix_end:end])
            offset = end
    values.append(offset + 1)
    return MultiValue(tuple(values))


def _packsize(fmt: bytes):
    parser = _PackFormat(fmt)
    offset = 0
    for option, _endian in parser.options():
        kind, _detail, size, alignment = option
        if kind in ("zero", "sized"):
            raise LuaRuntimeError("variable-length format")
        offset += _padding(offset, alignment)
        if kind in ("int", "float", "fixed", "padding"):
            offset += size
    return offset


def install_string_library(globals_table: LuaTable, vm) -> LuaTable:
    lib = LuaTable()

    def register(name, fn):
        lib.rawset(name.encode("ascii"), HostFunction(fn, f"string.{name}"))

    def byte_fn(s, i=1, j=None):
        s = need_bytes(s, 1, "byte")
        i = need_integer(i, 2, "byte")
        j = i if j is None else need_integer(j, 3, "byte")
        i, j = slice_bounds(i, j, len(s))
        if i > j:
            return MultiValue(())
        return MultiValue(tuple(s[index - 1] for index in range(i, j + 1)))

    def char_fn(*values):
        out = bytearray()
        for index, value in enumerate(values, 1):
            integer = need_integer(value, index, "char")
            if not 0 <= integer <= 255:
                raise LuaRuntimeError(f"bad argument #{index} to 'char' (value out of range)")
            out.append(integer)
        return bytes(out)

    def find(s, pattern, init=1, plain=False):
        s = need_bytes(s, 1, "find")
        pattern = need_bytes(pattern, 2, "find")
        start = _start_index(need_integer(init, 3, "find"), len(s))
        if start > len(s):
            return None
        if plain:
            found = s.find(pattern, start)
            if found < 0:
                return None
            return MultiValue((found + 1, found + len(pattern)))
        match = LuaPattern(pattern).search(s, start)
        if match is None:
            return None
        return MultiValue((match.start + 1, match.end, *match.capture_values(s)))

    def format_fn(formatstring, *values):
        fmt = need_bytes(formatstring, 1, "format")
        out = bytearray()
        i = 0
        arg = 0
        while i < len(fmt):
            if fmt[i] != ord("%"):
                out.append(fmt[i])
                i += 1
                continue
            if i + 1 < len(fmt) and fmt[i + 1] == ord("%"):
                out.append(ord("%"))
                i += 2
                continue
            i += 1
            start = i
            while i < len(fmt) and fmt[i] in b"-+#0 ":
                i += 1
            flags = fmt[start:i].decode("ascii")
            width_start = i
            while i < len(fmt) and 48 <= fmt[i] <= 57 and i - width_start < 2:
                i += 1
            width = fmt[width_start:i].decode("ascii")
            precision = ""
            if i < len(fmt) and fmt[i] == ord("."):
                pstart = i
                i += 1
                digits = i
                while i < len(fmt) and 48 <= fmt[i] <= 57 and i - digits < 2:
                    i += 1
                precision = fmt[pstart:i].decode("ascii")
            if i >= len(fmt):
                raise LuaRuntimeError("invalid conversion specification")
            spec = chr(fmt[i])
            i += 1
            if arg >= len(values):
                raise LuaRuntimeError("bad argument to 'format' (no value)")
            value = values[arg]
            arg += 1
            if spec == "q":
                if flags or width or precision:
                    raise LuaRuntimeError("specifier '%q' cannot have modifiers")
                rendered = _quote_lua(value)
            elif spec == "s":
                rendered = tostring_value(vm, value)
                if precision:
                    limit = int(precision[1:] or "0")
                    rendered = rendered[:limit]
                if width:
                    pad = max(0, int(width) - len(rendered))
                    fill = b" " * pad
                    rendered = rendered + fill if "-" in flags else fill + rendered
            elif spec == "c":
                integer = need_integer(value, arg + 1, "format")
                if not 0 <= integer <= 255:
                    raise LuaRuntimeError("value out of range")
                rendered = bytes((integer,))
            elif spec in "diouxX":
                integer = need_integer(value, arg + 1, "format")
                if spec == "u":
                    integer &= UINT_MASK
                    py_spec = "d"
                else:
                    py_spec = spec
                token = "%" + flags + width + precision + py_spec
                rendered = (token % integer).encode("ascii")
            elif spec in "eEfFgG":
                if type(value) not in (int, float):
                    raise LuaRuntimeError("number expected")
                token = "%" + flags + width + precision + spec
                rendered = (token % float(value)).encode("ascii")
            elif spec in "aA":
                if type(value) not in (int, float):
                    raise LuaRuntimeError("number expected")
                rendered = float(value).hex().encode("ascii")
                if spec == "A":
                    rendered = rendered.upper()
            elif spec == "p":
                if value is None or type(value) in (bool, int, float):
                    rendered = b"(null)"
                else:
                    rendered = f"0x{id(value):x}".encode("ascii")
            else:
                raise LuaRuntimeError(f"invalid conversion '%{spec}'")
            out.extend(rendered)
        return bytes(out)

    def match_fn(s, pattern, init=1):
        s = need_bytes(s, 1, "match")
        pattern = need_bytes(pattern, 2, "match")
        start = _start_index(need_integer(init, 3, "match"), len(s))
        if start > len(s):
            return None
        match = LuaPattern(pattern).search(s, start)
        if match is None:
            return None
        return MultiValue(_match_values(match, s))

    def gmatch(s, pattern, init=1):
        s = need_bytes(s, 1, "gmatch")
        pattern = need_bytes(pattern, 2, "gmatch")
        position = _start_index(need_integer(init, 3, "gmatch"), len(s))
        matcher = LuaPattern(pattern)
        finished = False

        def iterator(*_ignored):
            nonlocal position, finished
            if finished or position > len(s):
                return None
            match = matcher.search(s, position, gmatch=True)
            if match is None:
                finished = True
                return None
            if match.end == match.start:
                position = match.end + 1
            else:
                position = match.end
            if match.end == len(s) and match.start == match.end:
                finished = True
            return MultiValue(_match_values(match, s))

        return HostFunction(iterator, "string.gmatch iterator")

    def gsub(s, pattern, repl, n=None):
        s = need_bytes(s, 1, "gsub")
        pattern = need_bytes(pattern, 2, "gsub")
        limit = (len(s) + 1) if n is None else max(0, need_integer(n, 4, "gsub"))
        matcher = LuaPattern(pattern)
        output = bytearray()
        search_pos = 0
        copied = 0
        count = 0
        changed = False
        while count < limit and search_pos <= len(s):
            match = matcher.search(s, search_pos)
            if match is None:
                break
            output.extend(s[copied:match.start])
            captures = match.capture_values(s)
            args = captures if captures else (s[match.start:match.end],)
            replacement = None
            if isinstance(repl, bytes):
                built = bytearray()
                index = 0
                while index < len(repl):
                    if repl[index] != ord("%"):
                        built.append(repl[index])
                        index += 1
                        continue
                    if index + 1 >= len(repl):
                        raise LuaRuntimeError("invalid use of '%' in replacement string")
                    code = repl[index + 1]
                    if code == ord("%"):
                        built.append(ord("%"))
                    elif code == ord("0"):
                        built.extend(s[match.start:match.end])
                    elif 49 <= code <= 57:
                        capture_index = code - 49
                        if capture_index < len(captures):
                            value = captures[capture_index]
                        elif capture_index == 0 and not captures:
                            value = s[match.start:match.end]
                        else:
                            raise LuaRuntimeError(
                                f"invalid capture index %{chr(code)}"
                            )
                        built.extend(_replacement_bytes(value) or b"")
                    else:
                        raise LuaRuntimeError("invalid use of '%' in replacement string")
                    index += 2
                replacement = bytes(built)
            elif isinstance(repl, LuaTable):
                key = args[0]
                replacement = vm.index_sync(repl, key)
            else:
                try:
                    results = vm.call_sync(repl, args)
                except LuaRuntimeError:
                    raise
                replacement = results[0] if results else None
            rendered = _replacement_bytes(replacement)
            if rendered is None:
                if replacement is None or replacement is False:
                    rendered = s[match.start:match.end]
                else:
                    raise LuaRuntimeError(
                        f"invalid replacement value (a {lua_type_name(replacement)})"
                    )
            else:
                changed = True
            output.extend(rendered)
            count += 1
            copied = match.end
            if match.end == match.start:
                if match.end < len(s):
                    output.append(s[match.end])
                    copied = match.end + 1
                    search_pos = match.end + 1
                else:
                    search_pos = len(s) + 1
            else:
                search_pos = match.end
        output.extend(s[copied:])
        return MultiValue((bytes(output) if changed else s, count))

    def rep(s, n, sep=b""):
        s = need_bytes(s, 1, "rep")
        n = need_integer(n, 2, "rep")
        sep = need_bytes(sep, 3, "rep")
        if n <= 0:
            return b""
        total = len(s) * n + len(sep) * (n - 1)
        if total > 64 * 1024 * 1024:
            raise LuaRuntimeError("resulting string too large")
        return sep.join([s] * n)

    def sub(s, i, j=-1):
        s = need_bytes(s, 1, "sub")
        i = need_integer(i, 2, "sub")
        j = need_integer(j, 3, "sub")
        i, j = slice_bounds(i, j, len(s))
        return b"" if i > j else s[i - 1:j]

    def pack(fmt, *values):
        return _pack(need_bytes(fmt, 1, "pack"), values)

    def packsize(fmt):
        return _packsize(need_bytes(fmt, 1, "packsize"))

    def unpack(fmt, s, pos=1):
        fmt = need_bytes(fmt, 1, "unpack")
        s = need_bytes(s, 2, "unpack")
        pos = normalize_index(need_integer(pos, 3, "unpack"), len(s))
        if pos < 1 or pos > len(s) + 1:
            raise LuaRuntimeError("initial position out of string")
        return _unpack(fmt, s, pos - 1)

    for name, fn in (
        ("byte", byte_fn),
        ("char", char_fn),
        ("find", find),
        ("format", format_fn),
        ("gmatch", gmatch),
        ("gsub", gsub),
        ("len", lambda s: len(need_bytes(s, 1, "len"))),
        ("lower", lambda s: need_bytes(s, 1, "lower").lower()),
        ("match", match_fn),
        ("pack", pack),
        ("packsize", packsize),
        ("rep", rep),
        ("reverse", lambda s: need_bytes(s, 1, "reverse")[::-1]),
        ("sub", sub),
        ("unpack", unpack),
        ("upper", lambda s: need_bytes(s, 1, "upper").upper()),
    ):
        register(name, fn)

    string_mt = LuaTable()
    string_mt.rawset(b"__index", lib)
    vm.type_metatables[b"string"] = string_mt
    globals_table.rawset(b"string", lib)
    return lib
