from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .capabilities import RuntimeCapabilities
from .errors import LuaRuntimeError
from .stdlib_support import lua_c_function, need_bytes, need_integer, number_to_bytes
from .table import LuaTable
from .values import MultiValue
from .vm import HostFunction


_READ_NUMBER = re.compile(
    rb"[+-]?(?:0[xX](?:[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?|\.[0-9a-fA-F]+)(?:[pP][+-]?[0-9]+)?|(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
)


@dataclass(slots=True)
class _StreamState:
    data: bytearray
    position: int = 0
    readable: bool = True
    writable: bool = True
    closed: bool = False
    append: bool = False
    name: str | None = None
    sink: object = None
    closable: bool = True
    flush_error: bool = False
    buffer_mode: bytes = b"no"


@dataclass(slots=True, eq=False)
class _LuaFile:
    metatable: LuaTable

    def __repr__(self):
        return f"file: 0x{id(self):x}"


def install_io_library(globals_table: LuaTable, capabilities: RuntimeCapabilities) -> LuaTable:
    """Install streams backed only by memory and explicit read capabilities."""
    library = LuaTable()
    stream_states: dict[int, _StreamState] = {}

    def check(state: _StreamState):
        if state.closed:
            raise LuaRuntimeError("attempt to use a closed file")

    def sync(state: _StreamState, *, force=False):
        if state.name is not None and state.writable and (
            force or state.buffer_mode == b"no"
        ):
            capabilities.write_virtual_file(state.name, bytes(state.data))

    def make_stream(state: _StreamState) -> _LuaFile:
        methods = LuaTable()
        metatable = LuaTable()
        metatable.rawset(b"__index", methods)
        metatable.rawset(b"__name", b"FILE*")
        stream = _LuaFile(metatable)
        stream_states[id(stream)] = state

        def write(_self, *values):
            check(state)
            if not state.writable:
                return MultiValue((None, b"file is not writable", 9))
            chunks = []
            for value in values:
                if isinstance(value, bytes):
                    chunks.append(value)
                elif type(value) in (int, float):
                    chunks.append(number_to_bytes(value))
                else:
                    raise LuaRuntimeError("bad argument to 'write' (string expected)")
            payload = b"".join(chunks)
            if state.sink is not None:
                state.sink(payload)
            else:
                if state.append:
                    state.position = len(state.data)
                end = state.position + len(payload)
                if end > len(state.data):
                    state.data.extend(b"\0" * (end - len(state.data)))
                state.data[state.position:end] = payload
                state.position = end
                sync(state, force=state.buffer_mode == b"line" and b"\n" in payload)
            return stream

        def read(_self, *formats):
            check(state)
            if not state.readable:
                return MultiValue((None, b"file is not readable", 9))
            if state.name is not None and not state.writable:
                latest = capabilities.virtual_files.get(state.name)
                if latest is not None:
                    state.data[:] = latest
            formats = formats or (b"l",)
            results = []
            for fmt in formats:
                if type(fmt) in (int, float):
                    count = need_integer(fmt, name="read")
                    if count == 0:
                        results.append(b"" if state.position < len(state.data) else None)
                        continue
                    end = min(len(state.data), state.position + count)
                    value = bytes(state.data[state.position:end])
                    state.position = end
                    results.append(value if value else None)
                    continue
                fmt = need_bytes(fmt, name="read")
                if fmt.startswith(b"*"):
                    fmt = fmt[1:]
                if fmt in (b"a", b"all"):
                    value = bytes(state.data[state.position:])
                    state.position = len(state.data)
                    results.append(value)
                elif fmt in (b"l", b"L"):
                    if state.position >= len(state.data):
                        results.append(None)
                    else:
                        newline = state.data.find(b"\n", state.position)
                        end = len(state.data) if newline < 0 else newline + 1
                        value = bytes(state.data[state.position:end])
                        state.position = end
                        results.append(value if fmt == b"L" else value.rstrip(b"\n").rstrip(b"\r"))
                elif fmt == b"n":
                    tail = bytes(state.data[state.position:])
                    stripped = tail.lstrip()
                    skipped = len(tail) - len(stripped)
                    from .values import parse_lua_number
                    match = _READ_NUMBER.match(stripped)
                    token = match.group(0) if match is not None else b""
                    consumed = len(token)
                    # scanf-style reads consume a malformed numeric prefix.
                    if token.endswith(b".") and stripped[consumed:consumed + 1] in (b"e", b"E"):
                        consumed += 1
                        if stripped[consumed:consumed + 1] in (b"+", b"-"):
                            consumed += 1
                        while stripped[consumed:consumed + 1].isdigit():
                            consumed += 1
                        value = None
                    elif re.match(rb"[+-]?0[xX]", stripped) and not re.match(
                        rb"[+-]?0[xX][0-9a-fA-F.]", stripped
                    ):
                        consumed = 3 if stripped[:1] in (b"+", b"-") else 2
                        value = None
                    elif len(token) > 200:
                        consumed = 200
                        value = None
                    else:
                        value = parse_lua_number(token)
                        if isinstance(value, float) and math.isinf(value) and b"e" not in token.lower():
                            consumed = min(200, len(token))
                            value = None
                    if value is None:
                        if consumed == 0 and stripped[:1] in (b"+", b"-", b"."):
                            consumed = 1
                        state.position += skipped + consumed
                        results.append(None)
                    else:
                        state.position += skipped + consumed
                        results.append(value)
                else:
                    raise LuaRuntimeError("invalid format")
            return MultiValue(tuple(results))

        def seek(_self, whence=b"cur", offset=0):
            check(state)
            whence = need_bytes(whence, 1, "seek")
            offset = need_integer(offset, 2, "seek")
            if state.name is not None and state.readable and not state.writable:
                latest = capabilities.virtual_files.get(state.name)
                if latest is not None:
                    state.data[:] = latest
            base = {b"set": 0, b"cur": state.position, b"end": len(state.data)}.get(whence)
            if base is None:
                raise LuaRuntimeError("invalid option")
            position = base + offset
            if position < 0:
                return MultiValue((None, b"invalid offset"))
            state.position = position
            return position

        def close(_self=None):
            if _self is not stream:
                raise LuaRuntimeError("bad argument #1 to 'close' (got no value)")
            check(state)
            if not state.closable:
                return MultiValue((None, b"cannot close standard file"))
            sync(state, force=True)
            state.closed = True
            return True

        def close_metamethod(_self, _error=None):
            if _self is not stream:
                raise LuaRuntimeError("bad argument #1 to 'close' (file expected)")
            if state.closed:
                return None
            return close(_self)

        def tostring_file(_self):
            if _self is not stream:
                raise LuaRuntimeError("bad argument #1 to 'tostring' (file expected)")
            return b"file (closed)" if state.closed else f"file (0x{id(stream):x})".encode("ascii")

        def flush(_self=None):
            check(state)
            if state.flush_error:
                return MultiValue((None, b"no space left on device", 28))
            sync(state, force=True)
            return True

        def setvbuf(_self, mode, _size=None):
            check(state)
            mode = need_bytes(mode, 1, "setvbuf")
            if mode not in (b"no", b"full", b"line"):
                raise LuaRuntimeError("invalid option")
            state.buffer_mode = mode
            return True

        def lines(_self, *formats):
            formats = formats or (b"l",)

            def iterator():
                values = read(stream, *formats).values
                if not values or values[0] is None:
                    if len(values) > 1 and values[1] is not None:
                        message = values[1]
                        if isinstance(message, bytes):
                            message = message.decode("utf-8", "replace")
                        raise LuaRuntimeError(str(message))
                    return None
                return MultiValue(tuple(values))

            return HostFunction(iterator, "file:lines", max_args=0)

        for name, fn in (
            (b"write", write), (b"read", read), (b"seek", seek),
            (b"close", close), (b"flush", flush), (b"lines", lines),
            (b"setvbuf", setvbuf),
        ):
            methods.rawset(name, HostFunction(fn, f"file:{name.decode()}"))
        metatable.rawset(b"__close", HostFunction(close_metamethod, "file:__close"))
        metatable.rawset(b"__tostring", HostFunction(tostring_file, "file:__tostring"))
        return stream

    def stream_method(stream: _LuaFile, name: bytes):
        return stream.metatable.rawget(b"__index").rawget(name)

    stdout = make_stream(_StreamState(bytearray(), readable=False, sink=lambda data: capabilities.output_sink(data), closable=False))
    stderr = make_stream(_StreamState(bytearray(), readable=False, sink=lambda data: capabilities.warning_sink(data), closable=False))
    stdin = make_stream(_StreamState(bytearray(), writable=False, closable=False))
    current_input = stdin
    current_output = stdout

    def open_file(name, mode=b"r"):
        name_b = need_bytes(name, 1, "open")
        mode = need_bytes(mode, 2, "open")
        if mode not in (b"r", b"w", b"a", b"r+", b"w+", b"a+",
                        b"rb", b"wb", b"ab", b"r+b", b"w+b", b"a+b"):
            raise LuaRuntimeError("invalid mode")
        normalized = mode.replace(b"b", b"")
        filename = name_b.decode("utf-8", "surrogateescape")
        if filename == "/dev/null" and normalized[:1] in (b"w", b"a"):
            return make_stream(_StreamState(bytearray(), readable=False, sink=lambda _data: None))
        if filename == "/dev/full" and normalized[:1] in (b"w", b"a"):
            return make_stream(_StreamState(bytearray(), readable=False, flush_error=True))
        if filename.startswith(("/", "\\")) or ".." in filename.replace("\\", "/").split("/"):
            return MultiValue((None, b"file is outside the in-memory sandbox", 13))
        readable = b"r" in normalized or b"+" in normalized
        writable = normalized[:1] in (b"w", b"a") or b"+" in normalized
        if normalized.startswith(b"w"):
            data = b""
        else:
            data, error = capabilities.read_file(filename)
            if data is None:
                if normalized.startswith(b"a"):
                    data = b""
                else:
                    return MultiValue((None, error, 2))
        if writable:
            capabilities.write_virtual_file(filename, data)
        state = _StreamState(bytearray(data), readable=readable, writable=writable,
                             append=normalized.startswith(b"a"), name=filename)
        if state.append:
            state.position = len(state.data)
        return make_stream(state)

    temporary_counter = 0

    def tmpfile():
        nonlocal temporary_counter
        temporary_counter += 1
        name = f"@luapyre-tmpfile/{temporary_counter}"
        capabilities.write_virtual_file(name, b"")
        return make_stream(_StreamState(bytearray(), name=name))

    def io_type(value):
        if not isinstance(value, _LuaFile):
            return None
        state = stream_states.get(id(value))
        if state is None:
            return None
        return b"closed file" if state.closed else b"file"

    def close_file(stream=None):
        stream = current_output if stream is None else stream
        if not isinstance(stream, _LuaFile) or id(stream) not in stream_states:
            raise LuaRuntimeError("bad argument #1 to 'close' (file expected)")
        return stream_method(stream, b"close").fn(stream)

    def input_file(value=None):
        nonlocal current_input
        if value is None:
            return current_input
        opened = open_file(value) if isinstance(value, bytes) else value
        if isinstance(opened, MultiValue):
            error = opened.values[1] if len(opened.values) > 1 else b"cannot open file"
            raise LuaRuntimeError(error.decode("utf-8", "replace"))
        current_input = opened
        return current_input

    def output_file(value=None):
        nonlocal current_output
        if value is None:
            return current_output
        opened = open_file(value, b"w") if isinstance(value, bytes) else value
        if isinstance(opened, MultiValue):
            error = opened.values[1] if len(opened.values) > 1 else b"cannot open file"
            raise LuaRuntimeError(error.decode("utf-8", "replace"))
        current_output = opened
        return current_output

    library.rawset(b"stdin", stdin)
    library.rawset(b"stdout", stdout)
    library.rawset(b"stderr", stderr)
    def lines_file(name=None, *formats):
        if len(formats) > 250:
            raise LuaRuntimeError("too many arguments")
        if name is None:
            return stream_method(current_input, b"lines").fn(current_input, *formats)
        opened = open_file(name)
        if isinstance(opened, MultiValue):
            error = opened.values[1] if len(opened.values) > 1 else b"cannot open file"
            raise LuaRuntimeError(error.decode("utf-8", "replace"))
        iterator = stream_method(opened, b"lines").fn(opened, *formats)

        def owned_iterator():
            state = stream_states[id(opened)]
            if state.closed:
                raise LuaRuntimeError("file is already closed")
            result = iterator.fn()
            first = result.values[0] if isinstance(result, MultiValue) and result.values else result
            if first is None:
                stream_method(opened, b"close").fn(opened)
            return result

        return MultiValue((
            HostFunction(owned_iterator, "io.lines iterator", max_args=0),
            None,
            None,
            opened,
        ))

    def read_current(*formats):
        if stream_states[id(current_input)].closed:
            raise LuaRuntimeError("standard input file is closed")
        return stream_method(current_input, b"read").fn(current_input, *formats)

    def write_current(*values):
        if stream_states[id(current_output)].closed:
            raise LuaRuntimeError("standard output file is closed")
        return stream_method(current_output, b"write").fn(current_output, *values)

    for name, fn in (
        ("open", open_file), ("tmpfile", tmpfile), ("type", io_type),
        ("close", close_file), ("input", input_file), ("output", output_file),
        ("read", read_current),
        ("write", write_current),
        ("flush", lambda: stream_method(current_output, b"flush").fn(current_output)),
        ("lines", lines_file),
    ):
        library.rawset(name.encode("ascii"), lua_c_function(fn, f"io.{name}"))
    globals_table.rawset(b"io", library)
    return library
