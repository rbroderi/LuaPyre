from __future__ import annotations

import re

from .binary_chunks import NATIVE_MAGIC, PUC_MAGIC, fresh_loaded_closure, load_native_chunk
from .bytecode import Closure
from .diagnostics import chunk_id, error_value, where_from_frames
from .errors import LuaPyreError, LuaQuotaError, LuaRaisedError, LuaRuntimeError
from .native_debug_chunks import DEBUG_NATIVE_MAGIC, dump_debug_chunk, load_debug_chunk
from .parser import Parser
from .puc55 import load_puc55_chunk
from .source_compiler import SourceCompiler
from .values import MultiValue, truthy
from .vm import HostFunction


_LINE_RE = re.compile(r"(?:at )?line\s+(\d+)")


def _as_bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return value


def _syntax_message(error: Exception, source_name) -> bytes:
    message = str(error)
    line = getattr(error, "line", None)
    if line is None:
        match = _LINE_RE.search(message)
        line = int(match.group(1)) if match else 1
    return chunk_id(source_name) + b":" + str(line).encode("ascii") + b": " + message.encode("utf-8", "replace")


def install_diagnostic_stdlib(globals_table, vm) -> None:
    """Replace the small base-library surface whose semantics need Lua frames."""

    def put(name, fn):
        globals_table.rawset(name.encode("ascii"), HostFunction(fn, name))

    def lua_error(value=None, level=1):
        if type(level) is not int:
            if type(level) is float and level.is_integer():
                level = int(level)
            else:
                raise LuaRuntimeError("bad argument #2 to 'error' (number has no integer representation)")
        original_is_string = isinstance(value, bytes)
        located = False
        if original_is_string and level > 0:
            prefix = where_from_frames(vm._active_frames or (), level)
            if prefix:
                value = prefix + value
                located = True
        if value is None:
            value = b"<no error object>"
        raise LuaRaisedError(value, located=located)

    put("error", lua_error)

    def lua_assert(*args):
        value = args[0] if args else None
        if truthy(value):
            return MultiValue(tuple(args))
        message = args[1] if len(args) > 1 else b"assertion failed!"
        return lua_error(message, 1)

    put("assert", lua_assert)

    def pcall(fn, *args):
        try:
            results = vm.call_sync(fn, args)
            return MultiValue((True, *results))
        except LuaQuotaError:
            raise
        except LuaRuntimeError as error:
            return MultiValue((False, error_value(error)))

    put("pcall", pcall)

    def xpcall(fn, handler, *args):
        try:
            results = vm.call_sync(fn, args)
            return MultiValue((True, *results))
        except LuaQuotaError:
            raise
        except LuaRuntimeError as error:
            original = error_value(error)
            try:
                handled = vm.call_sync(handler, (original,))
                replacement = handled[0] if handled else None
            except LuaQuotaError:
                raise
            except LuaRuntimeError as handler_error:
                replacement = error_value(handler_error)
            return MultiValue((False, replacement))

    put("xpcall", xpcall)

    def load(chunk, chunkname=None, mode=b"bt", env=None):
        if mode is None:
            mode = b"bt"
        if not isinstance(mode, bytes):
            raise LuaRuntimeError("bad argument #3 to 'load' (string expected)")
        if any(char not in b"bt" for char in mode) or not mode:
            raise LuaRuntimeError("bad argument #3 to 'load' (invalid mode)")

        string_input = isinstance(chunk, bytes)
        if string_input:
            source = chunk
            source_name = source if chunkname is None else _as_bytes(chunkname)
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
            source_name = b"=(load)" if chunkname is None else _as_bytes(chunkname)

        if not isinstance(source_name, (bytes, str)):
            raise LuaRuntimeError("bad argument #2 to 'load' (string expected)")

        binary = source.startswith(b"\x1b")
        if binary and b"b" not in mode:
            return MultiValue((None, b"attempt to load a binary chunk (mode is 't')"))
        if not binary and b"t" not in mode:
            return MultiValue((None, b"attempt to load a text chunk (mode is 'b')"))

        environment = globals_table if env is None else env
        try:
            if source.startswith(DEBUG_NATIVE_MAGIC):
                proto = load_debug_chunk(source)
                if proto.source is None:
                    proto.source = source_name
            elif source.startswith(NATIVE_MAGIC):
                proto = load_native_chunk(source)
                if proto.source in (None, "=?"):
                    proto.source = source_name
            elif source.startswith(PUC_MAGIC) or binary:
                proto = load_puc55_chunk(source)
            else:
                text = source.decode("utf-8")
                proto = SourceCompiler(source_name).compile(Parser(text).parse())
        except UnicodeDecodeError as error:
            return MultiValue((None, _syntax_message(error, source_name)))
        except LuaPyreError as error:
            return MultiValue((None, _syntax_message(error, source_name)))
        return fresh_loaded_closure(proto, environment)

    put("load", load)

    stringlib = globals_table.rawget(b"string")
    if stringlib is not None:
        def string_dump(fn, strip=False):
            if not isinstance(fn, Closure):
                raise LuaRuntimeError("bad argument #1 to 'dump' (function expected)")
            return dump_debug_chunk(fn.proto, strip=truthy(strip))

        stringlib.rawset(b"dump", HostFunction(string_dump, "string.dump"))
