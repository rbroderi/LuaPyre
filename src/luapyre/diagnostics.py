from __future__ import annotations

from .errors import LuaRaisedError, LuaRuntimeError, LuaTraceFrame


LUA_IDSIZE = 60


def _bytes(value: str | bytes | None) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8", "replace")


def chunk_id(source: str | bytes | None) -> bytes:
    """Format a source name using Lua's luaO_chunkid rules."""
    raw = _bytes(source)
    if raw is None:
        return b"?"
    if raw.startswith(b"="):
        return raw[1:][: LUA_IDSIZE - 1]
    if raw.startswith(b"@"):
        name = raw[1:]
        if len(name) < LUA_IDSIZE:
            return name
        return b"..." + name[-(LUA_IDSIZE - 4):]

    prefix = b'[string "'
    suffix = b'"]'
    room = LUA_IDSIZE - len(prefix) - len(b"...") - len(suffix) - 1
    first_line = raw.split(b"\n", 1)[0]
    if len(raw) < room and b"\n" not in raw:
        body = raw
        ellipsis = b""
    else:
        body = first_line[:room]
        ellipsis = b"..."
    return prefix + body + ellipsis + suffix


def frame_line(frame, *, pc: int | None = None) -> int:
    if pc is None:
        pc = frame.pc - 1
    return frame.proto.line_for_pc(pc)


def trace_frame(frame, *, pc: int | None = None) -> LuaTraceFrame:
    return LuaTraceFrame(
        frame.proto.source,
        frame_line(frame, pc=pc),
        frame.proto.name,
    )


def where(frame, *, pc: int | None = None) -> bytes:
    """Equivalent of luaL_where: no prefix when current line is unavailable."""
    source = frame.proto.source
    line = frame_line(frame, pc=pc)
    if source is None or line <= 0:
        return b""
    return chunk_id(source) + b":" + str(line).encode("ascii") + b": "


def where_from_frames(frames, level: int) -> bytes:
    if level <= 0 or not frames or level > len(frames):
        return b""
    return where(frames[-level])


def error_value(error: LuaRuntimeError):
    if error.value is not None:
        return error.value
    return str(error).encode("utf-8", "replace")


def attach_runtime_context(error: LuaRuntimeError, frame, *, pc: int | None = None) -> LuaRuntimeError:
    """Attach the originating source location and one structured Lua frame.

    Explicit Lua ``error`` objects are preserved verbatim. VM/host runtime
    failures follow luaG_addinfo, which prints ``?:?:`` only when the source is
    absent and otherwise prints the source plus even an unavailable (-1) line.
    """
    error.add_trace_frame(trace_frame(frame, pc=pc))

    if isinstance(error, LuaRaisedError):
        return error
    if error.located:
        return error

    raw = error_value(error)
    line = frame_line(frame, pc=pc)
    if frame.proto.source is None:
        prefix = b"?:?: "
    else:
        prefix = chunk_id(frame.proto.source) + b":" + str(line).encode("ascii") + b": "
    value = prefix + raw
    error.value = value
    error.located = True
    error.args = (value.decode("utf-8", "replace"),)
    return error


def capture_error(error: LuaRuntimeError, frames) -> LuaRuntimeError:
    """Capture an origin plus its Lua callers before protected unwinding."""
    if not frames:
        return error
    attach_runtime_context(error, frames[-1])
    for frame in reversed(frames[:-1]):
        error.add_trace_frame(trace_frame(frame))
    return error


def add_unwind_frame(error: LuaRuntimeError, frame) -> None:
    error.add_trace_frame(trace_frame(frame))


def format_traceback(error: LuaRuntimeError, *, include_message: bool = True) -> bytes:
    """Render the captured Lua stack in the conventional Lua traceback shape."""
    parts: list[bytes] = []
    if include_message:
        value = error_value(error)
        if isinstance(value, bytes):
            parts.append(value)
        else:
            parts.append(str(value).encode("utf-8", "replace"))
    parts.append(b"stack traceback:")
    for frame in error.trace:
        loc = chunk_id(frame.source)
        if frame.line <= 0:
            prefix = b"\t" + loc + b": in "
        else:
            prefix = b"\t" + loc + b":" + str(frame.line).encode("ascii") + b": in "
        if frame.name == "<chunk>":
            desc = b"main chunk"
        elif frame.name == "<anonymous>":
            desc = b"function <?>"
        else:
            desc = b"function '" + frame.name.encode("utf-8", "replace") + b"'"
        parts.append(prefix + desc)
    return b"\n".join(parts)
