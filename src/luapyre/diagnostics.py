from __future__ import annotations

from .bytecode import Op
from .errors import LuaRaisedError, LuaRuntimeError, LuaTraceFrame


LUA_IDSIZE = 60

_ARITH_OPS = frozenset({
    Op.ADD, Op.ADD_I, Op.ADD_F,
    Op.SUB, Op.SUB_I, Op.SUB_F,
    Op.MUL, Op.MUL_I, Op.MUL_F,
    Op.DIV, Op.IDIV, Op.MOD, Op.POW,
    Op.NEG,
})
_CALL_OPS = frozenset({Op.CALL, Op.CALLV, Op.TAILCALL, Op.TAILCALLV})


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
        frame.trace_name or frame.proto.name,
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


def _number(value) -> bool:
    return type(value) in (int, float)


def _enrich_runtime_error(error: LuaRuntimeError, frame, pc: int) -> None:
    """Add Lua's debug-variable suffix without touching the normal VM path.

    PUC's symbolic debug information is projected onto translated VM registers
    at compile/load time. We consult it only after an exception has already
    occurred, keeping diagnostic fidelity out of the hot opcode loop.
    """
    if isinstance(error, LuaRaisedError) or error.value is not None:
        return
    if pc < 0 or pc >= len(frame.proto.code):
        return

    message = str(error)
    ins = frame.proto.code[pc]
    if ins.op in _CALL_OPS and message.startswith("attempt to call a "):
        bad_reg = ins.b
        if pc < len(frame.proto.value_origins):
            origin = frame.proto.value_origins[pc].get(bad_reg)
            if origin is not None:
                kind, name = origin
                error.args = (message + f" ({kind} '{name}')",)
        return
    if ins.op in (Op.GETTABLE, Op.SETTABLE) and message.startswith("attempt to index a "):
        bad_reg = ins.b if ins.op is Op.GETTABLE else ins.a
        if pc < len(frame.proto.value_origins):
            origin = frame.proto.value_origins[pc].get(bad_reg)
            if origin is not None:
                kind, name = origin
                error.args = (message + f" ({kind} '{name}')",)
        return
    if ins.op not in _ARITH_OPS or not message.startswith("attempt to perform arithmetic on a "):
        return

    left = frame.regs[ins.b]
    right = frame.regs[ins.c]
    if not _number(left):
        bad_reg, bad_value = ins.b, left
    elif not _number(right):
        bad_reg, bad_value = ins.c, right
    else:
        return

    if pc < len(frame.proto.value_origins):
        origin = frame.proto.value_origins[pc].get(bad_reg)
        if origin is not None:
            kind, name = origin
            message += f" ({kind} '{name}')"
    error.args = (message,)


def attach_runtime_context(error: LuaRuntimeError, frame, *, pc: int | None = None) -> LuaRuntimeError:
    """Attach the originating source location and one structured Lua frame.

    Explicit Lua ``error`` objects are preserved verbatim. VM/host runtime
    failures follow luaG_addinfo, which prints ``?:?:`` only when the source is
    absent and otherwise prints the source plus even an unavailable (-1) line.
    """
    if pc is None:
        pc = frame.pc - 1
    error.add_trace_frame(trace_frame(frame, pc=pc))

    if isinstance(error, LuaRaisedError):
        return error
    if error.located:
        return error

    _enrich_runtime_error(error, frame, pc)
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
        elif frame.name == "__close":
            desc = b"metamethod 'close'"
        else:
            desc = b"function '" + frame.name.encode("utf-8", "replace") + b"'"
        parts.append(prefix + desc)
    return b"\n".join(parts)
