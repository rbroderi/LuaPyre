from __future__ import annotations

import base64
import binascii
import json
import struct

from .binary_chunks import (
    NATIVE_MAGIC,
    BinaryChunkError,
    dump_native_chunk,
    load_native_chunk,
)
from .bytecode import Proto


DEBUG_NATIVE_MAGIC = NATIVE_MAGIC + b"DBG2"
_MAX_TOTAL = 16 * 1024 * 1024
_MAX_DEBUG = 8 * 1024 * 1024


def _encode_source(source):
    if source is None:
        return None
    if isinstance(source, bytes):
        return {"t": "b", "v": base64.b64encode(source).decode("ascii")}
    return {"t": "s", "v": str(source)}


def _decode_source(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"t", "v"}:
        raise BinaryChunkError("invalid native debug source")
    if value["t"] == "s" and isinstance(value["v"], str):
        return value["v"]
    if value["t"] == "b" and isinstance(value["v"], str):
        try:
            return base64.b64decode(value["v"], validate=True)
        except (ValueError, binascii.Error) as error:
            raise BinaryChunkError("invalid native debug source") from error
    raise BinaryChunkError("invalid native debug source")


def _metadata(proto: Proto, strip: bool):
    return {
        "source": None if strip else _encode_source(proto.source),
        "linedefined": 0 if strip else proto.linedefined,
        "lastlinedefined": 0 if strip else proto.lastlinedefined,
        "lineinfo": [] if strip else list(proto.lineinfo),
        "children": [_metadata(child, strip) for child in proto.children],
    }


def dump_debug_chunk(proto: Proto, *, strip: bool = False) -> bytes:
    """Serialize VM data plus a separately validated source/line section."""
    core = dump_native_chunk(proto, strip=True)
    payload = core[len(NATIVE_MAGIC):]
    debug = json.dumps(
        _metadata(proto, strip), separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    if len(debug) > _MAX_DEBUG:
        raise BinaryChunkError("native debug metadata too large")
    result = DEBUG_NATIVE_MAGIC + struct.pack(">I", len(payload)) + payload + debug
    if len(result) > _MAX_TOTAL:
        raise BinaryChunkError("binary chunk too large")
    return result


def _apply(proto: Proto, metadata, *, depth=0):
    if depth > 200 or not isinstance(metadata, dict):
        raise BinaryChunkError("invalid native debug metadata")
    required = {"source", "linedefined", "lastlinedefined", "lineinfo", "children"}
    if set(metadata) != required:
        raise BinaryChunkError("invalid native debug metadata")
    children = metadata["children"]
    lineinfo = metadata["lineinfo"]
    if not isinstance(children, list) or len(children) != len(proto.children):
        raise BinaryChunkError("native debug prototype mismatch")
    if not isinstance(lineinfo, list) or len(lineinfo) not in (0, len(proto.code)):
        raise BinaryChunkError("native debug line table mismatch")
    if not all(type(line) is int and -1 <= line <= (1 << 31) - 1 for line in lineinfo):
        raise BinaryChunkError("invalid native debug line")
    linedefined = metadata["linedefined"]
    lastlinedefined = metadata["lastlinedefined"]
    if type(linedefined) is not int or type(lastlinedefined) is not int:
        raise BinaryChunkError("invalid native debug line range")

    proto.source = _decode_source(metadata["source"])
    proto.linedefined = linedefined
    proto.lastlinedefined = lastlinedefined
    proto.lineinfo = list(lineinfo)
    for child, child_metadata in zip(proto.children, children):
        _apply(child, child_metadata, depth=depth + 1)


def load_debug_chunk(data: bytes) -> Proto:
    if len(data) > _MAX_TOTAL:
        raise BinaryChunkError("binary chunk too large")
    if not data.startswith(DEBUG_NATIVE_MAGIC):
        raise BinaryChunkError("not a LuaPyre debug binary chunk")
    pos = len(DEBUG_NATIVE_MAGIC)
    if len(data) < pos + 4:
        raise BinaryChunkError("truncated native debug chunk")
    core_len = struct.unpack(">I", data[pos:pos + 4])[0]
    pos += 4
    if core_len > _MAX_TOTAL or pos + core_len > len(data):
        raise BinaryChunkError("invalid native debug core length")
    core = NATIVE_MAGIC + data[pos:pos + core_len]
    debug = data[pos + core_len:]
    if not debug or len(debug) > _MAX_DEBUG:
        raise BinaryChunkError("invalid native debug metadata size")
    try:
        metadata = json.loads(debug.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BinaryChunkError("invalid native debug metadata") from error
    proto = load_native_chunk(core)
    _apply(proto, metadata)
    return proto
