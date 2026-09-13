from __future__ import annotations

import json
import struct
import sys
import zlib

from .binary_chunks import (
    NATIVE_MAGIC,
    PUC_MAGIC,
    BinaryChunkError,
    dump_native_chunk,
    load_native_chunk,
)
from .bytecode import Proto


# Lua 5.5's portable binary signature and platform checks.  LuaPyre keeps its
# own validated instruction payload (the VM instruction sets are deliberately
# different), but emits the public 5.5 header so dump consumers can identify
# the version, format, endian, and numeric representation in the usual way.
_BYTEORDER = sys.byteorder
_FLOAT_PREFIX = "<" if _BYTEORDER == "little" else ">"
PUC55_HEADER = (
    PUC_MAGIC
    + bytes((0x55, 0))
    + b"\x19\x93\r\n\x1a\n"
    + bytes((4,))
    + (-0x5678).to_bytes(4, _BYTEORDER, signed=True)
    + bytes((4,))
    + (0x12345678).to_bytes(4, _BYTEORDER, signed=False)
    + bytes((8,))
    + (-0x5678).to_bytes(8, _BYTEORDER, signed=True)
    + bytes((8,))
    + struct.pack(_FLOAT_PREFIX + "d", -370.5)
)
DEBUG_NATIVE_MAGIC = PUC55_HEADER
_DEBUG_TAG = b"LuaPyreDBG3\0"
_MAX_TOTAL = 16 * 1024 * 1024
_MAX_DEBUG = 8 * 1024 * 1024


def _encode_source(source):
    if source is None:
        return None
    if isinstance(source, bytes):
        # Latin-1 is a reversible byte mapping and, unlike base64, preserves
        # ordinary source spelling in a dump.  This mirrors Lua's shared dump
        # string table observably: source text and a matching constant each
        # occur once instead of encoding a second opaque copy.
        return {"t": "l", "v": source.decode("latin-1")}
    return {"t": "s", "v": str(source)}


def _decode_source(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"t", "v"}:
        raise BinaryChunkError("invalid native debug source")
    if value["t"] == "s" and isinstance(value["v"], str):
        return value["v"]
    if value["t"] == "l" and isinstance(value["v"], str):
        try:
            return value["v"].encode("latin-1")
        except UnicodeEncodeError as error:
            raise BinaryChunkError("invalid native debug source") from error
    raise BinaryChunkError("invalid native debug source")


def _metadata(proto: Proto, strip: bool, parent_source=object()):
    source = None if strip or proto.source == parent_source else _encode_source(proto.source)
    return {
        "source": source,
        # PUC preserves the function's definition range in stripped chunks;
        # only per-instruction lines, source names, and symbolic names go.
        "linedefined": proto.linedefined,
        "lastlinedefined": proto.lastlinedefined,
        "lineinfo": [] if strip else list(proto.lineinfo),
        # Our VM does not reuse registers as aggressively as PUC. Keep only
        # anonymous scope/range information so debug.getlocal can expose the
        # same live stack slots without leaking stripped local names.
        "l": [
            ["(temporary)" if strip else name, register, start, end]
            for name, register, start, end in proto.debug_locals
        ],
        "s": strip,
        "children": [_metadata(child, strip, proto.source) for child in proto.children],
    }


def _string_catalog(proto: Proto) -> bytes:
    """Emit each string constant once, mirroring PUC dump string reuse."""
    values: list[bytes] = []
    seen: set[bytes] = set()

    def visit(current: Proto) -> None:
        for constant in current.constants:
            if isinstance(constant, bytes):
                value = constant
            elif isinstance(constant, str):
                value = constant.encode("utf-8")
            else:
                continue
            if value not in seen:
                seen.add(value)
                values.append(value)
        for child in current.children:
            visit(child)

    visit(proto)
    out = bytearray(struct.pack(">I", len(values)))
    for value in values:
        out.extend(struct.pack(">I", len(value)))
        out.extend(value)
    return bytes(out)


def dump_debug_chunk(proto: Proto, *, strip: bool = False) -> bytes:
    """Serialize VM data plus a separately validated source/line section."""
    core = dump_native_chunk(proto, strip=True)
    payload = zlib.compress(core[len(NATIVE_MAGIC):], level=9)
    debug = json.dumps(
        _metadata(proto, strip), separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    if len(debug) > _MAX_DEBUG:
        raise BinaryChunkError("native debug metadata too large")
    catalog = _string_catalog(proto)
    root_upvalues = len(proto.upvalues) + int(proto.env_reg >= 0)
    if root_upvalues > 255:
        raise BinaryChunkError("too many root upvalues")
    result = (
        DEBUG_NATIVE_MAGIC
        + bytes((root_upvalues,))
        + _DEBUG_TAG
        + struct.pack(">I", len(payload))
        + payload
        + struct.pack(">I", len(debug))
        + debug
        + catalog
    )
    if len(result) > _MAX_TOTAL:
        raise BinaryChunkError("binary chunk too large")
    return result


def _apply(proto: Proto, metadata, *, depth=0, parent_source=None):
    if depth > 200 or not isinstance(metadata, dict):
        raise BinaryChunkError("invalid native debug metadata")
    required = {
        "source", "linedefined", "lastlinedefined", "lineinfo", "l", "s",
        "children",
    }
    if set(metadata) != required:
        raise BinaryChunkError("invalid native debug metadata")
    children = metadata["children"]
    lineinfo = metadata["lineinfo"]
    locals_metadata = metadata["l"]
    stripped = metadata["s"]
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
    if type(stripped) is not bool or not isinstance(locals_metadata, list):
        raise BinaryChunkError("invalid native local metadata")
    decoded_locals = []
    for item in locals_metadata:
        if (
            not isinstance(item, list)
            or len(item) != 4
            or not isinstance(item[0], str)
            or any(type(value) is not int for value in item[1:])
        ):
            raise BinaryChunkError("invalid native local metadata")
        name, register, start, end = item
        if register < -1 or register >= proto.register_count or start < 0 or end < start:
            raise BinaryChunkError("invalid native local metadata")
        decoded_locals.append((name, register, start, end))

    decoded_source = _decode_source(metadata["source"])
    proto.source = parent_source if depth and decoded_source is None else decoded_source
    proto.linedefined = linedefined
    proto.lastlinedefined = lastlinedefined
    proto.lineinfo = list(lineinfo)
    proto.debug_locals = decoded_locals
    proto.debug_stripped = stripped
    if stripped:
        proto.upvalues = [
            type(upvalue)(upvalue.kind, upvalue.index, "(no name)")
            for upvalue in proto.upvalues
        ]
    for child, child_metadata in zip(proto.children, children):
        _apply(
            child,
            child_metadata,
            depth=depth + 1,
            parent_source=proto.source,
        )


def load_debug_chunk(data: bytes) -> Proto:
    if len(data) > _MAX_TOTAL:
        raise BinaryChunkError("binary chunk too large")
    if not is_debug_chunk(data):
        raise BinaryChunkError("not a LuaPyre debug binary chunk")
    pos = len(DEBUG_NATIVE_MAGIC) + 1 + len(_DEBUG_TAG)
    if len(data) < pos + 4:
        raise BinaryChunkError("truncated native debug chunk")
    core_len = struct.unpack(">I", data[pos:pos + 4])[0]
    pos += 4
    if core_len > _MAX_TOTAL:
        raise BinaryChunkError("invalid native debug core length")
    if pos + core_len > len(data):
        raise BinaryChunkError("truncated native debug chunk")
    compressed = data[pos:pos + core_len]
    pos += core_len
    try:
        inflater = zlib.decompressobj()
        payload = inflater.decompress(compressed, _MAX_TOTAL + 1)
        if len(payload) > _MAX_TOTAL or inflater.unconsumed_tail:
            raise BinaryChunkError("native debug core is too large")
        payload += inflater.flush(_MAX_TOTAL + 1 - len(payload))
    except zlib.error as error:
        raise BinaryChunkError("truncated or invalid native payload") from error
    if not inflater.eof or inflater.unused_data or len(payload) > _MAX_TOTAL:
        raise BinaryChunkError("truncated or invalid native payload")
    core = NATIVE_MAGIC + payload
    if pos + 4 > len(data):
        raise BinaryChunkError("truncated native debug metadata")
    debug_len = struct.unpack(">I", data[pos:pos + 4])[0]
    pos += 4
    if debug_len > _MAX_DEBUG or pos + debug_len > len(data):
        raise BinaryChunkError("truncated or invalid native debug metadata")
    debug = data[pos:pos + debug_len]
    pos += debug_len
    if not debug:
        raise BinaryChunkError("truncated native debug metadata")
    if len(debug) > _MAX_DEBUG:
        raise BinaryChunkError("invalid native debug metadata size")
    try:
        metadata = json.loads(debug.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BinaryChunkError("truncated or invalid native debug metadata") from error
    proto = load_native_chunk(core)
    _apply(proto, metadata)
    if pos + 4 > len(data):
        raise BinaryChunkError("truncated native string catalog")
    count = struct.unpack(">I", data[pos:pos + 4])[0]
    pos += 4
    if count > 1_000_000:
        raise BinaryChunkError("invalid native string catalog")
    for _ in range(count):
        if pos + 4 > len(data):
            raise BinaryChunkError("truncated native string catalog")
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        pos += 4
        if size > _MAX_TOTAL or pos + size > len(data):
            raise BinaryChunkError("truncated native string catalog")
        pos += size
    if pos != len(data):
        raise BinaryChunkError("trailing data in native debug chunk")
    return proto


def is_debug_chunk(data: bytes) -> bool:
    """Return whether a complete LuaPyre tag follows the Lua 5.5 header."""
    pos = len(DEBUG_NATIVE_MAGIC)
    return (
        len(data) >= pos + 1 + len(_DEBUG_TAG)
        and data.startswith(DEBUG_NATIVE_MAGIC)
        and data[pos + 1:pos + 1 + len(_DEBUG_TAG)] == _DEBUG_TAG
    )
