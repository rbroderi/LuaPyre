from __future__ import annotations

from dataclasses import dataclass, field
import struct

from .binary_chunks import (
    BinaryChunkError,
    _MAX_CODE,
    _MAX_CONSTANTS,
    _MAX_DEPTH,
    _MAX_PROTOS,
    _PucProto,
    _PucReader,
    _PucUpvalue,
    _Translator,
)
from .bytecode import Proto, UpvalueDesc
from .typesys import ANY


_ABSLINEINFO = -0x80


@dataclass(frozen=True, slots=True)
class PucLocal:
    name: bytes | None
    startpc: int
    endpc: int


@dataclass(slots=True)
class DebugPucProto(_PucProto):
    raw_lineinfo: list[int] = field(default_factory=list)
    abslineinfo: list[tuple[int, int]] = field(default_factory=list)
    locvars: list[PucLocal] = field(default_factory=list)


class DebugPucReader(_PucReader):
    """PUC-Lua 5.5 reader that keeps validated debug metadata.

    The 0.8 reader already consumed these arrays so malformed chunks could not
    hide data after the executable payload. 0.10 retains the same bounded
    parsing while preserving the information needed for source lines and local
    names.
    """

    def proto(self, depth: int = 0) -> DebugPucProto:
        if depth > _MAX_DEPTH:
            raise BinaryChunkError("PUC-Lua prototype nesting too deep")
        self.protos += 1
        if self.protos > _MAX_PROTOS:
            raise BinaryChunkError("too many PUC-Lua prototypes")

        p = DebugPucProto()
        p.linedefined = self.count((1 << 31) - 1)
        p.lastlinedefined = self.count((1 << 31) - 1)
        p.numparams = self.byte()
        p.flags = self.byte() & ~4
        p.maxstacksize = self.byte()
        if p.maxstacksize == 0:
            raise BinaryChunkError("invalid PUC-Lua stack size")

        ncode = self.count(_MAX_CODE)
        self.instructions += ncode
        if self.instructions > _MAX_CODE:
            raise BinaryChunkError("too many PUC-Lua instructions")
        self.align(4)
        p.code = [
            int.from_bytes(self.read(4), self.endian, signed=False)
            for _ in range(ncode)
        ]

        nconstants = self.count(_MAX_CONSTANTS)
        self.constants += nconstants
        if self.constants > _MAX_CONSTANTS:
            raise BinaryChunkError("too many PUC-Lua constants")
        constants = []
        prefix = "<" if self.endian == "little" else ">"
        for _ in range(nconstants):
            tag = self.byte()
            if tag == 0:
                constants.append(None)
            elif tag == 1:
                constants.append(False)
            elif tag == 17:
                constants.append(True)
            elif tag == 3:
                constants.append(self.integer())
            elif tag == 19:
                constants.append(struct.unpack(prefix + "d", self.read(8))[0])
            elif tag in (4, 20):
                value = self.string()
                if value is None:
                    raise BinaryChunkError("nil string constant")
                constants.append(value)
            else:
                raise BinaryChunkError(f"unsupported PUC-Lua constant tag {tag}")
        p.constants = constants

        nup = self.count(1_000_000)
        p.upvalues = [
            _PucUpvalue(self.byte(), self.byte(), self.byte())
            for _ in range(nup)
        ]

        nchildren = self.count(_MAX_PROTOS)
        p.children = [self.proto(depth + 1) for _ in range(nchildren)]
        p.source = self.string()

        nline = self.count(_MAX_CODE)
        if nline not in (0, len(p.code)):
            raise BinaryChunkError("PUC-Lua line table size mismatch")
        p.raw_lineinfo = [
            int.from_bytes(self.read(1), "little", signed=True)
            for _ in range(nline)
        ]

        nabs = self.count(_MAX_CODE)
        abslineinfo: list[tuple[int, int]] = []
        if nabs:
            self.align(4)
            for _ in range(nabs):
                pc = int.from_bytes(self.read(4), self.endian, signed=True)
                line = int.from_bytes(self.read(4), self.endian, signed=True)
                abslineinfo.append((pc, line))
        p.abslineinfo = abslineinfo
        self._validate_lines(p)

        nlocals = self.count(1_000_000)
        locals_: list[PucLocal] = []
        for _ in range(nlocals):
            name = self.string()
            startpc = self.count(_MAX_CODE)
            endpc = self.count(_MAX_CODE)
            if startpc > endpc or endpc > len(p.code):
                raise BinaryChunkError("invalid PUC-Lua local variable range")
            locals_.append(PucLocal(name, startpc, endpc))
        p.locvars = locals_

        nupnames = self.count(1_000_000)
        if nupnames:
            # PUC treats this count as a boolean debug-info flag and then reads
            # exactly sizeupvalues names (lundump.c:loadDebug).
            for upvalue in p.upvalues:
                upvalue.name = self.string()
        return p

    @staticmethod
    def _validate_lines(proto: DebugPucProto) -> None:
        if not proto.raw_lineinfo:
            if proto.abslineinfo:
                raise BinaryChunkError("absolute lines without PUC-Lua line table")
            return

        previous = -1
        absolute_pcs: list[int] = []
        for pc, line in proto.abslineinfo:
            if pc <= previous or pc < 0 or pc >= len(proto.code) or line < 0:
                raise BinaryChunkError("invalid PUC-Lua absolute line table")
            previous = pc
            absolute_pcs.append(pc)

        markers = [
            pc for pc, delta in enumerate(proto.raw_lineinfo)
            if delta == _ABSLINEINFO
        ]
        if markers != absolute_pcs:
            raise BinaryChunkError("PUC-Lua absolute line markers do not match")


def decode_puc_chunk_with_debug(data: bytes) -> DebugPucProto:
    reader = DebugPucReader(data)
    nupvalues = reader.header()
    proto = reader.proto()
    if nupvalues != len(proto.upvalues):
        raise BinaryChunkError("PUC-Lua root upvalue count mismatch")
    if reader.pos != len(data):
        raise BinaryChunkError("trailing data in PUC-Lua chunk")
    return proto


def reconstruct_puc_lines(proto: DebugPucProto) -> list[int]:
    """Expand Lua 5.5's delta/absolute line encoding to one line per PUC op."""
    if not proto.raw_lineinfo:
        return []

    absolutes = dict(proto.abslineinfo)
    current = proto.linedefined
    lines: list[int] = []
    for pc, delta in enumerate(proto.raw_lineinfo):
        if delta == _ABSLINEINFO:
            try:
                current = absolutes[pc]
            except KeyError as error:  # validation should make this unreachable
                raise BinaryChunkError("missing PUC-Lua absolute line") from error
        else:
            current += delta
        lines.append(current)
    return lines


class _TrackingPCMap(dict[int, int]):
    def __init__(self, owner: "DebugTranslator"):
        super().__init__()
        self.owner = owner

    def __setitem__(self, key: int, value: int) -> None:
        super().__setitem__(key, value)
        self.owner.current_puc_pc = key if 0 <= key < len(self.owner._puc_lines) else None


class DebugTranslator(_Translator):
    """Reuse the 0.8 PUC lowering while projecting source lines onto VM ops."""

    def __init__(self, source: DebugPucProto):
        self.source = source
        name = source.source.decode("utf-8", "replace") if source.source else "<binary>"
        self.proto = Proto(name)
        self.proto.source = source.source
        self.proto.linedefined = source.linedefined
        self.proto.lastlinedefined = source.lastlinedefined
        self.proto.constants = list(source.constants)
        self.proto.param_count = source.numparams
        self.proto.param_types = [ANY] * source.numparams
        self.proto.return_types = [ANY]
        self.proto.is_vararg = bool(source.flags & 3)
        self.proto.vararg_name_reg = source.numparams if source.flags & 2 else -1
        self.proto.vararg_type = ANY
        self.proto.env_reg = -1
        self.proto.upvalues = [
            UpvalueDesc(
                "local" if uv.instack else "upvalue",
                uv.index,
                uv.name.decode("utf-8", "replace") if uv.name else "?",
            )
            for uv in source.upvalues
        ]
        self.proto.children = [DebugTranslator(child).translate() for child in source.children]
        self.captured = {
            uv.index
            for child in self.proto.children
            for uv in child.upvalues
            if uv.kind == "local"
        }
        self.next_reg = source.maxstacksize
        self.max_reg = source.maxstacksize
        self._puc_lines = reconstruct_puc_lines(source)
        self.current_puc_pc: int | None = None
        self.pcmap = _TrackingPCMap(self)
        self.patches: list[tuple[int, str, int]] = []
        self.open_result: tuple[int, int] | None = None

    def emit(self, op, a=0, b=0, c=0, d=0, e=0):
        index = super().emit(op, a, b, c, d, e)
        if self.current_puc_pc is None or not self._puc_lines:
            self.proto.lineinfo.append(-1)
        else:
            self.proto.lineinfo.append(self._puc_lines[self.current_puc_pc])
        return index
