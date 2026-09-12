from __future__ import annotations

from dataclasses import dataclass, field
import math
import struct

from .bytecode import Cell, Closure, Ins, Op, Proto, UpvalueDesc
from .errors import LuaRuntimeError
from .typesys import ANY, parse_simple_type


NATIVE_MAGIC = b"\x1bLuaPyre\x55\x01\r\n\x1a\n"
PUC_MAGIC = b"\x1bLua"
_MAX_CHUNK = 16 * 1024 * 1024
_MAX_STRING = 16 * 1024 * 1024
_MAX_PROTOS = 10_000
_MAX_CODE = 1_000_000
_MAX_CONSTANTS = 1_000_000
_MAX_DEPTH = 200


class BinaryChunkError(LuaRuntimeError):
    pass


# ---------------------------------------------------------------------------
# LuaPyre native chunks
# ---------------------------------------------------------------------------


def _put_varuint(out: bytearray, value: int) -> None:
    if value < 0:
        raise ValueError("negative varuint")
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return


def _put_sint(out: bytearray, value: int) -> None:
    encoded = value * 2 if value >= 0 else (-value * 2 - 1)
    _put_varuint(out, encoded)


def _put_bytes(out: bytearray, value: bytes) -> None:
    _put_varuint(out, len(value))
    out.extend(value)


def _put_text(out: bytearray, value: str) -> None:
    _put_bytes(out, value.encode("utf-8"))


def _dump_constant(out: bytearray, value) -> None:
    if value is None:
        out.append(0)
    elif value is False:
        out.append(1)
    elif value is True:
        out.append(2)
    elif type(value) is int:
        out.append(3)
        _put_sint(out, value)
    elif type(value) is float:
        out.append(4)
        out.extend(struct.pack(">d", value))
    elif isinstance(value, bytes):
        out.append(5)
        _put_bytes(out, value)
    elif isinstance(value, str):
        out.append(6)
        _put_text(out, value)
    else:
        raise BinaryChunkError(f"cannot dump constant of type {type(value).__name__}")


def _dump_native_proto(out: bytearray, proto: Proto, depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise BinaryChunkError("prototype nesting too deep")
    _put_text(out, proto.name)
    _put_varuint(out, proto.register_count)
    _put_varuint(out, proto.param_count)
    out.append(1 if proto.is_vararg else 0)
    _put_sint(out, proto.vararg_name_reg)
    _put_sint(out, proto.env_reg)

    _put_varuint(out, len(proto.param_types))
    for typ in proto.param_types:
        _put_text(out, typ.name)
    _put_varuint(out, len(proto.return_types))
    for typ in proto.return_types:
        _put_text(out, typ.name)
    _put_text(out, proto.vararg_type.name)

    _put_varuint(out, len(proto.constants))
    for constant in proto.constants:
        _dump_constant(out, constant)

    _put_varuint(out, len(proto.upvalues))
    for upvalue in proto.upvalues:
        if upvalue.kind not in ("local", "upvalue"):
            raise BinaryChunkError(f"invalid upvalue kind {upvalue.kind!r}")
        out.append(0 if upvalue.kind == "local" else 1)
        _put_varuint(out, upvalue.index)
        _put_text(out, upvalue.name)

    _put_varuint(out, len(proto.children))
    for child in proto.children:
        _dump_native_proto(out, child, depth + 1)

    _put_varuint(out, len(proto.code))
    for ins in proto.code:
        _put_varuint(out, int(ins.op))
        for value in (ins.a, ins.b, ins.c, ins.d, ins.e):
            _put_sint(out, value)


def dump_native_chunk(proto: Proto, *, strip: bool = False) -> bytes:
    """Serialize a LuaPyre prototype without Python pickle or host objects."""
    out = bytearray(NATIVE_MAGIC)
    _dump_native_proto(out, proto)
    if len(out) > _MAX_CHUNK:
        raise BinaryChunkError("binary chunk too large")
    return bytes(out)


class _NativeReader:
    def __init__(self, data: bytes):
        if len(data) > _MAX_CHUNK:
            raise BinaryChunkError("binary chunk too large")
        self.data = data
        self.pos = 0
        self.protos = 0
        self.instructions = 0
        self.constants = 0

    def read(self, size: int) -> bytes:
        if size < 0 or self.pos + size > len(self.data):
            raise BinaryChunkError("truncated binary chunk")
        value = self.data[self.pos:self.pos + size]
        self.pos += size
        return value

    def byte(self) -> int:
        return self.read(1)[0]

    def varuint(self, *, maximum=(1 << 63) - 1) -> int:
        value = 0
        shift = 0
        while True:
            byte = self.byte()
            value |= (byte & 0x7F) << shift
            if value > maximum:
                raise BinaryChunkError("integer overflow in binary chunk")
            if not byte & 0x80:
                return value
            shift += 7
            if shift > 70:
                raise BinaryChunkError("integer overflow in binary chunk")

    def sint(self) -> int:
        value = self.varuint(maximum=(1 << 64) - 1)
        return -(value // 2) - 1 if value & 1 else value // 2

    def bytes(self) -> bytes:
        size = self.varuint(maximum=_MAX_STRING)
        return self.read(size)

    def text(self) -> str:
        try:
            return self.bytes().decode("utf-8")
        except UnicodeDecodeError as error:
            raise BinaryChunkError("invalid UTF-8 metadata in binary chunk") from error

    def constant(self):
        tag = self.byte()
        if tag == 0:
            return None
        if tag == 1:
            return False
        if tag == 2:
            return True
        if tag == 3:
            value = self.sint()
            if value < -(1 << 63) or value > (1 << 63) - 1:
                raise BinaryChunkError("integer constant out of range")
            return value
        if tag == 4:
            return struct.unpack(">d", self.read(8))[0]
        if tag == 5:
            return self.bytes()
        if tag == 6:
            return self.text()
        raise BinaryChunkError("invalid native constant tag")

    def proto(self, depth: int = 0) -> Proto:
        if depth > _MAX_DEPTH:
            raise BinaryChunkError("prototype nesting too deep")
        self.protos += 1
        if self.protos > _MAX_PROTOS:
            raise BinaryChunkError("too many prototypes")

        proto = Proto(self.text())
        proto.register_count = self.varuint(maximum=1_000_000)
        proto.param_count = self.varuint(maximum=proto.register_count)
        is_vararg = self.byte()
        if is_vararg not in (0, 1):
            raise BinaryChunkError("invalid vararg flag")
        proto.is_vararg = bool(is_vararg)
        proto.vararg_name_reg = self.sint()
        proto.env_reg = self.sint()

        nparams = self.varuint(maximum=1_000_000)
        proto.param_types = [parse_simple_type(self.text()) for _ in range(nparams)]
        nreturns = self.varuint(maximum=1_000_000)
        proto.return_types = [parse_simple_type(self.text()) for _ in range(nreturns)]
        proto.vararg_type = parse_simple_type(self.text())
        if len(proto.param_types) != proto.param_count:
            raise BinaryChunkError("parameter metadata mismatch")

        count = self.varuint(maximum=_MAX_CONSTANTS)
        self.constants += count
        if self.constants > _MAX_CONSTANTS:
            raise BinaryChunkError("too many constants")
        proto.constants = [self.constant() for _ in range(count)]

        count = self.varuint(maximum=1_000_000)
        upvalues = []
        for _ in range(count):
            tag = self.byte()
            if tag not in (0, 1):
                raise BinaryChunkError("invalid upvalue descriptor")
            index = self.varuint(maximum=1_000_000)
            upvalues.append(UpvalueDesc("local" if tag == 0 else "upvalue", index, self.text()))
        proto.upvalues = upvalues

        count = self.varuint(maximum=_MAX_PROTOS)
        proto.children = [self.proto(depth + 1) for _ in range(count)]

        count = self.varuint(maximum=_MAX_CODE)
        self.instructions += count
        if self.instructions > _MAX_CODE:
            raise BinaryChunkError("too many instructions")
        code = []
        for _ in range(count):
            rawop = self.varuint(maximum=10_000)
            try:
                op = Op(rawop)
            except ValueError as error:
                raise BinaryChunkError("unknown native opcode") from error
            values = [self.sint() for _ in range(5)]
            code.append(Ins(op, *values))
        proto.code = code
        return proto


def load_native_chunk(data: bytes) -> Proto:
    if not data.startswith(NATIVE_MAGIC):
        raise BinaryChunkError("not a LuaPyre binary chunk")
    reader = _NativeReader(data)
    reader.read(len(NATIVE_MAGIC))
    proto = reader.proto()
    if reader.pos != len(data):
        raise BinaryChunkError("trailing data in binary chunk")
    return proto


# ---------------------------------------------------------------------------
# Official PUC-Lua 5.5.1 chunks
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PucUpvalue:
    instack: int
    index: int
    kind: int
    name: bytes | None = None


@dataclass(slots=True)
class _PucProto:
    linedefined: int = 0
    lastlinedefined: int = 0
    numparams: int = 0
    flags: int = 0
    maxstacksize: int = 0
    code: list[int] = field(default_factory=list)
    constants: list[object] = field(default_factory=list)
    upvalues: list[_PucUpvalue] = field(default_factory=list)
    children: list["_PucProto"] = field(default_factory=list)
    source: bytes | None = None


class _PucReader:
    DATA = b"\x19\x93\r\n\x1a\n"

    def __init__(self, data: bytes):
        if len(data) > _MAX_CHUNK:
            raise BinaryChunkError("binary chunk too large")
        self.data = data
        self.pos = 0
        self.endian = "little"
        self.saved_strings: list[bytes] = []
        self.protos = 0
        self.instructions = 0
        self.constants = 0

    def read(self, size: int) -> bytes:
        if size < 0 or self.pos + size > len(self.data):
            raise BinaryChunkError("truncated PUC-Lua chunk")
        value = self.data[self.pos:self.pos + size]
        self.pos += size
        return value

    def byte(self) -> int:
        return self.read(1)[0]

    def align(self, alignment: int) -> None:
        padding = (-self.pos) % alignment
        if padding:
            self.read(padding)

    def varint(self, maximum=(1 << 64) - 1) -> int:
        value = 0
        limit = maximum >> 7
        while True:
            byte = self.byte()
            if value > limit:
                raise BinaryChunkError("integer overflow in PUC-Lua chunk")
            value = (value << 7) | (byte & 0x7F)
            if not byte & 0x80:
                return value

    def count(self, maximum: int) -> int:
        return self.varint(maximum)

    def integer(self) -> int:
        encoded = self.varint((1 << 64) - 1)
        value = ~(encoded >> 1) if encoded & 1 else encoded >> 1
        if value < -(1 << 63) or value > (1 << 63) - 1:
            raise BinaryChunkError("Lua integer out of range")
        return value

    def string(self) -> bytes | None:
        size = self.varint(_MAX_STRING + 1)
        if size == 0:
            index = self.varint(len(self.saved_strings))
            if index == 0:
                return None
            if index > len(self.saved_strings):
                raise BinaryChunkError("invalid PUC-Lua string index")
            return self.saved_strings[index - 1]
        if size < 1 or size - 1 > _MAX_STRING:
            raise BinaryChunkError("PUC-Lua string too large")
        raw = self.read(size)
        if not raw or raw[-1] != 0:
            raise BinaryChunkError("invalid PUC-Lua string terminator")
        value = raw[:-1]
        self.saved_strings.append(value)
        return value

    def header(self) -> int:
        if self.read(4) != PUC_MAGIC:
            raise BinaryChunkError("not a PUC-Lua binary chunk")
        if self.byte() != 0x55:
            raise BinaryChunkError("PUC-Lua version mismatch")
        if self.byte() != 0:
            raise BinaryChunkError("PUC-Lua format mismatch")
        if self.read(len(self.DATA)) != self.DATA:
            raise BinaryChunkError("corrupted PUC-Lua chunk")

        int_size = self.byte()
        if int_size != 4:
            raise BinaryChunkError("unsupported PUC-Lua int size")
        marker = self.read(int_size)
        little = int.from_bytes(marker, "little", signed=True)
        big = int.from_bytes(marker, "big", signed=True)
        if little == -0x5678:
            self.endian = "little"
        elif big == -0x5678:
            self.endian = "big"
        else:
            raise BinaryChunkError("PUC-Lua int format mismatch")

        instruction_size = self.byte()
        if instruction_size != 4:
            raise BinaryChunkError("unsupported PUC-Lua instruction size")
        if int.from_bytes(self.read(4), self.endian, signed=False) != 0x12345678:
            raise BinaryChunkError("PUC-Lua instruction format mismatch")

        integer_size = self.byte()
        if integer_size != 8:
            raise BinaryChunkError("unsupported PUC-Lua integer size")
        if int.from_bytes(self.read(8), self.endian, signed=True) != -0x5678:
            raise BinaryChunkError("PUC-Lua integer format mismatch")

        number_size = self.byte()
        if number_size != 8:
            raise BinaryChunkError("unsupported PUC-Lua number size")
        prefix = "<" if self.endian == "little" else ">"
        if struct.unpack(prefix + "d", self.read(8))[0] != -370.5:
            raise BinaryChunkError("PUC-Lua number format mismatch")
        return self.byte()

    def proto(self, depth: int = 0) -> _PucProto:
        if depth > _MAX_DEPTH:
            raise BinaryChunkError("PUC-Lua prototype nesting too deep")
        self.protos += 1
        if self.protos > _MAX_PROTOS:
            raise BinaryChunkError("too many PUC-Lua prototypes")

        p = _PucProto()
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
        p.code = [int.from_bytes(self.read(4), self.endian, signed=False) for _ in range(ncode)]

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
        p.upvalues = [_PucUpvalue(self.byte(), self.byte(), self.byte()) for _ in range(nup)]

        nchildren = self.count(_MAX_PROTOS)
        p.children = [self.proto(depth + 1) for _ in range(nchildren)]
        p.source = self.string()

        # Debug arrays are parsed even when discarded so malformed/truncated
        # chunks cannot use them to escape the validated reader.
        nline = self.count(_MAX_CODE)
        self.read(nline)
        nabs = self.count(_MAX_CODE)
        if nabs:
            self.align(4)
            self.read(nabs * 8)
        nlocals = self.count(1_000_000)
        for _ in range(nlocals):
            self.string()
            self.count(_MAX_CODE)
            self.count(_MAX_CODE)
        nupnames = self.count(1_000_000)
        if nupnames:
            for upvalue in p.upvalues:
                upvalue.name = self.string()
        return p


def _decode_puc_chunk(data: bytes) -> _PucProto:
    reader = _PucReader(data)
    nupvalues = reader.header()
    proto = reader.proto()
    if nupvalues != len(proto.upvalues):
        raise BinaryChunkError("PUC-Lua root upvalue count mismatch")
    if reader.pos != len(data):
        raise BinaryChunkError("trailing data in PUC-Lua chunk")
    return proto


# PUC 5.5 opcode numbers, deliberately written out so format drift fails closed.
(
    P_MOVE, P_LOADI, P_LOADF, P_LOADK, P_LOADKX, P_LOADFALSE,
    P_LFALSESKIP, P_LOADTRUE, P_LOADNIL, P_GETUPVAL, P_SETUPVAL,
    P_GETTABUP, P_GETTABLE, P_GETI, P_GETFIELD, P_SETTABUP,
    P_SETTABLE, P_SETI, P_SETFIELD, P_NEWTABLE, P_SELF, P_ADDI,
    P_ADDK, P_SUBK, P_MULK, P_MODK, P_POWK, P_DIVK, P_IDIVK,
    P_BANDK, P_BORK, P_BXORK, P_SHLI, P_SHRI, P_ADD, P_SUB,
    P_MUL, P_MOD, P_POW, P_DIV, P_IDIV, P_BAND, P_BOR, P_BXOR,
    P_SHL, P_SHR, P_MMBIN, P_MMBINI, P_MMBINK, P_UNM, P_BNOT,
    P_NOT, P_LEN, P_CONCAT, P_CLOSE, P_TBC, P_JMP, P_EQ, P_LT,
    P_LE, P_EQK, P_EQI, P_LTI, P_LEI, P_GTI, P_GEI, P_TEST,
    P_TESTSET, P_CALL, P_TAILCALL, P_RETURN, P_RETURN0, P_RETURN1,
    P_FORLOOP, P_FORPREP, P_TFORPREP, P_TFORCALL, P_TFORLOOP,
    P_SETLIST, P_CLOSURE, P_VARARG, P_GETVARG, P_ERRNNIL,
    P_VARARGPREP, P_EXTRAARG,
) = range(85)

_ARITH_REG = {
    P_ADD: Op.ADD, P_SUB: Op.SUB, P_MUL: Op.MUL, P_MOD: Op.MOD,
    P_POW: Op.POW, P_DIV: Op.DIV, P_IDIV: Op.IDIV,
    P_BAND: Op.BAND, P_BOR: Op.BOR, P_BXOR: Op.BXOR,
    P_SHL: Op.SHL, P_SHR: Op.SHR,
}
_ARITH_CONST = {
    P_ADDK: Op.ADD, P_SUBK: Op.SUB, P_MULK: Op.MUL, P_MODK: Op.MOD,
    P_POWK: Op.POW, P_DIVK: Op.DIV, P_IDIVK: Op.IDIV,
    P_BANDK: Op.BAND, P_BORK: Op.BOR, P_BXORK: Op.BXOR,
}


def _field(word: int, pos: int, size: int) -> int:
    return (word >> pos) & ((1 << size) - 1)


def _op(word): return word & 0x7F

def _a(word): return _field(word, 7, 8)

def _k(word): return _field(word, 15, 1)

def _b(word): return _field(word, 16, 8)

def _c(word): return _field(word, 24, 8)

def _vb(word): return _field(word, 16, 6)

def _vc(word): return _field(word, 22, 10)

def _bx(word): return _field(word, 15, 17)

def _sbx(word): return _bx(word) - 65535

def _ax(word): return _field(word, 7, 25)

def _sj(word): return _field(word, 7, 25) - 16777215

def _sb(word): return _b(word) - 127

def _sc(word): return _c(word) - 127


class _Translator:
    def __init__(self, source: _PucProto):
        self.source = source
        name = source.source.decode("utf-8", "replace") if source.source else "<binary>"
        self.proto = Proto(name)
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
        self.proto.children = [_Translator(child).translate() for child in source.children]
        self.captured = {
            uv.index
            for child in self.proto.children
            for uv in child.upvalues
            if uv.kind == "local"
        }
        self.next_reg = source.maxstacksize
        self.max_reg = source.maxstacksize
        self.pcmap: dict[int, int] = {}
        self.patches: list[tuple[int, str, int]] = []
        self.open_result: tuple[int, int] | None = None

    def alloc(self, count=1):
        base = self.next_reg
        self.next_reg += count
        self.max_reg = max(self.max_reg, self.next_reg)
        return base

    def emit(self, op, a=0, b=0, c=0, d=0, e=0):
        self.proto.code.append(Ins(op, a, b, c, d, e))
        return len(self.proto.code) - 1

    def addconst(self, value):
        return self.proto.add_const(value)

    def readreg(self, reg: int) -> int:
        if reg not in self.captured:
            return reg
        temp = self.alloc()
        self.emit(Op.GETCELL, temp, reg)
        return temp

    def target(self, reg: int) -> tuple[int, bool]:
        return (self.alloc(), True) if reg in self.captured else (reg, False)

    def commit(self, reg: int, temp: int, captured: bool) -> None:
        if captured:
            self.emit(Op.SETCELL, reg, temp)

    def write_from(self, reg: int, source: int) -> None:
        if reg in self.captured:
            self.emit(Op.SETCELL, reg, source)
        elif reg != source:
            self.emit(Op.MOVE, reg, source)

    def load_value(self, reg: int, value) -> None:
        target, captured = self.target(reg)
        self.emit(Op.LOADK, target, self.addconst(value))
        self.commit(reg, target, captured)

    def constant_reg(self, value) -> int:
        reg = self.alloc()
        self.emit(Op.LOADK, reg, self.addconst(value))
        return reg

    def patch_a(self, index: int, target_pc: int) -> None:
        self.patches.append((index, "a", target_pc))

    def patch_d(self, index: int, target_pc: int) -> None:
        self.patches.append((index, "d", target_pc))

    def _consume_mm(self, pc: int) -> int:
        if pc + 1 >= len(self.source.code) or _op(self.source.code[pc + 1]) not in (P_MMBIN, P_MMBINI, P_MMBINK):
            raise BinaryChunkError("arithmetic opcode without metamethod companion")
        self.pcmap[pc + 1] = len(self.proto.code)
        return pc + 2

    def _binary(self, opcode: Op, dest: int, left: int, right: int) -> None:
        out, captured = self.target(dest)
        self.emit(opcode, out, left, right)
        self.commit(dest, out, captured)

    def _comparison_jump(self, opcode: Op, left: int, right: int, accepted: int, pc: int) -> int:
        if pc + 1 >= len(self.source.code) or _op(self.source.code[pc + 1]) != P_JMP:
            raise BinaryChunkError("comparison without following jump")
        cond = self.alloc()
        self.emit(opcode, cond, left, right)
        jump = self.emit(Op.JMPIF if accepted else Op.JMPIFNOT, 0, cond)
        target = pc + 2 + _sj(self.source.code[pc + 1])
        self.patch_a(jump, target)
        self.pcmap[pc + 1] = len(self.proto.code)
        return pc + 2

    def _test_jump(self, reg: int, accepted: int, pc: int) -> int:
        if pc + 1 >= len(self.source.code) or _op(self.source.code[pc + 1]) != P_JMP:
            raise BinaryChunkError("test without following jump")
        jump = self.emit(Op.JMPIF if accepted else Op.JMPIFNOT, 0, reg)
        self.patch_a(jump, pc + 2 + _sj(self.source.code[pc + 1]))
        self.pcmap[pc + 1] = len(self.proto.code)
        return pc + 2

    def _staged_call(self, a: int, b: int, c: int, *, tail: bool = False) -> None:
        fn = self.readreg(a)
        fnstage = self.alloc()
        self.emit(Op.MOVE, fnstage, fn)

        open_mv = None
        if b == 0:
            if self.open_result is None or self.open_result[0] < a + 1:
                raise BinaryChunkError("open call has no preceding multi-result producer")
            open_start, open_mv = self.open_result
            fixed_regs = list(range(a + 1, open_start))
        else:
            fixed_regs = list(range(a + 1, a + b))

        args_base = self.alloc(len(fixed_regs)) if fixed_regs else self.alloc(0)
        for offset, reg in enumerate(fixed_regs):
            self.emit(Op.MOVE, args_base + offset, self.readreg(reg))

        if tail:
            if open_mv is None:
                self.emit(Op.TAILCALL, 0, fnstage, args_base, len(fixed_regs), 0)
            else:
                self.emit(Op.TAILCALLV, 0, fnstage, args_base, len(fixed_regs), open_mv)
            self.open_result = None
            return

        want = c - 1
        if c == 0:
            result = self.alloc()
            if open_mv is None:
                self.emit(Op.CALL, result, fnstage, args_base, len(fixed_regs), -1)
            else:
                self.emit(Op.CALLV, result, fnstage, args_base, len(fixed_regs), open_mv)
            self.open_result = (a, result)
            return

        if want == 0:
            if open_mv is None:
                self.emit(Op.CALL, 0, fnstage, args_base, len(fixed_regs), 0)
            else:
                self.emit(Op.CALLV, 0, fnstage, args_base, len(fixed_regs), open_mv)
            self.open_result = None
            return

        results = self.alloc(want)
        if open_mv is None:
            self.emit(Op.CALL, results, fnstage, args_base, len(fixed_regs), want)
        else:
            # CALLV stores all values as a MultiValue; unpack the requested prefix.
            mvresult = self.alloc()
            self.emit(Op.CALLV, mvresult, fnstage, args_base, len(fixed_regs), open_mv)
            self.emit(Op.UNPACK, results, mvresult, want)
        for offset in range(want):
            self.write_from(a + offset, results + offset)
        self.open_result = None

    def _return(self, a: int, b: int, close: bool) -> None:
        if close:
            self.emit(Op.PCLOSE, 0)
        if b == 0:
            if self.open_result is None or self.open_result[0] < a:
                raise BinaryChunkError("open return has no preceding multi-result producer")
            start, mv = self.open_result
            fixed = list(range(a, start))
            base = self.alloc(len(fixed)) if fixed else self.alloc(0)
            for offset, reg in enumerate(fixed):
                self.emit(Op.MOVE, base + offset, self.readreg(reg))
            self.emit(Op.RETURNV, base, len(fixed), mv)
        else:
            count = b - 1
            base = self.alloc(count) if count else self.alloc(0)
            for offset in range(count):
                self.emit(Op.MOVE, base + offset, self.readreg(a + offset))
            self.emit(Op.RETURN, base, count)
        self.open_result = None

    def translate(self) -> Proto:
        code = self.source.code
        pc = 0
        while pc < len(code):
            self.pcmap[pc] = len(self.proto.code)
            word = code[pc]
            opcode = _op(word)
            a, b, c, k = _a(word), _b(word), _c(word), _k(word)

            if opcode == P_MOVE:
                self.write_from(a, self.readreg(b)); pc += 1
            elif opcode == P_LOADI:
                self.load_value(a, _sbx(word)); pc += 1
            elif opcode == P_LOADF:
                self.load_value(a, float(_sbx(word))); pc += 1
            elif opcode == P_LOADK:
                if _bx(word) >= len(self.source.constants): raise BinaryChunkError("constant index out of range")
                self.load_value(a, self.source.constants[_bx(word)]); pc += 1
            elif opcode == P_LOADKX:
                if pc + 1 >= len(code) or _op(code[pc + 1]) != P_EXTRAARG: raise BinaryChunkError("LOADKX without EXTRAARG")
                index = _ax(code[pc + 1])
                if index >= len(self.source.constants): raise BinaryChunkError("constant index out of range")
                self.load_value(a, self.source.constants[index])
                self.pcmap[pc + 1] = len(self.proto.code); pc += 2
            elif opcode == P_LOADFALSE:
                self.load_value(a, False); pc += 1
            elif opcode == P_LFALSESKIP:
                self.load_value(a, False)
                self.pcmap[pc + 1] = len(self.proto.code)
                pc += 2
            elif opcode == P_LOADTRUE:
                self.load_value(a, True); pc += 1
            elif opcode == P_LOADNIL:
                for reg in range(a, a + b + 1): self.load_value(reg, None)
                pc += 1
            elif opcode == P_GETUPVAL:
                out, captured = self.target(a); self.emit(Op.GETUPVAL, out, b); self.commit(a, out, captured); pc += 1
            elif opcode == P_SETUPVAL:
                self.emit(Op.SETUPVAL, b, self.readreg(a)); pc += 1
            elif opcode == P_GETTABUP:
                if b >= len(self.source.upvalues) or c >= len(self.source.constants): raise BinaryChunkError("GETTABUP index out of range")
                table = self.alloc(); self.emit(Op.GETUPVAL, table, b)
                key = self.constant_reg(self.source.constants[c])
                out, captured = self.target(a); self.emit(Op.GETTABLE, out, table, key); self.commit(a, out, captured); pc += 1
            elif opcode == P_GETTABLE:
                out, captured = self.target(a); self.emit(Op.GETTABLE, out, self.readreg(b), self.readreg(c)); self.commit(a, out, captured); pc += 1
            elif opcode == P_GETI:
                key = self.constant_reg(c); out, captured = self.target(a); self.emit(Op.GETTABLE, out, self.readreg(b), key); self.commit(a, out, captured); pc += 1
            elif opcode == P_GETFIELD:
                if c >= len(self.source.constants): raise BinaryChunkError("GETFIELD constant out of range")
                key = self.constant_reg(self.source.constants[c]); out, captured = self.target(a); self.emit(Op.GETTABLE, out, self.readreg(b), key); self.commit(a, out, captured); pc += 1
            elif opcode == P_SETTABUP:
                if a >= len(self.source.upvalues) or b >= len(self.source.constants): raise BinaryChunkError("SETTABUP index out of range")
                table = self.alloc(); self.emit(Op.GETUPVAL, table, a); key = self.constant_reg(self.source.constants[b])
                value = self.constant_reg(self.source.constants[c]) if k else self.readreg(c)
                self.emit(Op.SETTABLE, table, key, value); pc += 1
            elif opcode == P_SETTABLE:
                value = self.constant_reg(self.source.constants[c]) if k else self.readreg(c)
                self.emit(Op.SETTABLE, self.readreg(a), self.readreg(b), value); pc += 1
            elif opcode == P_SETI:
                value = self.constant_reg(self.source.constants[c]) if k else self.readreg(c)
                self.emit(Op.SETTABLE, self.readreg(a), self.constant_reg(b), value); pc += 1
            elif opcode == P_SETFIELD:
                if b >= len(self.source.constants): raise BinaryChunkError("SETFIELD constant out of range")
                value = self.constant_reg(self.source.constants[c]) if k else self.readreg(c)
                self.emit(Op.SETTABLE, self.readreg(a), self.constant_reg(self.source.constants[b]), value); pc += 1
            elif opcode == P_NEWTABLE:
                out, captured = self.target(a); self.emit(Op.NEWTABLE, out); self.commit(a, out, captured)
                if k:
                    if pc + 1 >= len(code) or _op(code[pc + 1]) != P_EXTRAARG: raise BinaryChunkError("NEWTABLE without EXTRAARG")
                    self.pcmap[pc + 1] = len(self.proto.code); pc += 2
                else: pc += 1
            elif opcode == P_SELF:
                obj = self.readreg(b); self.write_from(a + 1, obj)
                if c >= len(self.source.constants): raise BinaryChunkError("SELF constant out of range")
                key = self.constant_reg(self.source.constants[c]); out, captured = self.target(a); self.emit(Op.GETTABLE, out, obj, key); self.commit(a, out, captured); pc += 1
            elif opcode == P_ADDI:
                self._binary(Op.ADD, a, self.readreg(b), self.constant_reg(_sc(word))); pc = self._consume_mm(pc)
            elif opcode in _ARITH_CONST:
                if c >= len(self.source.constants): raise BinaryChunkError("arithmetic constant out of range")
                self._binary(_ARITH_CONST[opcode], a, self.readreg(b), self.constant_reg(self.source.constants[c])); pc = self._consume_mm(pc)
            elif opcode == P_SHLI:
                self._binary(Op.SHL, a, self.constant_reg(_sc(word)), self.readreg(b)); pc = self._consume_mm(pc)
            elif opcode == P_SHRI:
                self._binary(Op.SHR, a, self.readreg(b), self.constant_reg(_sc(word))); pc = self._consume_mm(pc)
            elif opcode in _ARITH_REG:
                self._binary(_ARITH_REG[opcode], a, self.readreg(b), self.readreg(c)); pc = self._consume_mm(pc)
            elif opcode in (P_MMBIN, P_MMBINI, P_MMBINK):
                raise BinaryChunkError("orphan PUC-Lua metamethod instruction")
            elif opcode in (P_UNM, P_BNOT, P_NOT, P_LEN):
                op = {P_UNM: Op.NEG, P_BNOT: Op.BNOT, P_NOT: Op.NOT, P_LEN: Op.LEN}[opcode]
                out, captured = self.target(a); self.emit(op, out, self.readreg(b)); self.commit(a, out, captured); pc += 1
            elif opcode == P_CONCAT:
                count = b
                if count == 0: self.load_value(a, b"")
                else:
                    current = self.readreg(a + count - 1)
                    for reg in range(a + count - 2, a - 1, -1):
                        temp = self.alloc(); self.emit(Op.CONCAT, temp, self.readreg(reg), current); current = temp
                    self.write_from(a, current)
                pc += 1
            elif opcode == P_CLOSE:
                self.emit(Op.PCLOSE, a); pc += 1
            elif opcode == P_TBC:
                self.emit(Op.PTBC, a); pc += 1
            elif opcode == P_JMP:
                jump = self.emit(Op.JMP, 0); self.patch_a(jump, pc + 1 + _sj(word)); pc += 1
            elif opcode in (P_EQ, P_LT, P_LE):
                op = {P_EQ: Op.EQ, P_LT: Op.LT, P_LE: Op.LE}[opcode]
                pc = self._comparison_jump(op, self.readreg(a), self.readreg(b), k, pc)
            elif opcode == P_EQK:
                if b >= len(self.source.constants): raise BinaryChunkError("EQK constant out of range")
                pc = self._comparison_jump(Op.EQ, self.readreg(a), self.constant_reg(self.source.constants[b]), k, pc)
            elif opcode in (P_EQI, P_LTI, P_LEI, P_GTI, P_GEI):
                imm = float(_sb(word)) if c else _sb(word); immreg = self.constant_reg(imm); areg = self.readreg(a)
                if opcode == P_EQI: op, left, right = Op.EQ, areg, immreg
                elif opcode == P_LTI: op, left, right = Op.LT, areg, immreg
                elif opcode == P_LEI: op, left, right = Op.LE, areg, immreg
                elif opcode == P_GTI: op, left, right = Op.LT, immreg, areg
                else: op, left, right = Op.LE, immreg, areg
                pc = self._comparison_jump(op, left, right, k, pc)
            elif opcode == P_TEST:
                pc = self._test_jump(self.readreg(a), k, pc)
            elif opcode == P_TESTSET:
                if pc + 1 >= len(code) or _op(code[pc + 1]) != P_JMP: raise BinaryChunkError("TESTSET without following jump")
                source = self.readreg(b)
                skip = self.emit(Op.JMPIFNOT if k else Op.JMPIF, 0, source)
                self.patch_a(skip, pc + 2)
                self.write_from(a, source)
                jump = self.emit(Op.JMP, 0); self.patch_a(jump, pc + 2 + _sj(code[pc + 1]))
                self.pcmap[pc + 1] = len(self.proto.code); pc += 2
            elif opcode == P_CALL:
                self._staged_call(a, b, c); pc += 1
            elif opcode == P_TAILCALL:
                if k: self.emit(Op.PCLOSE, 0)
                self._staged_call(a, b, c, tail=True); pc += 1
            elif opcode == P_RETURN:
                self._return(a, b, bool(k)); pc += 1
            elif opcode == P_RETURN0:
                self.emit(Op.RETURN, 0, 0); self.open_result = None; pc += 1
            elif opcode == P_RETURN1:
                value = self.readreg(a); base = self.alloc(); self.emit(Op.MOVE, base, value); self.emit(Op.RETURN, base, 1); self.open_result = None; pc += 1
            elif opcode == P_FORPREP:
                ins = self.emit(Op.PFORPREP, a, 0, 0, 0); self.patch_d(ins, pc + _bx(word) + 2); pc += 1
            elif opcode == P_FORLOOP:
                ins = self.emit(Op.PFORLOOP, a, 0, 0, 0); self.patch_d(ins, pc + 1 - _bx(word)); pc += 1
            elif opcode == P_TFORPREP:
                ins = self.emit(Op.PTFORPREP, a, 0, 0, 0); self.patch_d(ins, pc + 1 + _bx(word)); pc += 1
            elif opcode == P_TFORCALL:
                fn = self.readreg(a); fnstage = self.alloc(); self.emit(Op.MOVE, fnstage, fn)
                args = self.alloc(2); self.emit(Op.MOVE, args, self.readreg(a + 1)); self.emit(Op.MOVE, args + 1, self.readreg(a + 3))
                results = self.alloc(c); self.emit(Op.CALL, results, fnstage, args, 2, c)
                for offset in range(c): self.write_from(a + 3 + offset, results + offset)
                pc += 1
            elif opcode == P_TFORLOOP:
                ins = self.emit(Op.PTFORLOOP, a, 0, 0, 0); self.patch_d(ins, pc + 1 - _bx(word)); pc += 1
            elif opcode == P_SETLIST:
                table = self.readreg(a); n, base_index = _vb(word), _vc(word)
                if k:
                    if pc + 1 >= len(code) or _op(code[pc + 1]) != P_EXTRAARG: raise BinaryChunkError("SETLIST without EXTRAARG")
                    base_index += _ax(code[pc + 1]) * 1024; self.pcmap[pc + 1] = len(self.proto.code); advance = 2
                else: advance = 1
                if n == 0:
                    if self.open_result is None or self.open_result[0] < a + 1: raise BinaryChunkError("open SETLIST has no multi-result producer")
                    start, mv = self.open_result
                    for offset, reg in enumerate(range(a + 1, start), 1): self.emit(Op.SETTABLE, table, self.constant_reg(base_index + offset), self.readreg(reg))
                    self.emit(Op.SETLISTV, table, base_index + (start - (a + 1)) + 1, mv); self.open_result = None
                else:
                    for offset in range(1, n + 1): self.emit(Op.SETTABLE, table, self.constant_reg(base_index + offset), self.readreg(a + offset))
                pc += advance
            elif opcode == P_CLOSURE:
                index = _bx(word)
                if index >= len(self.proto.children): raise BinaryChunkError("child prototype index out of range")
                out, captured = self.target(a); self.emit(Op.CLOSURE, out, index); self.commit(a, out, captured); pc += 1
            elif opcode == P_VARARG:
                count = c - 1
                vatab = b if k else -1
                if c == 0:
                    result = self.alloc(); self.emit(Op.PVARARG, result, -1, vatab); self.open_result = (a, result)
                else:
                    results = self.alloc(count); self.emit(Op.PVARARG, results, count, vatab)
                    for offset in range(count): self.write_from(a + offset, results + offset)
                    self.open_result = None
                pc += 1
            elif opcode == P_GETVARG:
                out, captured = self.target(a); self.emit(Op.PGETVARG, out, self.readreg(c)); self.commit(a, out, captured); pc += 1
            elif opcode == P_ERRNNIL:
                bx = _bx(word); name = self.source.constants[bx - 1] if bx and bx - 1 < len(self.source.constants) else b"?"
                self.emit(Op.CHECKNIL, self.readreg(a), self.addconst(name)); pc += 1
            elif opcode == P_VARARGPREP:
                pc += 1
            elif opcode == P_EXTRAARG:
                raise BinaryChunkError("orphan PUC-Lua EXTRAARG")
            else:
                raise BinaryChunkError(f"unsupported PUC-Lua opcode {opcode}")

        self.pcmap[len(code)] = len(self.proto.code)
        for index, field, target_pc in self.patches:
            if target_pc not in self.pcmap:
                raise BinaryChunkError("PUC-Lua jump target out of range")
            target = self.pcmap[target_pc]
            old = self.proto.code[index]
            if field == "a": self.proto.code[index] = Ins(old.op, target, old.b, old.c, old.d, old.e)
            else: self.proto.code[index] = Ins(old.op, old.a, old.b, old.c, target, old.e)

        self.proto.register_count = self.max_reg
        return self.proto


def load_puc55_chunk(data: bytes) -> Proto:
    return _Translator(_decode_puc_chunk(data)).translate()


def fresh_loaded_closure(proto: Proto, environment) -> Closure:
    """Create the fresh root upvalues required by Lua's load semantics."""
    upvalues = [Cell(None) for _ in proto.upvalues]
    if upvalues:
        upvalues[0].value = environment
    return Closure(proto, upvalues, environment)
