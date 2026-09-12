from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum, auto
from .typesys import LuaType, ANY


class Op(IntEnum):
    LOADK = auto(); MOVE = auto(); LOCAL = auto()
    GETGLOBAL = auto(); SETGLOBAL = auto(); GETUPVAL = auto(); SETUPVAL = auto(); GETCELL = auto(); SETCELL = auto(); CLOSURE = auto()
    NEWTABLE = auto(); GETTABLE = auto(); SETTABLE = auto(); SETLISTV = auto(); LEN = auto()
    ADD = auto(); ADD_I = auto(); ADD_F = auto(); SUB = auto(); SUB_I = auto(); SUB_F = auto(); MUL = auto(); MUL_I = auto(); MUL_F = auto(); DIV = auto(); IDIV = auto(); MOD = auto(); POW = auto()
    BAND = auto(); BOR = auto(); BXOR = auto(); SHL = auto(); SHR = auto(); BNOT = auto(); CONCAT = auto(); NEG = auto(); NOT = auto(); TOBOOL = auto(); EQ = auto(); LT = auto(); LE = auto()
    JMP = auto(); JMPIF = auto(); JMPIFNOT = auto(); JMPIFNIL = auto(); FORPREP = auto(); FORLOOP = auto()
    CALL = auto(); CALLV = auto(); TAILCALL = auto(); TAILCALLV = auto(); VARARG = auto(); UNPACK = auto()
    TBC = auto(); CLOSE = auto(); CHECKNIL = auto()
    RETURN = auto(); RETURNV = auto(); GUARD = auto(); HALT = auto()

    # Compatibility instructions used only by translated PUC-Lua 5.5 chunks.
    # Keeping them distinct from LuaPyre's compiler opcodes avoids bending the
    # native VM layout around PUC's register conventions.
    PFORPREP = auto(); PFORLOOP = auto()
    PTFORPREP = auto(); PTFORLOOP = auto()
    PTBC = auto(); PCLOSE = auto()
    PVARARG = auto(); PGETVARG = auto()


@dataclass(frozen=True, slots=True)
class Ins:
    op: Op
    a: int = 0
    b: int = 0
    c: int = 0
    d: int = 0
    e: int = 0


@dataclass(frozen=True, slots=True)
class UpvalueDesc:
    kind: str
    index: int
    name: str


@dataclass(slots=True)
class Proto:
    name: str
    code: list[Ins] = field(default_factory=list)
    constants: list[object] = field(default_factory=list)
    children: list[Proto] = field(default_factory=list)
    upvalues: list[UpvalueDesc] = field(default_factory=list)
    register_count: int = 0
    param_count: int = 0
    param_types: list[LuaType] = field(default_factory=list)
    return_types: list[LuaType] = field(default_factory=lambda: [ANY])
    is_vararg: bool = False
    vararg_name_reg: int = -1
    vararg_type: LuaType = ANY
    env_reg: int = -1

    def add_const(self, value):
        for i, current in enumerate(self.constants):
            if type(current) is type(value) and current == value:
                return i
        self.constants.append(value)
        return len(self.constants) - 1

    def disassemble(self) -> str:
        return "\n".join(
            f"{i:04d} {ins.op.name:<11} {ins.a:>3} {ins.b:>3} {ins.c:>3} {ins.d:>3} {ins.e:>3}"
            for i, ins in enumerate(self.code)
        )


@dataclass(slots=True)
class Cell:
    value: object = None


@dataclass(slots=True)
class Closure:
    proto: Proto
    upvalues: list[Cell] = field(default_factory=list)
    env: object = None
