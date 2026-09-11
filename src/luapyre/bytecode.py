from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from .typesys import LuaType, ANY


class Op(IntEnum):
    LOADK = auto(); MOVE = auto(); GETGLOBAL = auto(); SETGLOBAL = auto()
    ADD = auto(); ADD_I = auto(); ADD_F = auto(); SUB = auto(); SUB_I = auto(); SUB_F = auto()
    MUL = auto(); MUL_I = auto(); MUL_F = auto(); DIV = auto(); IDIV = auto(); MOD = auto(); POW = auto()
    NEG = auto(); NOT = auto(); EQ = auto(); LT = auto(); LE = auto()
    JMP = auto(); JMPIFNOT = auto(); CALL = auto(); RETURN = auto(); GUARD = auto(); HALT = auto()


@dataclass(frozen=True, slots=True)
class Ins:
    op: Op
    a: int = 0
    b: int = 0
    c: int = 0


@dataclass(slots=True)
class Proto:
    name: str
    code: list[Ins] = field(default_factory=list)
    constants: list[object] = field(default_factory=list)
    children: list[Proto] = field(default_factory=list)
    register_count: int = 0
    param_count: int = 0
    param_types: list[LuaType] = field(default_factory=list)
    return_type: LuaType = ANY

    def add_const(self, value):
        try:
            return self.constants.index(value)
        except ValueError:
            self.constants.append(value)
            return len(self.constants) - 1

    def disassemble(self) -> str:
        return "\n".join(
            f"{i:04d} {ins.op.name:<10} {ins.a:>3} {ins.b:>3} {ins.c:>3}"
            for i, ins in enumerate(self.code)
        )


@dataclass(slots=True)
class Closure:
    proto: Proto
