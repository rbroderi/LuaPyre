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

    # Source-compiler quickening marker for a numeric loop whose body is
    # structurally eligible for the tier-2 JIT. Interpreter semantics are
    # identical to FORLOOP; only TieredJITVM gives it a hotness hook.
    JFORLOOP = auto()


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
    source: str | bytes | None = "=?"
    linedefined: int = 0
    lastlinedefined: int = 0
    lineinfo: list[int] = field(default_factory=list)
    # Sparse, debug-only register provenance aligned to code. Each entry maps
    # VM register -> (Lua object kind, source name), e.g. ("field", "x").
    # It is consulted only while formatting an exceptional path.
    value_origins: list[dict[int, tuple[str, str]]] = field(default_factory=list)
    # Lua-visible local metadata: (name, register, start_pc, end_pc).  Debug
    # helpers consult this table without exposing the host Python stack.
    debug_locals: list[tuple[str, int, int, int]] = field(default_factory=list)
    debug_namewhat: str = ""
    # True for a binary prototype whose symbolic debug information was
    # deliberately stripped. It still receives one line hook with a nil line.
    debug_stripped: bool = False
    # Native source compilation only emits *_I/*_F opcodes when its optional
    # type analysis has proved the operand classes (including runtime GUARDs at
    # Any -> typed boundaries). The tiered JIT may therefore omit redundant
    # type guards for those specialized opcodes. Binary-chunk translation and
    # manually constructed Proto objects deliberately default to False.
    jit_trust_types: bool = False
    # Set only for source compiled under the explicit ``-- luapyre: typed``
    # contract. Every lexical binding in such a Proto has a non-Any type after
    # inference/validation. Dynamic table/global boundaries may still emit
    # GUARD instructions before values enter typed bindings. This stronger bit
    # is intentionally separate from ``jit_trust_types`` so future JIT tiers can
    # optimize typed source more aggressively without changing plain Lua.
    jit_fully_typed: bool = False

    def add_const(self, value):
        for i, current in enumerate(self.constants):
            if type(current) is type(value) and current == value:
                return i
        self.constants.append(value)
        return len(self.constants) - 1

    def line_for_pc(self, pc: int) -> int:
        if not self.lineinfo:
            return -1
        if pc < 0:
            pc = 0
        if pc >= len(self.lineinfo):
            pc = len(self.lineinfo) - 1
        return self.lineinfo[pc]

    def disassemble(self) -> str:
        return "\n".join(
            f"{i:04d} {ins.op.name:<11} {ins.a:>3} {ins.b:>3} {ins.c:>3} {ins.d:>3} {ins.e:>3}"
            for i, ins in enumerate(self.code)
        )


@dataclass(slots=True)
class Cell:
    value: object = None
    _gc_owner: object = None
    _gc_age: int = 0


@dataclass(slots=True)
class Closure:
    proto: Proto
    upvalues: list[Cell] = field(default_factory=list)
    env: object = None
    _gc_owner: object = None
    _gc_age: int = 0
