from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

from .bytecode import Ins, Op, Proto


class IRValueKind(str, Enum):
    """Kinds of values understood by the small typed optimizer IR.

    REGISTER is a materialized Lua VM register. CONSTANT is a Proto constant-pool
    entry. UPVALUE is a direct lexical-cell read that is safe to rematerialize at
    a deopt point. ENVIRONMENT marks the root ``_ENV`` register when static
    analysis proves that the function never assigns it.
    """

    REGISTER = "register"
    CONSTANT = "constant"
    UPVALUE = "upvalue"
    ENVIRONMENT = "environment"


@dataclass(frozen=True, slots=True)
class IRValue:
    kind: IRValueKind
    index: int
    type_name: str = "Any"
    is_environment: bool = False

    @classmethod
    def register(cls, index: int, type_name: str = "Any") -> "IRValue":
        return cls(IRValueKind.REGISTER, index, type_name)

    @classmethod
    def constant(cls, index: int, type_name: str = "Any") -> "IRValue":
        return cls(IRValueKind.CONSTANT, index, type_name)

    @classmethod
    def upvalue(
        cls, index: int, type_name: str = "Any", *, is_environment: bool = False
    ) -> "IRValue":
        return cls(IRValueKind.UPVALUE, index, type_name, is_environment)

    @classmethod
    def environment(cls, index: int) -> "IRValue":
        return cls(IRValueKind.ENVIRONMENT, index, "table", True)


@dataclass(frozen=True, slots=True)
class TypedIRInstruction:
    """One optimized LuaPyre instruction plus statically known source values."""

    pc: int
    ins: Ins
    sources: tuple[tuple[int, IRValue], ...]
    result_type: str | None = None
    specialization: str | None = None
    dead_definition: bool = False

    def value_for(self, register: int) -> IRValue:
        for source_reg, value in self.sources:
            if source_reg == register:
                return value
        return IRValue.register(register)


@dataclass(frozen=True, slots=True)
class TypedIRPlan:
    """Backend-neutral typed facts for a set of basic blocks.

    The plan deliberately does not contain Python syntax. Python AST is one
    backend; later native/AOT backends can consume the same constant, type,
    specialization, and deoptimization facts.
    """

    instructions: tuple[TypedIRInstruction, ...]
    state_before: tuple[tuple[int, tuple[tuple[int, IRValue], ...]], ...]
    cache_sites: tuple[int, ...]

    def instruction(self, pc: int) -> TypedIRInstruction | None:
        for item in self.instructions:
            if item.pc == pc:
                return item
        return None

    def virtual_state(self, pc: int) -> dict[int, IRValue]:
        for state_pc, entries in self.state_before:
            if state_pc == pc:
                return dict(entries)
        return {}


_BINARY_OPS = frozenset(
    {
        Op.ADD,
        Op.ADD_I,
        Op.ADD_F,
        Op.SUB,
        Op.SUB_I,
        Op.SUB_F,
        Op.MUL,
        Op.MUL_I,
        Op.MUL_F,
        Op.DIV,
        Op.IDIV,
        Op.MOD,
        Op.POW,
        Op.BAND,
        Op.BOR,
        Op.BXOR,
        Op.SHL,
        Op.SHR,
        Op.CONCAT,
        Op.EQ,
        Op.LT,
        Op.LE,
    }
)
_UNARY_OPS = frozenset({Op.LEN, Op.BNOT, Op.NEG, Op.NOT, Op.TOBOOL})


def _constant_type(value: object) -> str:
    if value is None:
        return "nil"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "float"
    if isinstance(value, bytes):
        return "string"
    return "Any"


def _result_type(ins: Ins, source_types: dict[int, str]) -> str | None:
    op = ins.op
    if op in (Op.ADD_I, Op.SUB_I, Op.MUL_I, Op.BAND, Op.BOR, Op.BXOR, Op.SHL, Op.SHR, Op.BNOT, Op.LEN):
        return "integer"
    if op in (Op.ADD_F, Op.SUB_F, Op.MUL_F, Op.DIV, Op.POW):
        return "float"
    if op in (Op.EQ, Op.LT, Op.LE, Op.NOT, Op.TOBOOL):
        return "boolean"
    if op is Op.CONCAT:
        return "string"
    if op is Op.NEWTABLE:
        return "table"
    if op in (Op.MOVE, Op.LOCAL):
        return source_types.get(ins.b, "Any")
    if op in (Op.ADD, Op.SUB, Op.MUL, Op.MOD, Op.IDIV):
        left = source_types.get(ins.b, "Any")
        right = source_types.get(ins.c, "Any")
        if left == right == "integer":
            return "integer"
        if left in ("integer", "float", "number") and right in ("integer", "float", "number"):
            return "number"
    return None


def _reads(ins: Ins) -> tuple[int, ...]:
    op = ins.op
    if op in (Op.LOADK, Op.GETGLOBAL, Op.GETUPVAL, Op.CLOSURE, Op.NEWTABLE, Op.JMP, Op.HALT):
        return ()
    if op in (Op.MOVE, Op.LOCAL, Op.GETCELL, Op.LEN, Op.BNOT, Op.NEG, Op.NOT, Op.TOBOOL):
        return (ins.b,)
    if op is Op.SETGLOBAL:
        return (ins.a,)
    if op is Op.SETUPVAL:
        return (ins.b,)
    if op is Op.SETCELL:
        return (ins.a, ins.b)
    if op in _BINARY_OPS or op is Op.GETTABLE:
        return (ins.b, ins.c)
    if op is Op.SETTABLE:
        return (ins.a, ins.b, ins.c)
    if op is Op.SETLISTV:
        return (ins.a, ins.c)
    if op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
        return (ins.b,)
    if op in (Op.FORPREP, Op.FORLOOP, Op.JFORLOOP):
        return (ins.a, ins.b, ins.c)
    if op is Op.GUARD or op in (Op.TBC, Op.CHECKNIL):
        return (ins.a,)
    if op is Op.CALL:
        return (ins.b, *range(ins.c, ins.c + max(0, ins.d)))
    if op is Op.CALLV:
        return (ins.b, ins.e, *range(ins.c, ins.c + max(0, ins.d)))
    if op in (Op.TAILCALL, Op.TAILCALLV):
        extra = (ins.e,) if op is Op.TAILCALLV else ()
        return (ins.b, *extra, *range(ins.c, ins.c + max(0, ins.d)))
    if op is Op.RETURN:
        return tuple(range(ins.a, ins.a + max(0, ins.b)))
    if op is Op.RETURNV:
        return (*range(ins.a, ins.a + max(0, ins.b)), ins.c)
    if op is Op.UNPACK:
        return (ins.b,)
    return ()


def _writes(ins: Ins) -> tuple[int, ...]:
    op = ins.op
    if op in (
        Op.LOADK,
        Op.MOVE,
        Op.LOCAL,
        Op.GETGLOBAL,
        Op.GETUPVAL,
        Op.GETCELL,
        Op.CLOSURE,
        Op.NEWTABLE,
        Op.GETTABLE,
        Op.LEN,
        Op.BNOT,
        Op.NEG,
        Op.NOT,
        Op.TOBOOL,
    ) or op in _BINARY_OPS:
        return (ins.a,)
    if op is Op.SETCELL:
        return (ins.a,)
    if op is Op.FORPREP:
        return (ins.a, ins.b, ins.c)
    if op in (Op.FORLOOP, Op.JFORLOOP):
        return (ins.a,)
    if op is Op.CALL:
        if ins.e == 0:
            return ()
        if ins.e == -1:
            return (ins.a,)
        return tuple(range(ins.a, ins.a + max(0, ins.e)))
    if op is Op.CALLV:
        return (ins.a,)
    if op is Op.VARARG:
        if ins.b == -1:
            return (ins.a,)
        return tuple(range(ins.a, ins.a + max(0, ins.b)))
    if op is Op.UNPACK:
        return tuple(range(ins.a, ins.a + max(0, ins.c)))
    return ()


def _virtualizable_use(ins: Ins, register: int) -> bool:
    """Whether the first Python-AST backend consumes this operand from IR.

    Keeping this explicit is important for deoptimization correctness: a value
    is never declared dead merely because analysis knows it if the current
    backend still reads its materialized register. Later backends can widen this
    set without changing the IR representation.
    """

    if ins.op is Op.GETTABLE:
        return register in (ins.b, ins.c)
    if ins.op is Op.SETTABLE:
        return register in (ins.a, ins.b)
    return False


class TypedIRCompiler:
    """Lower fully typed bytecode blocks into a compact optimization IR.

    This is intentionally small. It performs the data-flow work that is useful
    regardless of backend: constant propagation, conservative type propagation,
    recognition of constant-key/global table accesses, and removal of virtual
    LOADK/GETUPVAL temporaries. The Python backend is responsible only for
    turning these facts into AST and guards.
    """

    def __init__(self, proto: Proto):
        self.proto = proto

    def _stable_environment(self) -> bool:
        env = self.proto.env_reg
        if env < 0:
            return False
        return all(env not in _writes(ins) for ins in self.proto.code)

    def compile(self, blocks: tuple[tuple[tuple[int, Ins], ...], ...]) -> TypedIRPlan:
        lowered: list[TypedIRInstruction] = []
        states: list[tuple[int, tuple[tuple[int, IRValue], ...]]] = []
        candidate_defs: dict[int, int] = {}
        materialized_reads: set[int] = set()
        stable_env = self._stable_environment()

        for block in blocks:
            facts: dict[int, IRValue] = {}
            types: dict[int, str] = {
                index: typ.name
                for index, typ in enumerate(self.proto.param_types)
            }
            if stable_env:
                facts[self.proto.env_reg] = IRValue.environment(self.proto.env_reg)
                types[self.proto.env_reg] = "table"

            for pc, ins in block:
                states.append((pc, tuple(sorted(facts.items()))))
                sources: list[tuple[int, IRValue]] = []
                source_types: dict[int, str] = {}
                analytical_sources: dict[int, IRValue] = {}
                for reg in _reads(ins):
                    known = facts.get(reg, IRValue.register(reg, types.get(reg, "Any")))
                    analytical_sources[reg] = known
                    source_types[reg] = known.type_name
                    emitted = known if _virtualizable_use(ins, reg) else IRValue.register(
                        reg, types.get(reg, known.type_name)
                    )
                    sources.append((reg, emitted))
                    if emitted.kind is IRValueKind.REGISTER:
                        materialized_reads.add(reg)

                specialization = None
                if ins.op in (Op.GETTABLE, Op.SETTABLE):
                    table_reg = ins.b if ins.op is Op.GETTABLE else ins.a
                    key_reg = ins.c if ins.op is Op.GETTABLE else ins.b
                    table_value = analytical_sources.get(
                        table_reg, IRValue.register(table_reg, types.get(table_reg, "Any"))
                    )
                    key_value = analytical_sources.get(
                        key_reg, IRValue.register(key_reg, types.get(key_reg, "Any"))
                    )
                    if key_value.kind is IRValueKind.CONSTANT:
                        key = self.proto.constants[key_value.index]
                        valid_float = not (type(key) is float and math.isnan(key))
                        if (
                            isinstance(key, (bytes, int, float))
                            and type(key) is not bool
                            and valid_float
                        ):
                            if table_value.is_environment and isinstance(key, bytes):
                                specialization = (
                                    "global_get" if ins.op is Op.GETTABLE else "global_set"
                                )
                            else:
                                specialization = (
                                    "table_get_const" if ins.op is Op.GETTABLE else "table_set_const"
                                )
                    # Only specialized table ops consume virtual operands in the
                    # current AST backend. Generic accesses keep registers live.
                    if specialization is None:
                        sources = [
                            (reg, IRValue.register(reg, types.get(reg, "Any")))
                            for reg in _reads(ins)
                        ]
                        materialized_reads.update(_reads(ins))

                result_type = _result_type(ins, source_types)
                item = TypedIRInstruction(
                    pc,
                    ins,
                    tuple(sources),
                    result_type=result_type,
                    specialization=specialization,
                )
                lowered.append(item)

                # Calls can mutate lexical cells through other closures. Do not
                # carry a direct-upvalue rematerialization fact across a call.
                if ins.op in (Op.CALL, Op.CALLV, Op.TAILCALL, Op.TAILCALLV):
                    facts = {
                        reg: value
                        for reg, value in facts.items()
                        if value.kind is not IRValueKind.UPVALUE
                    }

                if ins.op is Op.SETUPVAL:
                    facts = {
                        reg: value
                        for reg, value in facts.items()
                        if not (
                            value.kind is IRValueKind.UPVALUE
                            and value.index == ins.a
                        )
                    }

                writes = _writes(ins)
                for reg in writes:
                    facts.pop(reg, None)
                    if result_type is not None:
                        types[reg] = result_type
                    else:
                        types.pop(reg, None)

                if ins.op is Op.LOADK:
                    typ = _constant_type(self.proto.constants[ins.b])
                    facts[ins.a] = IRValue.constant(ins.b, typ)
                    types[ins.a] = typ
                    candidate_defs[pc] = ins.a
                elif ins.op is Op.GETUPVAL:
                    desc = self.proto.upvalues[ins.b] if ins.b < len(self.proto.upvalues) else None
                    is_env = desc is not None and desc.name == "_ENV"
                    value = IRValue.upvalue(ins.b, "table" if is_env else "Any", is_environment=is_env)
                    facts[ins.a] = value
                    types[ins.a] = value.type_name
                    candidate_defs[pc] = ins.a
                elif ins.op in (Op.MOVE, Op.LOCAL):
                    source = analytical_sources.get(ins.b)
                    if source is not None and source.kind is IRValueKind.CONSTANT:
                        facts[ins.a] = source
                        types[ins.a] = source.type_name
                elif ins.op is Op.GUARD:
                    expected = self.proto.constants[ins.b]
                    if isinstance(expected, str):
                        types[ins.a] = expected

        # A propagated LOADK/GETUPVAL is virtual only when no backend operation
        # still needs the physical register. Cross-block uses are naturally
        # conservative because facts restart at every block and therefore show
        # up as REGISTER reads here.
        dead_pcs = {
            pc for pc, dest in candidate_defs.items() if dest not in materialized_reads
        }
        lowered = [
            replace(item, dead_definition=True)
            if item.pc in dead_pcs
            else item
            for item in lowered
        ]
        cache_sites = tuple(
            item.pc
            for item in lowered
            if item.specialization in ("global_get", "table_get_const")
        )
        return TypedIRPlan(tuple(lowered), tuple(states), cache_sites)
