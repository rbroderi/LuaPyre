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
    specialization, loop-invariance, and deoptimization facts.
    """

    instructions: tuple[TypedIRInstruction, ...]
    state_before: tuple[tuple[int, tuple[tuple[int, IRValue], ...]], ...]
    cache_sites: tuple[int, ...]
    invariant_sites: tuple[int, ...]

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

# Any of these can invalidate a value read from _ENV. SETTABLE is deliberately
# broad: without alias analysis an arbitrary table register could alias _ENV.
# This conservative set lets the optimizer hoist only when no Lua operation in
# the region can mutate the environment table or re-enter code that can do so.
_INVARIANT_GLOBAL_BARRIERS = frozenset(
    {
        Op.SETTABLE,
        Op.SETGLOBAL,
        Op.SETUPVAL,
        Op.CALL,
        Op.CALLV,
        Op.TAILCALL,
        Op.TAILCALLV,
    }
)


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
    if op in (
        Op.ADD_I,
        Op.SUB_I,
        Op.MUL_I,
        Op.BAND,
        Op.BOR,
        Op.BXOR,
        Op.SHL,
        Op.SHR,
        Op.BNOT,
        Op.LEN,
    ):
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
        if left in ("integer", "float", "number") and right in (
            "integer",
            "float",
            "number",
        ):
            return "number"
    return None


def _reads(ins: Ins) -> tuple[int, ...]:
    op = ins.op
    if op in (
        Op.LOADK,
        Op.GETGLOBAL,
        Op.GETUPVAL,
        Op.CLOSURE,
        Op.NEWTABLE,
        Op.JMP,
        Op.HALT,
    ):
        return ()
    if op in (
        Op.MOVE,
        Op.LOCAL,
        Op.GETCELL,
        Op.LEN,
        Op.BNOT,
        Op.NEG,
        Op.NOT,
        Op.TOBOOL,
    ):
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


def _cfg_targets(ins: Ins, fallthrough: int) -> tuple[int, ...]:
    if ins.op is Op.JMP:
        return (ins.a,)
    if ins.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
        return (ins.a, fallthrough)
    if ins.op is Op.FORPREP:
        return (ins.d, fallthrough)
    if ins.op in (Op.FORLOOP, Op.JFORLOOP):
        return (ins.d, fallthrough)
    if ins.op in (Op.RETURN, Op.RETURNV, Op.HALT, Op.TAILCALL, Op.TAILCALLV):
        return ()
    return (fallthrough,)


def _merge_identical(
    incoming: list[dict[int, object]],
    base: dict[int, object],
) -> dict[int, object]:
    """Meet predecessor facts without introducing phi nodes.

    A fact crosses a join only when every predecessor carries the exact same
    semantic value. ``base`` contains facts guaranteed independently of control
    flow (currently the stable root environment and certified parameter types).
    """

    merged = dict(base)
    if not incoming:
        return merged
    common = set(incoming[0])
    for state in incoming[1:]:
        common.intersection_update(state)
    for key in common:
        value = incoming[0][key]
        if all(state[key] == value for state in incoming[1:]):
            merged[key] = value
    return merged


class TypedIRCompiler:
    """Lower fully typed bytecode blocks into a compact optimization IR.

    This is intentionally small. It performs backend-neutral data-flow work:
    CFG fixed-point value/type propagation, recognition of constant-key/global
    table accesses, dead virtual temporary elimination, and conservative
    loop-invariance analysis. At control-flow joins a value survives only when
    every predecessor agrees, giving the optimizer SSA-like knowledge without
    needing phi nodes. The Python backend only turns these facts into AST and
    guards.
    """

    def __init__(self, proto: Proto):
        self.proto = proto

    def _stable_environment(self) -> bool:
        env = self.proto.env_reg
        if env < 0:
            return False
        return all(env not in _writes(ins) for ins in self.proto.code)

    def _base_state(
        self, stable_env: bool
    ) -> tuple[dict[int, IRValue], dict[int, str]]:
        facts: dict[int, IRValue] = {}
        types: dict[int, str] = {
            index: typ.name for index, typ in enumerate(self.proto.param_types)
        }
        if stable_env:
            facts[self.proto.env_reg] = IRValue.environment(self.proto.env_reg)
            types[self.proto.env_reg] = "table"
        return facts, types

    def _advance_state(
        self,
        block: tuple[tuple[int, Ins], ...],
        entry_facts: dict[int, IRValue],
        entry_types: dict[int, str],
    ) -> tuple[dict[int, IRValue], dict[int, str]]:
        facts = dict(entry_facts)
        types = dict(entry_types)
        for _pc, ins in block:
            analytical_sources = {
                reg: facts.get(reg, IRValue.register(reg, types.get(reg, "Any")))
                for reg in _reads(ins)
            }
            source_types = {
                reg: value.type_name for reg, value in analytical_sources.items()
            }
            result_type = _result_type(ins, source_types)

            # A call may mutate lexical cells through another closure, so a
            # direct upvalue load is not a stable rematerialization fact after
            # re-entry. Constants and the proven-stable root environment remain.
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
                        value.kind is IRValueKind.UPVALUE and value.index == ins.a
                    )
                }

            for reg in _writes(ins):
                facts.pop(reg, None)
                if result_type is not None:
                    types[reg] = result_type
                else:
                    types.pop(reg, None)

            if ins.op is Op.LOADK:
                typ = _constant_type(self.proto.constants[ins.b])
                facts[ins.a] = IRValue.constant(ins.b, typ)
                types[ins.a] = typ
            elif ins.op is Op.GETUPVAL:
                desc = (
                    self.proto.upvalues[ins.b]
                    if ins.b < len(self.proto.upvalues)
                    else None
                )
                is_env = desc is not None and desc.name == "_ENV"
                value = IRValue.upvalue(
                    ins.b,
                    "table" if is_env else "Any",
                    is_environment=is_env,
                )
                facts[ins.a] = value
                types[ins.a] = value.type_name
            elif ins.op in (Op.MOVE, Op.LOCAL):
                source = analytical_sources.get(ins.b)
                # Constants are immutable snapshots. Do not yet propagate direct
                # upvalue aliases through MOVE: reconstructing such an alias after
                # SETUPVAL/re-entry requires explicit snapshot/value-number IR.
                if source is not None and source.kind is IRValueKind.CONSTANT:
                    facts[ins.a] = source
                    types[ins.a] = source.type_name
            elif ins.op is Op.GUARD:
                expected = self.proto.constants[ins.b]
                if isinstance(expected, str):
                    types[ins.a] = expected
        return facts, types

    def _cfg_entry_states(
        self,
        blocks: tuple[tuple[tuple[int, Ins], ...], ...],
        stable_env: bool,
    ) -> tuple[tuple[dict[int, IRValue], dict[int, str]], ...]:
        if not blocks:
            return ()

        starts = {block[0][0]: index for index, block in enumerate(blocks) if block}
        predecessors: list[list[int]] = [[] for _ in blocks]
        for index, block in enumerate(blocks):
            if not block:
                continue
            pc, terminal = block[-1]
            for target in _cfg_targets(terminal, pc + 1):
                successor = starts.get(target)
                if successor is not None and index not in predecessors[successor]:
                    predecessors[successor].append(index)

        base_facts, base_types = self._base_state(stable_env)
        entry_states: list[tuple[dict[int, IRValue], dict[int, str]]] = [
            (dict(base_facts), dict(base_types)) for _ in blocks
        ]
        exit_states: list[tuple[dict[int, IRValue], dict[int, str]]] = [
            self._advance_state(block, base_facts, base_types) for block in blocks
        ]

        # The first block has an implicit predecessor from outside the compiled
        # region/function. Including the base state in its meet prevents a loop
        # backedge from inventing a fact that is not true on the first entry.
        changed = True
        while changed:
            changed = False
            for index, block in enumerate(blocks):
                incoming_facts = [exit_states[p][0] for p in predecessors[index]]
                incoming_types = [exit_states[p][1] for p in predecessors[index]]
                if index == 0:
                    incoming_facts.append(base_facts)
                    incoming_types.append(base_types)

                facts = _merge_identical(incoming_facts, base_facts)
                types = _merge_identical(incoming_types, base_types)
                new_entry = (facts, types)
                new_exit = self._advance_state(block, facts, types)
                if new_entry != entry_states[index] or new_exit != exit_states[index]:
                    entry_states[index] = new_entry
                    exit_states[index] = new_exit
                    changed = True

        return tuple(entry_states)

    def compile(self, blocks: tuple[tuple[tuple[int, Ins], ...], ...]) -> TypedIRPlan:
        lowered: list[TypedIRInstruction] = []
        states: list[tuple[int, tuple[tuple[int, IRValue], ...]]] = []
        candidate_defs: dict[int, int] = {}
        materialized_reads: set[int] = set()
        stable_env = self._stable_environment()
        region_instructions = tuple(ins for block in blocks for _pc, ins in block)
        global_hoist_safe = not any(
            ins.op in _INVARIANT_GLOBAL_BARRIERS for ins in region_instructions
        )
        entry_states = self._cfg_entry_states(blocks, stable_env)

        for block_index, block in enumerate(blocks):
            entry_facts, entry_types = entry_states[block_index]
            facts = dict(entry_facts)
            types = dict(entry_types)

            for pc, ins in block:
                states.append((pc, tuple(sorted(facts.items()))))
                sources: list[tuple[int, IRValue]] = []
                source_types: dict[int, str] = {}
                analytical_sources: dict[int, IRValue] = {}
                for reg in _reads(ins):
                    known = facts.get(
                        reg, IRValue.register(reg, types.get(reg, "Any"))
                    )
                    analytical_sources[reg] = known
                    source_types[reg] = known.type_name
                    emitted = (
                        known
                        if _virtualizable_use(ins, reg)
                        else IRValue.register(reg, types.get(reg, known.type_name))
                    )
                    sources.append((reg, emitted))
                    if emitted.kind is IRValueKind.REGISTER:
                        materialized_reads.add(reg)

                specialization = None
                if ins.op in (Op.GETTABLE, Op.SETTABLE):
                    table_reg = ins.b if ins.op is Op.GETTABLE else ins.a
                    key_reg = ins.c if ins.op is Op.GETTABLE else ins.b
                    table_value = analytical_sources.get(
                        table_reg,
                        IRValue.register(table_reg, types.get(table_reg, "Any")),
                    )
                    key_value = analytical_sources.get(
                        key_reg,
                        IRValue.register(key_reg, types.get(key_reg, "Any")),
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
                                    "global_get"
                                    if ins.op is Op.GETTABLE
                                    else "global_set"
                                )
                            else:
                                specialization = (
                                    "table_get_const"
                                    if ins.op is Op.GETTABLE
                                    else "table_set_const"
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
                lowered.append(
                    TypedIRInstruction(
                        pc,
                        ins,
                        tuple(sources),
                        result_type=result_type,
                        specialization=specialization,
                    )
                )

                # Keep the lowering state identical to the fixed-point transfer
                # function so deopt reconstruction records the exact IR values
                # that justified specialization at each instruction.
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

                for reg in _writes(ins):
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
                    desc = (
                        self.proto.upvalues[ins.b]
                        if ins.b < len(self.proto.upvalues)
                        else None
                    )
                    is_env = desc is not None and desc.name == "_ENV"
                    value = IRValue.upvalue(
                        ins.b,
                        "table" if is_env else "Any",
                        is_environment=is_env,
                    )
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

        # A propagated LOADK/GETUPVAL is virtual only when every backend use can
        # consume the IR value. CFG propagation means definitions can now die
        # even when their virtual use is in a successor block; precise state at
        # every instruction rematerializes them on a guard/deopt side exit.
        dead_pcs = {
            pc for pc, dest in candidate_defs.items() if dest not in materialized_reads
        }
        lowered = [
            replace(item, dead_definition=True)
            if item.pc in dead_pcs
            else item
            for item in lowered
        ]
        invariant_sites = (
            tuple(
                item.pc
                for item in lowered
                if item.specialization == "global_get"
            )
            if global_hoist_safe
            else ()
        )
        invariant_set = set(invariant_sites)
        cache_sites = tuple(
            item.pc
            for item in lowered
            if item.specialization in ("global_get", "table_get_const")
            and item.pc not in invariant_set
        )
        return TypedIRPlan(
            tuple(lowered),
            tuple(states),
            cache_sites,
            invariant_sites,
        )
