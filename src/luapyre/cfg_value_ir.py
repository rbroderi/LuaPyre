from __future__ import annotations

from dataclasses import dataclass

from .bytecode import Ins, Op, Proto
from .typed_ir import TypedIRCompiler, TypedIRPlan
from .value_ir import (
    ValueIRCompiler,
    ValueKind,
    ValueNode,
    _BOOL_UNARY,
    _COMPARE_OPS,
    _FLOAT_BINOPS,
    _INT_BINOPS,
    _safe_scalar_type,
)


_CFG_CONTROL = frozenset({Op.JMP, Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL})
_CFG_TERMINAL = frozenset({Op.RETURN, Op.HALT})
_CFG_SUPPORTED = frozenset(
    {
        Op.LOADK,
        Op.MOVE,
        Op.LOCAL,
        *_INT_BINOPS,
        *_FLOAT_BINOPS,
        *_BOOL_UNARY,
        *_COMPARE_OPS,
        *_CFG_CONTROL,
        *_CFG_TERMINAL,
    }
)


@dataclass(frozen=True, slots=True)
class CFGValueIRBlock:
    index: int
    start: int
    end: int
    predecessors: tuple[int, ...]
    successors: tuple[int, ...]
    instructions: tuple[tuple[int, Ins], ...]
    entry_values: tuple[tuple[int, int], ...]
    exit_values: tuple[tuple[int, int], ...]
    phi_nodes: tuple[int, ...]
    condition_value: int | None
    return_values: tuple[int, ...]

    @property
    def cost(self) -> int:
        return len(self.instructions)


@dataclass(frozen=True, slots=True)
class CFGValueIRPlan:
    """Acyclic SSA-like value graph with explicit merge values.

    0.18 deliberately starts with forward-only scalar CFGs. Every block carries
    an exact virtual register state at entry/exit, merge points use phi-like
    values, and instruction cost remains attached to the original block so a
    backend can preserve path-dependent Lua fuel while still eliminating copies,
    folding constants and reusing pure expressions inside a block.
    """

    typed_plan: TypedIRPlan
    nodes: tuple[ValueNode, ...]
    blocks: tuple[CFGValueIRBlock, ...]
    folded_pcs: tuple[int, ...]
    cse_pcs: tuple[int, ...]

    def node(self, node_id: int) -> ValueNode:
        return self.nodes[node_id]

    @property
    def phi_nodes(self) -> tuple[ValueNode, ...]:
        return tuple(node for node in self.nodes if node.op == "phi")


@dataclass(frozen=True, slots=True)
class _RawBlock:
    start: int
    end: int
    instructions: tuple[tuple[int, Ins], ...]
    successor_starts: tuple[int, ...]


def _successor_starts(ins: Ins, fallthrough: int) -> tuple[int, ...]:
    if ins.op is Op.JMP:
        return (ins.a,)
    if ins.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
        return (ins.a, fallthrough)
    if ins.op in _CFG_TERMINAL:
        return ()
    return (fallthrough,)


def _build_blocks(proto: Proto) -> tuple[_RawBlock, ...] | None:
    if not proto.code or any(ins.op not in _CFG_SUPPORTED for ins in proto.code):
        return None

    leaders = {0, len(proto.code)}
    for pc, ins in enumerate(proto.code):
        if ins.op in _CFG_CONTROL:
            for target in _successor_starts(ins, pc + 1):
                if target < 0 or target > len(proto.code):
                    return None
                leaders.add(target)
            leaders.add(pc + 1)
        elif ins.op in _CFG_TERMINAL:
            leaders.add(pc + 1)

    ordered = sorted(leaders)
    raw: list[_RawBlock] = []
    for index, start in enumerate(ordered[:-1]):
        end = ordered[index + 1]
        instructions = tuple((pc, proto.code[pc]) for pc in range(start, end))
        if not instructions:
            continue
        for _pc, ins in instructions[:-1]:
            if ins.op in _CFG_CONTROL or ins.op in _CFG_TERMINAL:
                return None
        terminal = instructions[-1][1]
        successor_starts = tuple(
            target
            for target in _successor_starts(terminal, end)
            if target < len(proto.code)
        )
        raw.append(_RawBlock(start, end, instructions, successor_starts))

    by_start = {block.start: index for index, block in enumerate(raw)}
    if 0 not in by_start:
        return None
    for block in raw:
        if any(target not in by_start for target in block.successor_starts):
            return None

    reachable: set[int] = set()
    stack = [0]
    while stack:
        start = stack.pop()
        if start in reachable:
            continue
        reachable.add(start)
        block = raw[by_start[start]]
        stack.extend(block.successor_starts)

    blocks = tuple(block for block in raw if block.start in reachable)
    # The first CFG-value tranche intentionally excludes cycles. This keeps phi
    # construction single-pass and makes exact block-entry rematerialization easy
    # to audit before loop-carried values are introduced.
    for block in blocks:
        if any(target <= block.start for target in block.successor_starts):
            return None
    if not any(block.instructions[-1][1].op in _CFG_TERMINAL for block in blocks):
        return None
    return blocks


class CFGValueIRCompiler(ValueIRCompiler):
    """Lower a forward-only fully typed scalar CFG into merge-aware Value IR."""

    def __init__(self, proto: Proto):
        raw = _build_blocks(proto)
        if raw is None:
            self._raw_blocks = None
            # A placeholder plan is never consumed when raw construction failed.
            super().__init__(proto, TypedIRCompiler(proto).compile(()))
            return
        blocks = tuple(block.instructions for block in raw)
        typed = TypedIRCompiler(proto).compile(blocks)
        super().__init__(proto, typed)
        self._raw_blocks = raw

    def _clear_expression_interns(self) -> None:
        # CSE is currently block-local. Reusing a value produced only in a
        # sibling branch would violate dominance, so only immutable argument /
        # literal / constant intern entries survive a block boundary.
        self._intern = {
            key: node_id
            for key, node_id in self._intern.items()
            if not key or key[0] is not ValueKind.EXPRESSION
        }

    def _initial_state(self) -> dict[int, int] | None:
        try:
            state = {
                register: self._literal(None)
                for register in range(self.proto.register_count)
            }
        except ValueError:
            return None
        for index in range(min(self.proto.param_count, self.proto.register_count)):
            if index >= len(self.proto.param_types):
                return None
            type_name = self.proto.param_types[index].name
            if type_name == "Any":
                return None
            state[index] = self._argument(index, type_name)
        return state

    def _merge_state(
        self,
        block_start: int,
        predecessor_indices: tuple[int, ...],
        predecessor_states: tuple[dict[int, int], ...],
    ) -> tuple[dict[int, int], tuple[int, ...]] | None:
        if not predecessor_states:
            return None
        merged: dict[int, int] = {}
        phis: list[int] = []
        for register in range(self.proto.register_count):
            incoming = tuple(state[register] for state in predecessor_states)
            if all(value == incoming[0] for value in incoming[1:]):
                merged[register] = incoming[0]
                continue
            types = tuple(self._nodes[value].type_name for value in incoming)
            if not all(type_name == types[0] for type_name in types[1:]):
                return None
            if not _safe_scalar_type(types[0]):
                return None
            node_id = self._new_node(
                ValueKind.EXPRESSION,
                types[0],
                op="phi",
                args=incoming,
                payload=predecessor_indices,
                def_pc=block_start,
            )
            merged[register] = node_id
            phis.append(node_id)
        return merged, tuple(phis)

    def _compile_pure_instruction(
        self, pc: int, ins: Ins, state: dict[int, int]
    ) -> bool:
        op = ins.op
        if op is Op.LOADK:
            try:
                state[ins.a] = self._constant(ins.b)
            except (IndexError, ValueError):
                return False
            return True
        if op in (Op.MOVE, Op.LOCAL):
            state[ins.a] = state[ins.b]
            return True
        if op in _INT_BINOPS:
            left = state[ins.b]
            right = state[ins.c]
            if self._nodes[left].type_name != "integer" or self._nodes[right].type_name != "integer":
                return False
            state[ins.a] = self._binary(pc, _INT_BINOPS[op], left, right, "integer")
            return True
        if op in _FLOAT_BINOPS:
            left = state[ins.b]
            right = state[ins.c]
            numeric = {"integer", "float", "number"}
            if self._nodes[left].type_name not in numeric or self._nodes[right].type_name not in numeric:
                return False
            state[ins.a] = self._binary(pc, _FLOAT_BINOPS[op], left, right, "float")
            return True
        if op in _BOOL_UNARY:
            source = state[ins.b]
            state[ins.a] = self._unary(pc, _BOOL_UNARY[op], source, "boolean")
            return True
        if op in _COMPARE_OPS:
            left = state[ins.b]
            right = state[ins.c]
            left_type = self._nodes[left].type_name
            right_type = self._nodes[right].type_name
            if not (_safe_scalar_type(left_type) and _safe_scalar_type(right_type)):
                return False
            if op in (Op.LT, Op.LE):
                numeric = {"integer", "float", "number"}
                if not (
                    (left_type in numeric and right_type in numeric)
                    or (left_type == right_type == "string")
                ):
                    return False
            state[ins.a] = self._binary(pc, _COMPARE_OPS[op], left, right, "boolean")
            return True
        return False

    def compile(self) -> CFGValueIRPlan | None:
        raw_blocks = self._raw_blocks
        if raw_blocks is None or len(raw_blocks) < 2:
            return None
        initial = self._initial_state()
        if initial is None:
            return None

        index_by_start = {block.start: index for index, block in enumerate(raw_blocks)}
        predecessor_lists: list[list[int]] = [[] for _ in raw_blocks]
        successor_indices: list[tuple[int, ...]] = []
        for index, block in enumerate(raw_blocks):
            successors = tuple(index_by_start[start] for start in block.successor_starts)
            successor_indices.append(successors)
            for successor in successors:
                predecessor_lists[successor].append(index)

        entry_states: list[dict[int, int] | None] = [None] * len(raw_blocks)
        exit_states: list[dict[int, int] | None] = [None] * len(raw_blocks)
        compiled_blocks: list[CFGValueIRBlock] = []

        for index, block in enumerate(raw_blocks):
            predecessors = tuple(predecessor_lists[index])
            if index == 0:
                if predecessors:
                    return None
                state = dict(initial)
                phi_nodes: tuple[int, ...] = ()
            else:
                predecessor_states = tuple(
                    exit_states[pred] for pred in predecessors if exit_states[pred] is not None
                )
                if len(predecessor_states) != len(predecessors) or not predecessors:
                    return None
                merged = self._merge_state(
                    block.start,
                    predecessors,
                    tuple(state for state in predecessor_states if state is not None),
                )
                if merged is None:
                    return None
                state, phi_nodes = merged

            entry_states[index] = dict(state)
            self._clear_expression_interns()
            condition_value: int | None = None
            return_values: tuple[int, ...] = ()

            for pc, ins in block.instructions:
                if ins.op in _CFG_CONTROL:
                    if (pc, ins) != block.instructions[-1]:
                        return None
                    if ins.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
                        condition_value = state[ins.b]
                    continue
                if ins.op is Op.RETURN:
                    if (pc, ins) != block.instructions[-1]:
                        return None
                    return_values = tuple(
                        state[register]
                        for register in range(ins.a, ins.a + max(0, ins.b))
                    )
                    continue
                if ins.op is Op.HALT:
                    if (pc, ins) != block.instructions[-1]:
                        return None
                    return_values = ()
                    continue
                if not self._compile_pure_instruction(pc, ins, state):
                    return None

            exit_states[index] = dict(state)
            compiled_blocks.append(
                CFGValueIRBlock(
                    index=index,
                    start=block.start,
                    end=block.end,
                    predecessors=predecessors,
                    successors=successor_indices[index],
                    instructions=block.instructions,
                    entry_values=tuple(sorted(entry_states[index].items())),
                    exit_values=tuple(sorted(state.items())),
                    phi_nodes=phi_nodes,
                    condition_value=condition_value,
                    return_values=return_values,
                )
            )

        if not any(block.phi_nodes for block in compiled_blocks):
            # Branch-only functions without any merge value are already handled
            # efficiently by the older structured/typed function tiers. 0.18 is
            # specifically validating SSA joins first.
            return None

        return CFGValueIRPlan(
            typed_plan=self.typed_plan,
            nodes=tuple(self._nodes),
            blocks=tuple(compiled_blocks),
            folded_pcs=tuple(sorted(self._folded_pcs)),
            cse_pcs=tuple(sorted(self._cse_pcs)),
        )
