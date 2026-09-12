from __future__ import annotations

from dataclasses import dataclass

from .bytecode import Ins, Op, Proto
from .typed_ir import TypedIRCompiler, TypedIRPlan, _reads, _writes
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
class CFGNaturalLoop:
    header: int
    latch: int
    blocks: tuple[int, ...]
    preheaders: tuple[int, ...]
    exits: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class CFGValueIRBlock:
    index: int
    start: int
    end: int
    predecessors: tuple[int, ...]
    successors: tuple[int, ...]
    instructions: tuple[tuple[int, Ins], ...]
    live_in: tuple[int, ...]
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
    """Dominance-aware SSA-like scalar graph for reducible typed CFGs.

    0.19 extends the 0.18 forward-DAG representation with explicit dominators,
    natural backedges, loop-carried phi values, and dominance-safe expression
    reuse. Only bytecode-live inputs participate in merge or side-exit state.
    Reducibility remains a hard correctness boundary: cycles that are not natural
    loops fail closed before code generation.
    """

    typed_plan: TypedIRPlan
    nodes: tuple[ValueNode, ...]
    blocks: tuple[CFGValueIRBlock, ...]
    dominators: tuple[tuple[int, ...], ...]
    backedges: tuple[tuple[int, int], ...]
    natural_loops: tuple[CFGNaturalLoop, ...]
    folded_pcs: tuple[int, ...]
    cse_pcs: tuple[int, ...]
    cross_block_cse_pcs: tuple[int, ...]

    def node(self, node_id: int) -> ValueNode:
        return self.nodes[node_id]

    def dominates(self, dominator: int, block: int) -> bool:
        return dominator in self.dominators[block]

    @property
    def phi_nodes(self) -> tuple[ValueNode, ...]:
        return tuple(node for node in self.nodes if node.op == "phi")

    @property
    def has_cycles(self) -> bool:
        return bool(self.backedges)


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
    if any(
        target not in by_start
        for block in raw
        for target in block.successor_starts
    ):
        return None

    reachable: set[int] = set()
    stack = [0]
    while stack:
        start = stack.pop()
        if start in reachable:
            continue
        reachable.add(start)
        stack.extend(raw[by_start[start]].successor_starts)

    blocks = tuple(block for block in raw if block.start in reachable)
    if not any(block.instructions[-1][1].op in _CFG_TERMINAL for block in blocks):
        return None
    return blocks


def _block_liveness(
    blocks: tuple[_RawBlock, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    index_by_start = {block.start: index for index, block in enumerate(blocks)}
    successors = tuple(
        tuple(index_by_start[start] for start in block.successor_starts)
        for block in blocks
    )
    uses: list[set[int]] = []
    defs: list[set[int]] = []
    for block in blocks:
        block_uses: set[int] = set()
        block_defs: set[int] = set()
        for _pc, ins in block.instructions:
            for register in _reads(ins):
                if register not in block_defs:
                    block_uses.add(register)
            block_defs.update(_writes(ins))
        uses.append(block_uses)
        defs.append(block_defs)

    live_in = [set() for _ in blocks]
    live_out = [set() for _ in blocks]
    changed = True
    while changed:
        changed = False
        for index in range(len(blocks) - 1, -1, -1):
            new_out: set[int] = set()
            for successor in successors[index]:
                new_out.update(live_in[successor])
            new_in = uses[index] | (new_out - defs[index])
            if new_out != live_out[index] or new_in != live_in[index]:
                live_out[index] = new_out
                live_in[index] = new_in
                changed = True
    return successors, tuple(tuple(sorted(registers)) for registers in live_in)


def _predecessors(successors: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    incoming: list[list[int]] = [[] for _ in successors]
    for source, targets in enumerate(successors):
        for target in targets:
            incoming[target].append(source)
    return tuple(tuple(items) for items in incoming)


def _dominators(
    successors: tuple[tuple[int, ...], ...],
    predecessors: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    all_blocks = set(range(len(successors)))
    dom = [set(all_blocks) for _ in successors]
    dom[0] = {0}
    changed = True
    while changed:
        changed = False
        for block in range(1, len(successors)):
            preds = predecessors[block]
            if not preds:
                new = {block}
            else:
                common = set(dom[preds[0]])
                for pred in preds[1:]:
                    common.intersection_update(dom[pred])
                new = common | {block}
            if new != dom[block]:
                dom[block] = new
                changed = True
    return tuple(tuple(sorted(items)) for items in dom)


def _find_backedges(
    successors: tuple[tuple[int, ...], ...],
    dominators: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for source, targets in enumerate(successors):
        source_dominators = set(dominators[source])
        for target in targets:
            if target in source_dominators:
                result.append((source, target))
    return tuple(result)


def _topological_without_backedges(
    successors: tuple[tuple[int, ...], ...],
    backedges: tuple[tuple[int, int], ...],
) -> tuple[int, ...] | None:
    removed = set(backedges)
    indegree = [0] * len(successors)
    for source, targets in enumerate(successors):
        for target in targets:
            if (source, target) not in removed:
                indegree[target] += 1
    ready = [index for index, count in enumerate(indegree) if count == 0]
    ready.sort(reverse=True)
    order: list[int] = []
    while ready:
        block = ready.pop()
        order.append(block)
        for target in successors[block]:
            if (block, target) in removed:
                continue
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort(reverse=True)
    return tuple(order) if len(order) == len(successors) else None


def _natural_loop_blocks(
    header: int,
    latch: int,
    predecessors: tuple[tuple[int, ...], ...],
) -> frozenset[int]:
    loop = {header, latch}
    stack = [] if latch == header else [latch]
    while stack:
        block = stack.pop()
        for pred in predecessors[block]:
            if pred in loop:
                continue
            loop.add(pred)
            if pred != header:
                stack.append(pred)
    return frozenset(loop)


def _build_natural_loops(
    successors: tuple[tuple[int, ...], ...],
    predecessors: tuple[tuple[int, ...], ...],
    backedges: tuple[tuple[int, int], ...],
) -> tuple[CFGNaturalLoop, ...] | None:
    by_header: dict[int, list[int]] = {}
    for latch, header in backedges:
        by_header.setdefault(header, []).append(latch)

    result: list[CFGNaturalLoop] = []
    for header, latches in sorted(by_header.items()):
        # Multi-latch loops need phi inputs from more than one cyclic edge. Keep
        # the first cyclic tranche exact and fail closed until that representation
        # is explicit rather than silently guessing an incoming order.
        if len(latches) != 1:
            return None
        latch = latches[0]
        blocks = _natural_loop_blocks(header, latch, predecessors)
        preheaders = tuple(pred for pred in predecessors[header] if pred not in blocks)
        if len(preheaders) != 1:
            return None
        exits = tuple(
            (source, target)
            for source in sorted(blocks)
            for target in successors[source]
            if target not in blocks
        )
        result.append(
            CFGNaturalLoop(
                header=header,
                latch=latch,
                blocks=tuple(sorted(blocks)),
                preheaders=preheaders,
                exits=exits,
            )
        )
    return tuple(result)


class CFGValueIRCompiler(ValueIRCompiler):
    """Lower reducible fully typed scalar CFGs into dominance-aware Value IR."""

    def __init__(self, proto: Proto):
        raw = _build_blocks(proto)
        if raw is None:
            self._raw_blocks = None
            super().__init__(proto, TypedIRCompiler(proto).compile(()))
            return
        typed = TypedIRCompiler(proto).compile(tuple(block.instructions for block in raw))
        super().__init__(proto, typed)
        self._raw_blocks = raw

    def _prepare_expression_interns(
        self,
        block_index: int,
        dominators: tuple[tuple[int, ...], ...],
        node_def_blocks: dict[int, int],
    ) -> None:
        dominating = set(dominators[block_index])
        self._intern = {
            key: node_id
            for key, node_id in self._intern.items()
            if (
                not key
                or key[0] is not ValueKind.EXPRESSION
                or node_def_blocks.get(node_id) in dominating
            )
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
        live_registers: frozenset[int],
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
            if register not in live_registers:
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

    @staticmethod
    def _loop_writes(
        raw_blocks: tuple[_RawBlock, ...], loop: CFGNaturalLoop
    ) -> frozenset[int]:
        writes: set[int] = set()
        for block_index in loop.blocks:
            for _pc, ins in raw_blocks[block_index].instructions:
                writes.update(_writes(ins))
        return frozenset(writes)

    def compile(self) -> CFGValueIRPlan | None:
        raw_blocks = self._raw_blocks
        if raw_blocks is None or len(raw_blocks) < 2:
            return None
        initial = self._initial_state()
        if initial is None:
            return None

        successor_indices, live_in = _block_liveness(raw_blocks)
        predecessor_indices = _predecessors(successor_indices)
        dominators = _dominators(successor_indices, predecessor_indices)
        backedges = _find_backedges(successor_indices, dominators)
        order = _topological_without_backedges(successor_indices, backedges)
        if order is None:
            return None
        natural_loops = _build_natural_loops(
            successor_indices, predecessor_indices, backedges
        )
        if natural_loops is None:
            return None
        loop_by_header = {loop.header: loop for loop in natural_loops}
        loop_writes = {
            loop.header: self._loop_writes(raw_blocks, loop)
            for loop in natural_loops
        }

        entry_states: list[dict[int, int] | None] = [None] * len(raw_blocks)
        exit_states: list[dict[int, int] | None] = [None] * len(raw_blocks)
        compiled_blocks: list[CFGValueIRBlock | None] = [None] * len(raw_blocks)
        node_def_blocks: dict[int, int] = {}
        cross_block_cse_pcs: set[int] = set()
        pending_loop_phis: list[tuple[int, int, int, int, int, int]] = []

        for index in order:
            block = raw_blocks[index]
            predecessors = predecessor_indices[index]
            if index == 0:
                if predecessors:
                    return None
                state = dict(initial)
                phi_nodes: tuple[int, ...] = ()
            elif index in loop_by_header:
                loop = loop_by_header[index]
                preheader = loop.preheaders[0]
                pre_state = exit_states[preheader]
                if pre_state is None:
                    return None
                state = dict(pre_state)
                phis: list[int] = []
                written = loop_writes[index]
                for register in live_in[index]:
                    if register not in written:
                        continue
                    pre_value = state[register]
                    type_name = self._nodes[pre_value].type_name
                    if not _safe_scalar_type(type_name):
                        return None
                    phi_id = self._new_node(
                        ValueKind.EXPRESSION,
                        type_name,
                        op="phi",
                        args=(pre_value,),
                        payload=(preheader, loop.latch),
                        def_pc=block.start,
                    )
                    node_def_blocks[phi_id] = index
                    state[register] = phi_id
                    phis.append(phi_id)
                    pending_loop_phis.append(
                        (phi_id, index, preheader, loop.latch, pre_value, register)
                    )
                phi_nodes = tuple(phis)
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
                    frozenset(live_in[index]),
                )
                if merged is None:
                    return None
                state, phi_nodes = merged
                for phi_id in phi_nodes:
                    node_def_blocks[phi_id] = index

            entry_states[index] = dict(state)
            self._prepare_expression_interns(index, dominators, node_def_blocks)
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
                    continue

                before_nodes = len(self._nodes)
                before_cse = pc in self._cse_pcs
                if not self._compile_pure_instruction(pc, ins, state):
                    return None
                for node_id in range(before_nodes, len(self._nodes)):
                    if self._nodes[node_id].kind is ValueKind.EXPRESSION:
                        node_def_blocks[node_id] = index
                if not before_cse and pc in self._cse_pcs:
                    for register in _writes(ins):
                        value_id = state[register]
                        definition_block = node_def_blocks.get(value_id)
                        if definition_block is not None and definition_block != index:
                            cross_block_cse_pcs.add(pc)

            exit_states[index] = dict(state)
            compiled_blocks[index] = CFGValueIRBlock(
                index=index,
                start=block.start,
                end=block.end,
                predecessors=predecessors,
                successors=successor_indices[index],
                instructions=block.instructions,
                live_in=live_in[index],
                entry_values=tuple(sorted(entry_states[index].items())),
                exit_values=tuple(sorted(state.items())),
                phi_nodes=phi_nodes,
                condition_value=condition_value,
                return_values=return_values,
            )

        # Loop headers are compiled before their latches. Patch the provisional
        # phi inputs only after every backedge state exists. The resulting graph
        # may be cyclic (phi -> expression -> phi), which is the intended SSA
        # representation of a loop-carried value.
        for phi_id, header, preheader, latch, pre_value, register in pending_loop_phis:
            latch_state = exit_states[latch]
            if latch_state is None:
                return None
            back_value = latch_state[register]
            node = self._nodes[phi_id]
            if self._nodes[back_value].type_name != node.type_name:
                return None
            self._nodes[phi_id] = ValueNode(
                id=node.id,
                kind=node.kind,
                type_name=node.type_name,
                op=node.op,
                args=(pre_value, back_value),
                payload=(preheader, latch),
                def_pc=node.def_pc,
                overflow_free=node.overflow_free,
            )

        blocks = tuple(block for block in compiled_blocks if block is not None)
        if len(blocks) != len(raw_blocks):
            return None
        if not (
            any(block.phi_nodes for block in blocks)
            or backedges
            or cross_block_cse_pcs
        ):
            return None

        return CFGValueIRPlan(
            typed_plan=self.typed_plan,
            nodes=tuple(self._nodes),
            blocks=blocks,
            dominators=dominators,
            backedges=backedges,
            natural_loops=natural_loops,
            folded_pcs=tuple(sorted(self._folded_pcs)),
            cse_pcs=tuple(sorted(self._cse_pcs)),
            cross_block_cse_pcs=tuple(sorted(cross_block_cse_pcs)),
        )
