from __future__ import annotations

from dataclasses import dataclass

from .bytecode import Proto
from .cfg_value_ir import CFGNaturalLoop, CFGValueIRPlan
from .value_ir import ValueKind


_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63
_TWO64 = 1 << 64


def _i64(value: int) -> int:
    value &= _MASK64
    return value - _TWO64 if value & _SIGN64 else value


@dataclass(frozen=True, slots=True)
class CFGInductionVariable:
    """Canonical loop-carried integer induction variable.

    ``phi_node`` is the loop-header value, ``update_node`` is the latch value,
    and ``step`` is the signed modulo-2**64 increment.  ``direct_update`` means
    the update node has no consumer other than the header phi and can therefore
    be materialized directly into the phi variable by a backend.
    """

    header: int
    phi_node: int
    init_node: int
    update_node: int
    update_pc: int
    step: int
    direct_update: bool
    compare_node: int | None = None
    limit_node: int | None = None
    compare_op: str | None = None


@dataclass(frozen=True, slots=True)
class CFGLoopOptimization:
    header: int
    invariant_nodes: tuple[int, ...]
    induction_variables: tuple[CFGInductionVariable, ...]

    @property
    def direct_update_nodes(self) -> frozenset[int]:
        return frozenset(
            item.update_node for item in self.induction_variables if item.direct_update
        )


@dataclass(frozen=True, slots=True)
class CFGLoopOptimizationPlan:
    loops: tuple[CFGLoopOptimization, ...]

    def for_header(self, header: int) -> CFGLoopOptimization | None:
        for item in self.loops:
            if item.header == header:
                return item
        return None

    @property
    def hoisted_nodes(self) -> frozenset[int]:
        return frozenset(node for loop in self.loops for node in loop.invariant_nodes)


class CFGLoopOptimizer:
    """Backend-neutral LICM and induction analysis for ``CFGValueIRPlan``.

    The pass never changes Lua instruction accounting.  It only proves which
    immutable Value-IR computations may be materialized outside a natural loop
    and which cyclic integer phis have the canonical ``phi +/- constant`` form.
    Python AST is one consumer of these facts; a later native backend can use the
    same proof unchanged.
    """

    def __init__(self, proto: Proto, plan: CFGValueIRPlan):
        self.proto = proto
        self.plan = plan
        self._pc_block = {
            pc: block.index
            for block in plan.blocks
            for pc, _ins in block.instructions
        }
        self._def_block = {
            node.id: self._pc_block[node.def_pc]
            for node in plan.nodes
            if node.kind is ValueKind.EXPRESSION
            and node.def_pc >= 0
            and node.def_pc in self._pc_block
        }
        consumers: dict[int, list[int]] = {}
        for node in plan.nodes:
            for arg in node.args:
                consumers.setdefault(arg, []).append(node.id)
        self._consumers = {key: tuple(value) for key, value in consumers.items()}

    def _defined_before_loop(self, node_id: int, loop: CFGNaturalLoop) -> bool:
        node = self.plan.node(node_id)
        if node.kind is not ValueKind.EXPRESSION:
            return True
        block = self._def_block.get(node_id)
        if block is None:
            return False
        if block in loop.blocks:
            return False
        return self.plan.dominates(block, loop.header)

    def _invariant_nodes(self, loop: CFGNaturalLoop) -> tuple[int, ...]:
        loop_blocks = set(loop.blocks)
        invariant: set[int] = set()
        changed = True
        while changed:
            changed = False
            for node in self.plan.nodes:
                if (
                    node.id in invariant
                    or node.kind is not ValueKind.EXPRESSION
                    or node.op == "phi"
                ):
                    continue
                block = self._def_block.get(node.id)
                if block not in loop_blocks:
                    continue
                if all(
                    arg in invariant or self._defined_before_loop(arg, loop)
                    for arg in node.args
                ):
                    invariant.add(node.id)
                    changed = True
        # Value nodes are created after their operands.  Keeping node-id order
        # therefore gives every backend a dependency-safe materialization order.
        return tuple(sorted(invariant))

    def _known_int(self, node_id: int) -> int | None:
        node = self.plan.node(node_id)
        if node.kind is ValueKind.LITERAL and type(node.payload) is int:
            return node.payload
        if node.kind is ValueKind.CONSTANT:
            value = self.proto.constants[int(node.payload)]
            return value if type(value) is int else None
        return None

    def _induction_for_phi(
        self,
        loop: CFGNaturalLoop,
        phi_id: int,
        invariant_nodes: frozenset[int],
    ) -> CFGInductionVariable | None:
        phi = self.plan.node(phi_id)
        if phi.op != "phi" or phi.type_name != "integer" or len(phi.args) != 2:
            return None
        init_node, update_node = phi.args
        update = self.plan.node(update_node)
        if update.op not in ("add_i", "sub_i") or len(update.args) != 2:
            return None
        if self._def_block.get(update_node) not in set(loop.blocks):
            return None

        constant_node: int | None = None
        raw_step: int | None = None
        if update.op == "add_i":
            if update.args[0] == phi_id:
                constant_node = update.args[1]
            elif update.args[1] == phi_id:
                constant_node = update.args[0]
        elif update.args[0] == phi_id:
            constant_node = update.args[1]
        if constant_node is None:
            return None
        constant = self._known_int(constant_node)
        if constant is None:
            return None
        raw_step = constant if update.op == "add_i" else -constant
        step = _i64(raw_step)

        # The backedge expression may disappear only when the phi is its sole
        # consumer.  Otherwise other bytecode-visible values still need it.
        direct = self._consumers.get(update_node, ()) == (phi_id,)

        compare_node: int | None = None
        limit_node: int | None = None
        compare_op: str | None = None
        header = self.plan.blocks[loop.header]
        condition_id = header.condition_value
        if condition_id is not None:
            condition = self.plan.node(condition_id)
            if condition.op in ("lt", "le") and len(condition.args) == 2:
                if condition.args[0] == phi_id:
                    other = condition.args[1]
                elif condition.args[1] == phi_id:
                    other = condition.args[0]
                else:
                    other = None
                if other is not None and (
                    other in invariant_nodes or self._defined_before_loop(other, loop)
                ):
                    compare_node = condition_id
                    limit_node = other
                    compare_op = condition.op

        return CFGInductionVariable(
            header=loop.header,
            phi_node=phi_id,
            init_node=init_node,
            update_node=update_node,
            update_pc=update.def_pc,
            step=step,
            direct_update=direct,
            compare_node=compare_node,
            limit_node=limit_node,
            compare_op=compare_op,
        )

    def analyze(self) -> CFGLoopOptimizationPlan:
        optimized: list[CFGLoopOptimization] = []
        for loop in self.plan.natural_loops:
            invariants = self._invariant_nodes(loop)
            invariant_set = frozenset(invariants)
            inductions = tuple(
                induction
                for phi_id in self.plan.blocks[loop.header].phi_nodes
                if (
                    induction := self._induction_for_phi(
                        loop, phi_id, invariant_set
                    )
                )
                is not None
            )
            optimized.append(
                CFGLoopOptimization(
                    header=loop.header,
                    invariant_nodes=invariants,
                    induction_variables=inductions,
                )
            )
        return CFGLoopOptimizationPlan(tuple(optimized))
