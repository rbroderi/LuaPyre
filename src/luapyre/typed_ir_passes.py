from __future__ import annotations

from dataclasses import replace

from .bytecode import Op
from .typed_ir import IRValueKind, TypedIRPlan, _cfg_targets, _writes


_REMATERIALIZABLE_DEFS = frozenset({Op.LOADK, Op.GETUPVAL})


def _physical_reads(plan: TypedIRPlan, pc: int) -> set[int]:
    """Registers the current backend must physically materialize at ``pc``.

    IR constants and direct-upvalue values are rematerializable and therefore do
    not make their temporary Lua register live. The Python backend can recreate
    those values exactly at a deopt point from ``TypedIRPlan.state_before``.
    """

    item = plan.instruction(pc)
    if item is None:
        return set()
    return {
        reg
        for reg, value in item.sources
        if value.kind is IRValueKind.REGISTER
    }


def eliminate_rematerializable_dead_defs(
    plan: TypedIRPlan,
    blocks: tuple[tuple[tuple[int, object], ...], ...],
) -> TypedIRPlan:
    """Run backward CFG liveness for exactly rematerializable definitions.

    The pass is intentionally narrower than generic DSE. Only ``LOADK`` and
    ``GETUPVAL`` are removed because their exact values already exist in the IR
    deoptimization state. Other operations may update cells, allocate, raise, or
    need expression/value-number IR before their result can be reconstructed.

    ``blocks`` must describe a closed CFG, such as a whole compiled function.
    Partial loop regions need explicit live-out information before this pass can
    safely be applied to them.
    """

    if not blocks:
        return plan

    starts = {block[0][0]: index for index, block in enumerate(blocks) if block}
    successors: list[set[int]] = [set() for _ in blocks]
    for index, block in enumerate(blocks):
        if not block:
            continue
        pc, terminal = block[-1]
        for target in _cfg_targets(terminal, pc + 1):
            successor = starts.get(target)
            if successor is not None:
                successors[index].add(successor)

    live_in: list[set[int]] = [set() for _ in blocks]
    live_out: list[set[int]] = [set() for _ in blocks]

    changed = True
    while changed:
        changed = False
        for index in range(len(blocks) - 1, -1, -1):
            out: set[int] = set()
            for successor in successors[index]:
                out.update(live_in[successor])

            live = set(out)
            for pc, ins in reversed(blocks[index]):
                live.difference_update(_writes(ins))
                live.update(_physical_reads(plan, pc))

            if out != live_out[index] or live != live_in[index]:
                live_out[index] = out
                live_in[index] = live
                changed = True

    dead_pcs: set[int] = set()
    for index, block in enumerate(blocks):
        live = set(live_out[index])
        for pc, ins in reversed(block):
            writes = _writes(ins)
            if (
                ins.op in _REMATERIALIZABLE_DEFS
                and len(writes) == 1
                and writes[0] not in live
            ):
                dead_pcs.add(pc)
            live.difference_update(writes)
            live.update(_physical_reads(plan, pc))

    instructions = tuple(
        replace(
            item,
            dead_definition=(
                item.pc in dead_pcs
                if item.ins.op in _REMATERIALIZABLE_DEFS
                else item.dead_definition
            ),
        )
        for item in plan.instructions
    )
    return replace(plan, instructions=instructions)
