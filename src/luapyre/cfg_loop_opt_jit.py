from __future__ import annotations

import ast

from .bytecode import Proto
from .cfg_value_ir import CFGValueIRBlock, CFGValueIRPlan
from .function_jit import CompiledAstFunction
from .loop_opt import CFGInductionVariable, CFGLoopOptimizer


class CFGLoopOptimizationJITMixin:
    """Consume backend-neutral 0.20 loop facts in the Python-AST backend.

    This mixin intentionally owns no semantic analysis.  ``CFGLoopOptimizer``
    proves LICM and induction facts; this backend only chooses a concrete Python
    representation for those facts.  Unsupported loop shapes fall through to
    the 0.19 CFG backend unchanged.
    """

    def _emit_block_expressions_skipping(
        self,
        plan: CFGValueIRPlan,
        block: CFGValueIRBlock,
        nodes_by_pc: dict[int, list[int]],
        indent: str,
        skip_nodes: frozenset[int],
    ) -> list[str]:
        out: list[str] = []
        for pc, _ins in block.instructions:
            for node_id in nodes_by_pc.get(pc, ()):
                if node_id in skip_nodes:
                    continue
                out.extend(self._emit_value_expression(plan, node_id, indent))
        # Lua fuel follows original bytecode, not the number of Python
        # assignments left after LICM/strength reduction.
        out.append(f"{indent}meter[0] += {block.cost}")
        return out

    @staticmethod
    def _direct_inductions(
        inductions: tuple[CFGInductionVariable, ...],
    ) -> dict[int, CFGInductionVariable]:
        return {
            item.phi_node: item
            for item in inductions
            if item.direct_update
        }

    def _emit_optimized_loop_phi(
        self,
        plan: CFGValueIRPlan,
        join: CFGValueIRBlock,
        predecessor: int,
        latch: int,
        inductions: tuple[CFGInductionVariable, ...],
        indent: str,
    ) -> list[str] | None:
        direct = self._direct_inductions(inductions) if predecessor == latch else {}
        out: list[str] = []
        for phi_id in join.phi_nodes:
            induction = direct.get(phi_id)
            if induction is not None:
                wide = f"_wide_ind_{phi_id}"
                out.extend(
                    [
                        f"{indent}{wide} = (_v{phi_id} + ({induction.step})) & _MASK64",
                        f"{indent}_v{phi_id} = {wide} - _TWO64 if {wide} & _SIGN64 else {wide}",
                    ]
                )
                continue
            phi = plan.node(phi_id)
            predecessors = tuple(int(value) for value in phi.payload)
            try:
                source_index = predecessors.index(predecessor)
            except ValueError:
                return None
            out.append(
                f"{indent}_v{phi_id} = {self._value_expr(plan, phi.args[source_index])}"
            )
        return out

    def _compile_structured_natural_loop(
        self, proto: Proto, plan: CFGValueIRPlan
    ) -> CompiledAstFunction | None:
        if len(plan.natural_loops) != 1:
            return super()._compile_structured_natural_loop(proto, plan)
        loop = plan.natural_loops[0]
        optimization = CFGLoopOptimizer(proto, plan).analyze().for_header(loop.header)
        if optimization is None:
            return super()._compile_structured_natural_loop(proto, plan)

        header_index = loop.header
        header = plan.blocks[header_index]
        if header.condition_value is None or len(header.successors) != 2:
            return super()._compile_structured_natural_loop(proto, plan)
        if len(loop.preheaders) != 1 or len(loop.exits) != 1:
            return super()._compile_structured_natural_loop(proto, plan)
        preheader_index = loop.preheaders[0]
        exit_source, exit_index = loop.exits[0]
        if preheader_index != 0 or exit_source != header_index:
            return super()._compile_structured_natural_loop(proto, plan)
        preheader = plan.blocks[preheader_index]
        exit_block = plan.blocks[exit_index]
        if preheader.successors != (header_index,):
            return super()._compile_structured_natural_loop(proto, plan)
        if self._emit_return(plan, exit_block, "") is None:
            return super()._compile_structured_natural_loop(proto, plan)

        outside = set(range(len(plan.blocks))) - set(loop.blocks)
        if outside != {preheader_index, exit_index}:
            return super()._compile_structured_natural_loop(proto, plan)

        condition = self._value_expr(plan, header.condition_value)
        jump_test = self._condition_test(header, condition)
        if jump_test is None:
            return super()._compile_structured_natural_loop(proto, plan)
        target, fallthrough = header.successors
        if target == exit_index and fallthrough in loop.blocks:
            exit_test = jump_test
            body_start = fallthrough
        elif fallthrough == exit_index and target in loop.blocks:
            exit_test = f"not ({jump_test})"
            body_start = target
        else:
            return super()._compile_structured_natural_loop(proto, plan)

        loop_blocks = set(loop.blocks)
        chain: list[int] = []
        seen: set[int] = set()
        current = body_start
        while current != header_index:
            if current in seen or current not in loop_blocks:
                return super()._compile_structured_natural_loop(proto, plan)
            seen.add(current)
            block = plan.blocks[current]
            if block.condition_value is not None or block.phi_nodes:
                return super()._compile_structured_natural_loop(proto, plan)
            if self._emit_return(plan, block, "") is not None:
                return super()._compile_structured_natural_loop(proto, plan)
            if len(block.successors) != 1:
                return super()._compile_structured_natural_loop(proto, plan)
            chain.append(current)
            current = block.successors[0]
        if not chain or chain[-1] != loop.latch:
            return super()._compile_structured_natural_loop(proto, plan)
        if set(chain) != loop_blocks - {header_index}:
            return super()._compile_structured_natural_loop(proto, plan)

        nodes_by_pc = self._nodes_by_pc(plan)
        invariant_nodes = frozenset(optimization.invariant_nodes)
        direct_update_nodes = optimization.direct_update_nodes
        skip_nodes = invariant_nodes | direct_update_nodes
        namespace = self._base_namespace(plan)
        lines = [
            "def _jit_cfg_value_ir_loop_opt(vm, frames, frame, budget, meter):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
        ]
        for index in range(proto.param_count):
            lines.append(f"    _arg{index} = regs[{index}]")

        # Execute/charge the preheader exactly as before.  Proven-pure loop
        # invariants are then materialized once.  Their original bytecode slots
        # remain in block.cost and are therefore still charged on every dynamic
        # execution of the source block.
        lines.extend(self._emit_block_preflight(plan, preheader, "    "))
        lines.extend(self._emit_block_expressions(plan, preheader, nodes_by_pc, "    "))
        for node_id in optimization.invariant_nodes:
            lines.extend(self._emit_value_expression(plan, node_id, "    "))

        phi = self._emit_optimized_loop_phi(
            plan,
            header,
            preheader_index,
            loop.latch,
            optimization.induction_variables,
            "    ",
        )
        if phi is None:
            return super()._compile_structured_natural_loop(proto, plan)
        lines.extend(phi)
        lines.append("    while True:")
        lines.extend(self._emit_block_preflight(plan, header, "        "))
        lines.extend(
            self._emit_block_expressions_skipping(
                plan, header, nodes_by_pc, "        ", skip_nodes
            )
        )
        lines.append(f"        if {exit_test}:")
        lines.append("            break")
        for block_index in chain:
            block = plan.blocks[block_index]
            lines.extend(self._emit_block_preflight(plan, block, "        "))
            lines.extend(
                self._emit_block_expressions_skipping(
                    plan, block, nodes_by_pc, "        ", skip_nodes
                )
            )
        phi = self._emit_optimized_loop_phi(
            plan,
            header,
            loop.latch,
            loop.latch,
            optimization.induction_variables,
            "        ",
        )
        if phi is None:
            return super()._compile_structured_natural_loop(proto, plan)
        lines.extend(phi)

        lines.extend(self._emit_block_preflight(plan, exit_block, "    "))
        lines.extend(self._emit_block_expressions(plan, exit_block, nodes_by_pc, "    "))
        returned = self._emit_return(plan, exit_block, "    ")
        if returned is None:
            return super()._compile_structured_natural_loop(proto, plan)
        lines.extend(returned)

        tree = ast.parse("\n".join(lines))
        ast.fix_missing_locations(tree)
        # Keep the 0.19 code filename stable for architecture/tests/tooling while
        # the Python function name identifies the optimized 0.20 lowering.
        exec(compile(tree, "<luapyre-cfg-value-ir-loop>", "exec"), namespace)
        return CompiledAstFunction(proto, namespace["_jit_cfg_value_ir_loop_opt"])
