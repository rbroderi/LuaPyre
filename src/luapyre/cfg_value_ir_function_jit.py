from __future__ import annotations

import ast

from .bytecode import Op, Proto
from .cfg_value_ir import CFGValueIRBlock, CFGValueIRCompiler, CFGValueIRPlan
from .function_jit import CompiledAstFunction, _FUNC_RETURN, _FUNC_SUSPEND
from .value_ir import ValueKind


_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63
_TWO64 = 1 << 64


class CFGValueIRFunctionJITMixin:
    """0.18 acyclic CFG value backend with exact block side exits.

    Reducible four-block diamonds lower directly to Python ``if``/``else`` so
    the common typed branch shape pays no synthetic state-dispatch overhead.
    More general forward DAGs keep the exact state-machine backend.
    """

    @staticmethod
    def _value_expr(plan: CFGValueIRPlan, node_id: int) -> str:
        node = plan.node(node_id)
        if node.kind is ValueKind.ARGUMENT:
            return f"_arg{int(node.payload)}"
        if node.kind is ValueKind.CONSTANT:
            return f"consts[{int(node.payload)}]"
        if node.kind is ValueKind.LITERAL:
            return f"_lit_{node.id}"
        return f"_v{node.id}"

    def _emit_value_expression(
        self, plan: CFGValueIRPlan, node_id: int, indent: str
    ) -> list[str]:
        node = plan.node(node_id)
        args = [self._value_expr(plan, arg) for arg in node.args]
        dest = f"_v{node.id}"
        op = node.op
        out: list[str] = []
        if op == "phi":
            return out
        if op in ("add_i", "sub_i", "mul_i"):
            symbol = {"add_i": "+", "sub_i": "-", "mul_i": "*"}[op]
            tmp = f"_wide_{node.id}"
            out.extend(
                [
                    f"{indent}{tmp} = ({args[0]} {symbol} {args[1]}) & _MASK64",
                    f"{indent}{dest} = {tmp} - _TWO64 if {tmp} & _SIGN64 else {tmp}",
                ]
            )
        elif op in ("add_f", "sub_f", "mul_f"):
            symbol = {"add_f": "+", "sub_f": "-", "mul_f": "*"}[op]
            out.append(f"{indent}{dest} = float({args[0]} {symbol} {args[1]})")
        elif op == "not":
            out.append(f"{indent}{dest} = ({args[0]} is None or {args[0]} is False)")
        elif op == "tobool":
            out.append(f"{indent}{dest} = not ({args[0]} is None or {args[0]} is False)")
        elif op == "eq":
            left_type = plan.node(node.args[0]).type_name
            right_type = plan.node(node.args[1]).type_name
            if left_type == right_type or (
                left_type in {"integer", "float", "number"}
                and right_type in {"integer", "float", "number"}
            ):
                out.append(f"{indent}{dest} = {args[0]} == {args[1]}")
            else:
                out.append(f"{indent}{dest} = False")
        elif op in ("lt", "le"):
            symbol = "<" if op == "lt" else "<="
            out.append(f"{indent}{dest} = {args[0]} {symbol} {args[1]}")
        return out

    @staticmethod
    def _condition_test(block: CFGValueIRBlock, condition: str) -> str | None:
        terminal = block.instructions[-1][1]
        if terminal.op is Op.JMPIF:
            return f"not ({condition} is None or {condition} is False)"
        if terminal.op is Op.JMPIFNOT:
            return f"({condition} is None or {condition} is False)"
        if terminal.op is Op.JMPIFNIL:
            return f"{condition} is None"
        return None

    def _base_namespace(self, plan: CFGValueIRPlan) -> dict[str, object]:
        namespace: dict[str, object] = {
            "_FUNC_RETURN": _FUNC_RETURN,
            "_FUNC_SUSPEND": _FUNC_SUSPEND,
            "_MASK64": _MASK64,
            "_SIGN64": _SIGN64,
            "_TWO64": _TWO64,
        }
        for node in plan.nodes:
            if node.kind is ValueKind.LITERAL:
                namespace[f"_lit_{node.id}"] = node.payload
        return namespace

    @staticmethod
    def _nodes_by_pc(plan: CFGValueIRPlan) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {}
        for node in plan.nodes:
            if node.kind is ValueKind.EXPRESSION and node.op != "phi" and node.def_pc >= 0:
                result.setdefault(node.def_pc, []).append(node.id)
        return result

    def _emit_block_preflight(
        self,
        plan: CFGValueIRPlan,
        block: CFGValueIRBlock,
        indent: str,
    ) -> list[str]:
        out = [f"{indent}if budget - meter[0] < {block.cost}:"]
        entry = dict(block.entry_values)
        for register in block.live_in:
            out.append(
                f"{indent}    regs[{register}] = {self._value_expr(plan, entry[register])}"
            )
        out.extend(
            [
                f"{indent}    frame.pc = {block.start}",
                f"{indent}    return _FUNC_SUSPEND, None",
            ]
        )
        return out

    def _emit_block_expressions(
        self,
        plan: CFGValueIRPlan,
        block: CFGValueIRBlock,
        nodes_by_pc: dict[int, list[int]],
        indent: str,
    ) -> list[str]:
        out: list[str] = []
        for pc, _ins in block.instructions:
            for node_id in nodes_by_pc.get(pc, ()):
                out.extend(self._emit_value_expression(plan, node_id, indent))
        out.append(f"{indent}meter[0] += {block.cost}")
        return out

    def _emit_phi_for_predecessor(
        self,
        plan: CFGValueIRPlan,
        join: CFGValueIRBlock,
        predecessor: int,
        indent: str,
    ) -> list[str] | None:
        out: list[str] = []
        for phi_id in join.phi_nodes:
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

    def _emit_return(
        self, plan: CFGValueIRPlan, block: CFGValueIRBlock, indent: str
    ) -> list[str] | None:
        terminal = block.instructions[-1][1]
        if terminal.op is Op.HALT:
            return [f"{indent}return _FUNC_RETURN, ()"]
        if terminal.op is not Op.RETURN:
            return None
        values = [self._value_expr(plan, node_id) for node_id in block.return_values]
        if not values:
            result = "()"
        elif len(values) == 1:
            result = f"({values[0]},)"
        else:
            result = "(" + ", ".join(values) + ")"
        return [f"{indent}return _FUNC_RETURN, {result}"]

    def _compile_structured_diamond(
        self, proto: Proto, plan: CFGValueIRPlan
    ) -> CompiledAstFunction | None:
        if len(plan.blocks) != 4:
            return None
        entry = plan.blocks[0]
        if entry.condition_value is None or len(entry.successors) != 2:
            return None
        target_index, fallthrough_index = entry.successors
        if target_index == fallthrough_index:
            return None
        target = plan.blocks[target_index]
        fallthrough = plan.blocks[fallthrough_index]
        if len(target.successors) != 1 or len(fallthrough.successors) != 1:
            return None
        if target.successors[0] != fallthrough.successors[0]:
            return None
        join_index = target.successors[0]
        if join_index >= len(plan.blocks):
            return None
        join = plan.blocks[join_index]
        if set(join.predecessors) != {target_index, fallthrough_index}:
            return None
        if self._emit_return(plan, join, "") is None:
            return None

        condition = self._value_expr(plan, entry.condition_value)
        test = self._condition_test(entry, condition)
        if test is None:
            return None
        nodes_by_pc = self._nodes_by_pc(plan)
        namespace = self._base_namespace(plan)
        lines = [
            "def _jit_cfg_value_ir_diamond(vm, frames, frame, budget, meter):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
        ]
        for index in range(proto.param_count):
            lines.append(f"    _arg{index} = regs[{index}]")

        lines.extend(self._emit_block_preflight(plan, entry, "    "))
        lines.extend(self._emit_block_expressions(plan, entry, nodes_by_pc, "    "))
        lines.append(f"    if {test}:")
        lines.extend(self._emit_block_preflight(plan, target, "        "))
        lines.extend(self._emit_block_expressions(plan, target, nodes_by_pc, "        "))
        phi = self._emit_phi_for_predecessor(plan, join, target_index, "        ")
        if phi is None:
            return None
        lines.extend(phi)
        lines.append("    else:")
        lines.extend(self._emit_block_preflight(plan, fallthrough, "        "))
        lines.extend(
            self._emit_block_expressions(plan, fallthrough, nodes_by_pc, "        ")
        )
        phi = self._emit_phi_for_predecessor(plan, join, fallthrough_index, "        ")
        if phi is None:
            return None
        lines.extend(phi)

        lines.extend(self._emit_block_preflight(plan, join, "    "))
        lines.extend(self._emit_block_expressions(plan, join, nodes_by_pc, "    "))
        returned = self._emit_return(plan, join, "    ")
        if returned is None:
            return None
        lines.extend(returned)

        tree = ast.parse("\n".join(lines))
        ast.fix_missing_locations(tree)
        exec(compile(tree, "<luapyre-cfg-value-ir-diamond>", "exec"), namespace)
        return CompiledAstFunction(proto, namespace["_jit_cfg_value_ir_diamond"])

    def _compile_ast_function(self, proto: Proto) -> CompiledAstFunction | None:
        plan = CFGValueIRCompiler(proto).compile()
        if plan is None:
            return super()._compile_ast_function(proto)

        structured = self._compile_structured_diamond(proto, plan)
        if structured is not None:
            return structured

        namespace = self._base_namespace(plan)
        nodes_by_pc = self._nodes_by_pc(plan)
        lines = [
            "def _jit_cfg_value_ir_function(vm, frames, frame, budget, meter):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
        ]
        for index in range(proto.param_count):
            lines.append(f"    _arg{index} = regs[{index}]")
        lines.extend(["    _state = 0", "    _pred = -1", "    while True:"])

        for block in plan.blocks:
            lines.append(f"        if _state == {block.index}:")
            body = "            "
            for phi_id in block.phi_nodes:
                phi = plan.node(phi_id)
                predecessors = tuple(int(value) for value in phi.payload)
                for branch_index, (pred, source) in enumerate(zip(predecessors, phi.args)):
                    keyword = "if" if branch_index == 0 else "elif"
                    lines.append(f"{body}{keyword} _pred == {pred}:")
                    lines.append(
                        f"{body}    _v{phi_id} = {self._value_expr(plan, source)}"
                    )
                lines.append(f"{body}else:")
                lines.append(
                    f"{body}    raise RuntimeError('CFG value IR predecessor mismatch')"
                )

            lines.extend(self._emit_block_preflight(plan, block, body))
            lines.extend(self._emit_block_expressions(plan, block, nodes_by_pc, body))
            terminal = block.instructions[-1][1]
            returned = self._emit_return(plan, block, body)
            if returned is not None:
                lines.extend(returned)
            elif terminal.op is Op.JMP:
                lines.extend(
                    [
                        f"{body}_pred = {block.index}",
                        f"{body}_state = {block.successors[0]}",
                        f"{body}continue",
                    ]
                )
            elif terminal.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
                if block.condition_value is None or len(block.successors) != 2:
                    return super()._compile_ast_function(proto)
                condition = self._value_expr(plan, block.condition_value)
                test = self._condition_test(block, condition)
                if test is None:
                    return super()._compile_ast_function(proto)
                target, fallthrough = block.successors
                lines.extend(
                    [
                        f"{body}_pred = {block.index}",
                        f"{body}_state = {target} if {test} else {fallthrough}",
                        f"{body}continue",
                    ]
                )
            elif block.successors:
                lines.extend(
                    [
                        f"{body}_pred = {block.index}",
                        f"{body}_state = {block.successors[0]}",
                        f"{body}continue",
                    ]
                )
            else:
                return super()._compile_ast_function(proto)

        lines.append("        raise RuntimeError('CFG value IR state mismatch')")
        tree = ast.parse("\n".join(lines))
        ast.fix_missing_locations(tree)
        exec(compile(tree, "<luapyre-cfg-value-ir-function>", "exec"), namespace)
        return CompiledAstFunction(proto, namespace["_jit_cfg_value_ir_function"])
