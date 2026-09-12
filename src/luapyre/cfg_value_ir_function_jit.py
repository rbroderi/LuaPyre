from __future__ import annotations

import ast

from .bytecode import Op, Proto
from .cfg_value_ir import CFGValueIRCompiler, CFGValueIRPlan
from .function_jit import CompiledAstFunction, _FUNC_RETURN, _FUNC_SUSPEND
from .value_ir import ValueKind


_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63
_TWO64 = 1 << 64


class CFGValueIRFunctionJITMixin:
    """0.18 acyclic CFG value backend with exact block side exits."""

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

    def _compile_ast_function(self, proto: Proto) -> CompiledAstFunction | None:
        plan = CFGValueIRCompiler(proto).compile()
        if plan is None:
            return super()._compile_ast_function(proto)

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

        lines = [
            "def _jit_cfg_value_ir_function(vm, frames, frame, budget, meter):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
        ]
        for index in range(proto.param_count):
            lines.append(f"    _arg{index} = regs[{index}]")
        lines.extend(["    _state = 0", "    _pred = -1", "    while True:"])

        def emit_expr(node_id: int, indent: str) -> list[str]:
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

        nodes_by_pc: dict[int, list[int]] = {}
        for node in plan.nodes:
            if node.kind is ValueKind.EXPRESSION and node.op != "phi" and node.def_pc >= 0:
                nodes_by_pc.setdefault(node.def_pc, []).append(node.id)

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

            lines.append(f"{body}if budget - meter[0] < {block.cost}:")
            entry = dict(block.entry_values)
            for register in block.live_in:
                lines.append(
                    f"{body}    regs[{register}] = {self._value_expr(plan, entry[register])}"
                )
            lines.extend(
                [
                    f"{body}    frame.pc = {block.start}",
                    f"{body}    return _FUNC_SUSPEND, None",
                ]
            )

            for pc, _ins in block.instructions:
                for node_id in nodes_by_pc.get(pc, ()):
                    lines.extend(emit_expr(node_id, body))
            lines.append(f"{body}meter[0] += {block.cost}")

            terminal = block.instructions[-1][1]
            if terminal.op is Op.RETURN:
                values = [self._value_expr(plan, node_id) for node_id in block.return_values]
                if not values:
                    result = "()"
                elif len(values) == 1:
                    result = f"({values[0]},)"
                else:
                    result = "(" + ", ".join(values) + ")"
                lines.append(f"{body}return _FUNC_RETURN, {result}")
            elif terminal.op is Op.HALT:
                lines.append(f"{body}return _FUNC_RETURN, ()")
            elif terminal.op is Op.JMP:
                target = block.successors[0]
                lines.extend(
                    [
                        f"{body}_pred = {block.index}",
                        f"{body}_state = {target}",
                        f"{body}continue",
                    ]
                )
            elif terminal.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
                if block.condition_value is None or len(block.successors) != 2:
                    return super()._compile_ast_function(proto)
                condition = self._value_expr(plan, block.condition_value)
                if terminal.op is Op.JMPIF:
                    test = f"not ({condition} is None or {condition} is False)"
                elif terminal.op is Op.JMPIFNOT:
                    test = f"({condition} is None or {condition} is False)"
                else:
                    test = f"{condition} is None"
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
