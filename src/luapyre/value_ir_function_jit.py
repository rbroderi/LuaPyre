from __future__ import annotations

import ast
from copy import copy

from .bytecode import Op, Proto
from .function_jit import CompiledAstFunction, _FUNC_RETURN, _FUNC_SUSPEND
from .typed_ir import TypedIRCompiler
from .value_ir import ValueIRCompiler, ValueIRPlan, ValueKind


_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63
_TWO64 = 1 << 64


class ValueIRFunctionJITMixin:
    """0.17 pure-expression whole-function backend.

    This tier sits in front of the 0.16 typed-IR function compiler. It accepts a
    deliberately small but common class of fully typed leaf functions whose body
    is a straight-line pure expression graph. Values are promoted out of Lua
    registers into SSA-like Python locals, aliases disappear, constant subgraphs
    fold at JIT-compile time, repeated expressions share one value number, and
    dead pure definitions emit no Python operation at all.

    Unsupported or effectful functions immediately fall through to 0.16/0.15.
    """

    @staticmethod
    def _value_expr(plan: ValueIRPlan, node_id: int) -> str:
        node = plan.node(node_id)
        if node.kind is ValueKind.ARGUMENT:
            return f"_arg{int(node.payload)}"
        if node.kind is ValueKind.CONSTANT:
            return f"consts[{int(node.payload)}]"
        if node.kind is ValueKind.LITERAL:
            return f"_lit_{node.id}"
        return f"_v{node.id}"

    def _compile_ast_function(self, proto: Proto) -> CompiledAstFunction | None:
        if proto.children:
            return super()._compile_ast_function(proto)

        # Native source compilation always appends a fallback RETURN after the
        # body, even when an explicit straight-line RETURN already makes it
        # unreachable. Value IR compiles only the actually reachable prefix.
        # Any control-flow opcode before that terminal is unsupported by the
        # value compiler and therefore falls through to the older exact tier.
        terminal_pc = next(
            (pc for pc, ins in enumerate(proto.code) if ins.op in (Op.RETURN, Op.HALT)),
            None,
        )
        if terminal_pc is None:
            return super()._compile_ast_function(proto)
        reachable_code = list(proto.code[: terminal_pc + 1])
        analysis_proto = copy(proto)
        analysis_proto.code = reachable_code
        block_code = (tuple(enumerate(reachable_code)),)
        typed_plan = TypedIRCompiler(analysis_proto).compile(block_code)
        plan = ValueIRCompiler(analysis_proto, typed_plan).compile()
        if plan is None:
            return super()._compile_ast_function(proto)

        # The specialized backend is worthwhile even for a simple live typed
        # expression: unlike the generic function compiler it removes the local
        # block dispatcher and all register-copy traffic. Tiny literal-only
        # functions stay on the older tier to avoid pointless code variants.
        live_exprs = [
            node
            for node in plan.nodes
            if node.kind is ValueKind.EXPRESSION and node.id in plan.live_nodes
        ]
        if not live_exprs and not plan.folded_pcs:
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

        definition_at_pc = {
            pc: node_id
            for node_id, pc in plan.definition_pcs
            if node_id in plan.live_nodes
        }

        lines = [
            "def _jit_value_ir_function(vm, frames, frame, budget, meter):",
            "    if budget - meter[0] < _cost:",
            "        frame.pc = 0",
            "        return _FUNC_SUSPEND, None",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
        ]
        namespace["_cost"] = plan.instruction_count

        for index in range(proto.param_count):
            lines.append(f"    _arg{index} = regs[{index}]")

        for pc in range(plan.return_pc):
            node_id = definition_at_pc.get(pc)
            if node_id is None:
                continue
            node = plan.node(node_id)
            args = [self._value_expr(plan, arg) for arg in node.args]
            dest = f"_v{node.id}"
            op = node.op
            if op in ("add_i", "sub_i", "mul_i"):
                symbol = {"add_i": "+", "sub_i": "-", "mul_i": "*"}[op]
                tmp = f"_wide_{node.id}"
                lines.extend(
                    [
                        f"    {tmp} = ({args[0]} {symbol} {args[1]}) & _MASK64",
                        f"    {dest} = {tmp} - _TWO64 if {tmp} & _SIGN64 else {tmp}",
                    ]
                )
            elif op in ("add_f", "sub_f", "mul_f"):
                symbol = {"add_f": "+", "sub_f": "-", "mul_f": "*"}[op]
                lines.append(f"    {dest} = float({args[0]} {symbol} {args[1]})")
            elif op == "not":
                lines.append(f"    {dest} = ({args[0]} is None or {args[0]} is False)")
            elif op == "tobool":
                lines.append(f"    {dest} = not ({args[0]} is None or {args[0]} is False)")
            elif op == "eq":
                left_type = plan.node(node.args[0]).type_name
                right_type = plan.node(node.args[1]).type_name
                numeric = {"integer", "float", "number"}
                if left_type in numeric and right_type in numeric:
                    lines.append(f"    {dest} = {args[0]} == {args[1]}")
                elif left_type == right_type:
                    lines.append(f"    {dest} = {args[0]} == {args[1]}")
                else:
                    lines.append(f"    {dest} = False")
            elif op in ("lt", "le"):
                symbol = "<" if op == "lt" else "<="
                lines.append(f"    {dest} = {args[0]} {symbol} {args[1]}")
            else:
                return super()._compile_ast_function(proto)

        result_exprs = [self._value_expr(plan, node_id) for node_id in plan.return_values]
        if not result_exprs:
            result = "()"
        elif len(result_exprs) == 1:
            result = f"({result_exprs[0]},)"
        else:
            result = "(" + ", ".join(result_exprs) + ")"
        lines.extend(
            [
                "    meter[0] += _cost",
                f"    return _FUNC_RETURN, {result}",
            ]
        )

        tree = ast.parse("\n".join(lines))
        ast.fix_missing_locations(tree)
        exec(compile(tree, "<luapyre-value-ir-function>", "exec"), namespace)
        return CompiledAstFunction(proto, namespace["_jit_value_ir_function"])
