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
    """0.17 value-numbered whole-function backend.

    This tier sits in front of the 0.16 typed-IR function compiler. It accepts a
    deliberately exact class of fully typed functions whose reachable execution
    can be represented as a pure SSA-like value graph. Values are promoted out
    of Lua registers, aliases disappear, constant subgraphs fold, repeated
    expressions share one value number, dead pure definitions emit no Python,
    and eligible static lexical CALLs splice the callee value graph directly into
    the caller.

    Unsupported, effectful, escaping-closure, dynamic-call, branch, upvalue and
    recursive shapes immediately fall through to the proven 0.16/0.15 tiers.
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

        live_exprs = [
            node
            for node in plan.nodes
            if node.kind is ValueKind.EXPRESSION and node.id in plan.live_nodes
        ]
        if not live_exprs and not plan.folded_pcs and not plan.call_sites:
            return super()._compile_ast_function(proto)

        namespace: dict[str, object] = {
            "_FUNC_RETURN": _FUNC_RETURN,
            "_FUNC_SUSPEND": _FUNC_SUSPEND,
            "_MASK64": _MASK64,
            "_SIGN64": _SIGN64,
            "_TWO64": _TWO64,
            "_cost": plan.instruction_count,
        }
        for node in plan.nodes:
            if node.kind is ValueKind.LITERAL:
                namespace[f"_lit_{node.id}"] = node.payload

        lines = [
            "def _jit_value_ir_function(vm, frames, frame, budget, meter):",
            "    if budget - meter[0] < _cost:",
            "        frame.pc = 0",
            "        return _FUNC_SUSPEND, None",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
        ]
        for index in range(proto.param_count):
            lines.append(f"    _arg{index} = regs[{index}]")

        # Value IDs are allocated after their operands, including expressions
        # cloned from an inlined child. Emitting live EXPRESSION nodes in ID order
        # is therefore a topological schedule with no Lua-register traffic.
        for node in plan.nodes:
            if node.kind is not ValueKind.EXPRESSION or node.id not in plan.live_nodes:
                continue
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
