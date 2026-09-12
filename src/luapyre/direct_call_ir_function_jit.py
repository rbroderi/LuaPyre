from __future__ import annotations

import ast
from copy import copy

from .bytecode import Closure, Ins, Op, Proto
from .call_ir import CallIRPlan, analyze_direct_calls
from .function_jit import CompiledAstFunction, _FUNC_RETURN, _FUNC_SUSPEND


_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63
_TWO64 = 1 << 64


_DIRECT_PARENT_OPS = frozenset(
    {
        Op.LOADK,
        Op.MOVE,
        Op.LOCAL,
        Op.CLOSURE,
        Op.ADD_I,
        Op.SUB_I,
        Op.MUL_I,
        Op.ADD_F,
        Op.SUB_F,
        Op.MUL_F,
        Op.NOT,
        Op.TOBOOL,
        Op.EQ,
        Op.LT,
        Op.LE,
        Op.CALL,
        Op.RETURN,
        Op.HALT,
    }
)


class DirectCallIRFunctionJITMixin:
    """Lower statically resolved non-inlined CALL IR through real Lua frames.

    ValueIRFunctionJITMixin gets first refusal and erases only tiny pure calls
    whose closure/frame identity is proven unobservable. This next tier handles
    larger lexical callees without erasing those semantics: CLOSURE still
    allocates a real Closure, CALL still creates a real VM Frame through
    ``run_compiled_child``, and caller/child share the exact fuel meter.

    The first admitted caller shape is deliberately straight-line. The callee
    may itself contain branches/loops/table operations as long as one of the
    proven older function backends can compile it. Unsupported shapes suspend or
    fail closed to the existing tiers.
    """

    @staticmethod
    def _direct_reachable_proto(proto: Proto) -> Proto | None:
        terminal = next(
            (pc for pc, ins in enumerate(proto.code) if ins.op in (Op.RETURN, Op.HALT)),
            None,
        )
        if terminal is None:
            return None
        result = copy(proto)
        result.code = list(proto.code[: terminal + 1])
        return result

    def _compile_ast_function(self, proto: Proto) -> CompiledAstFunction | None:
        analysis = self._direct_reachable_proto(proto)
        if analysis is None:
            return super()._compile_ast_function(proto)
        code = analysis.code
        if any(ins.op not in _DIRECT_PARENT_OPS for ins in code):
            return super()._compile_ast_function(proto)
        if any(ins.op in (Op.RETURN, Op.HALT) for ins in code[:-1]):
            return super()._compile_ast_function(proto)

        call_plan: CallIRPlan = analyze_direct_calls(analysis)
        if not call_plan.direct_sites:
            return super()._compile_ast_function(proto)
        direct_by_pc = {site.pc: site for site in call_plan.direct_sites}
        if any(ins.op is Op.CALL and pc not in direct_by_pc for pc, ins in enumerate(code)):
            return super()._compile_ast_function(proto)

        # This backend materializes child closures exactly but currently only for
        # children with no captures/nested children, matching analyze_direct_calls.
        for ins in code:
            if ins.op is Op.CLOSURE:
                if ins.b < 0 or ins.b >= len(proto.children):
                    return super()._compile_ast_function(proto)
                child = proto.children[ins.b]
                if child.upvalues or child.children or child.is_vararg:
                    return super()._compile_ast_function(proto)

        registers = tuple(range(max(1, proto.register_count)))
        namespace: dict[str, object] = {
            "_Closure": Closure,
            "_FUNC_RETURN": _FUNC_RETURN,
            "_FUNC_SUSPEND": _FUNC_SUSPEND,
            "_MASK64": _MASK64,
            "_SIGN64": _SIGN64,
            "_TWO64": _TWO64,
        }
        for site in call_plan.direct_sites:
            namespace[f"_child_{site.pc}"] = proto.children[site.child_index]

        lines = [
            "def _jit_direct_call_ir_function(vm, frames, frame, budget, meter):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
            "    used = 0",
        ]
        for reg in registers:
            lines.append(f"    _r{reg} = regs[{reg}]")

        def spill(indent: str = "    ") -> list[str]:
            return [f"{indent}regs[{reg}] = _r{reg}" for reg in registers]

        def suspend(pc: int, indent: str = "    ") -> list[str]:
            return [
                f"{indent}meter[0] += used",
                *spill(indent),
                f"{indent}frame.pc = {pc}",
                f"{indent}return _FUNC_SUSPEND, None",
            ]

        def ensure_fuel(pc: int) -> None:
            lines.append("    if budget - meter[0] - used < 1:")
            lines.extend(suspend(pc, "        "))

        def emit_i64(dest: str, expression: str, tag: str) -> None:
            temp = f"_wide_{tag}"
            lines.extend(
                [
                    f"    {temp} = ({expression}) & _MASK64",
                    f"    {dest} = {temp} - _TWO64 if {temp} & _SIGN64 else {temp}",
                ]
            )

        for pc, ins in enumerate(code):
            ensure_fuel(pc)
            op = ins.op
            a, b, c = f"_r{ins.a}", f"_r{ins.b}", f"_r{ins.c}"

            if op is Op.LOADK:
                lines.extend(["    used += 1", f"    {a} = consts[{ins.b}]"])
            elif op in (Op.MOVE, Op.LOCAL):
                lines.extend(["    used += 1", f"    {a} = {b}"])
            elif op is Op.CLOSURE:
                lines.extend(
                    [
                        "    used += 1",
                        f"    {a} = _Closure(frame.proto.children[{ins.b}], [], frame.closure.env)",
                    ]
                )
            elif op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                symbol = {Op.ADD_I: "+", Op.SUB_I: "-", Op.MUL_I: "*"}[op]
                lines.append("    used += 1")
                emit_i64(a, f"{b} {symbol} {c}", str(pc))
            elif op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                symbol = {Op.ADD_F: "+", Op.SUB_F: "-", Op.MUL_F: "*"}[op]
                lines.extend(["    used += 1", f"    {a} = float({b} {symbol} {c})"])
            elif op is Op.NOT:
                lines.extend(["    used += 1", f"    {a} = ({b} is None or {b} is False)"])
            elif op is Op.TOBOOL:
                lines.extend(["    used += 1", f"    {a} = not ({b} is None or {b} is False)"])
            elif op is Op.EQ:
                lines.extend(["    used += 1", f"    {a} = ({b} == {c})"])
            elif op in (Op.LT, Op.LE):
                symbol = "<" if op is Op.LT else "<="
                lines.extend(["    used += 1", f"    {a} = ({b} {symbol} {c})"])
            elif op is Op.CALL:
                site = direct_by_pc[pc]
                child_name = f"_child_{pc}"
                fn_name = f"_fn_{pc}"
                compiled_name = f"_compiled_{pc}"
                lines.append(f"    {fn_name} = {b}")
                lines.append(
                    f"    if not isinstance({fn_name}, _Closure) or {fn_name}.proto is not {child_name}:"
                )
                lines.extend(suspend(pc, "        "))
                lines.append(f"    {compiled_name} = vm.jit.get_compiled_function({fn_name})")
                lines.append(f"    if {compiled_name} is None:")
                lines.extend(suspend(pc, "        "))
                lines.append("    used += 1")
                # Commit all caller work before entering the child. The child
                # uses the same meter and may return, suspend, or raise.
                lines.extend(
                    [
                        "    meter[0] += used",
                        "    used = 0",
                        *spill("    "),
                        f"    frame.pc = {pc + 1}",
                    ]
                )
                args = ", ".join(f"_r{site.arg_base + i}" for i in range(site.arg_count))
                if site.arg_count == 1:
                    args += ","
                lines.append(
                    f"    _status_{pc}, _values_{pc} = vm.jit.run_compiled_child("
                    f"vm, frames, frame, {fn_name}, ({args}), {site.result_base}, "
                    f"{site.result_count}, budget, meter, {compiled_name})"
                )
                lines.append(f"    if _status_{pc} == _FUNC_SUSPEND:")
                lines.append("        return _FUNC_SUSPEND, None")
                for index in range(site.result_count):
                    lines.append(
                        f"    _r{site.result_base + index} = _values_{pc}[{index}] "
                        f"if {index} < len(_values_{pc}) else None"
                    )
            elif op is Op.RETURN:
                lines.append("    used += 1")
                values = ", ".join(f"_r{ins.a + i}" for i in range(ins.b))
                if ins.b == 1:
                    values += ","
                lines.extend(
                    [
                        "    meter[0] += used",
                        f"    return _FUNC_RETURN, ({values})",
                    ]
                )
            elif op is Op.HALT:
                lines.extend(
                    [
                        "    used += 1",
                        "    meter[0] += used",
                        "    return _FUNC_RETURN, ()",
                    ]
                )
            else:
                return super()._compile_ast_function(proto)

        tree = ast.parse("\n".join(lines))
        ast.fix_missing_locations(tree)
        exec(compile(tree, "<luapyre-direct-call-ir-function>", "exec"), namespace)
        return CompiledAstFunction(proto, namespace["_jit_direct_call_ir_function"])
