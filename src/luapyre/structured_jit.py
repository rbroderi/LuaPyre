from __future__ import annotations

import ast
from types import FunctionType

from .bytecode import Closure, Op
from .jit import CompiledLoop, IRBlock, IRInstruction, IRLoop
from .values import type_matches


_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63
_TWO64 = 1 << 64
_INT_MIN = -(1 << 63)
_INT_MAX = (1 << 63) - 1


class StructuredTypedLoopJITMixin:
    """Fast path for straight-line fully typed numeric loops.

    The general 0.15 backend intentionally starts from a local jump-list CFG and
    AST-inlines the block functions. That representation is excellent for nested
    loops and irregular control flow, but a single straight-line block does not
    need a state machine at all. This mixin recognizes that case and emits one
    native Python ``while`` with promoted locals. Small static typed callees are
    spliced into the body, eliminating Lua frame creation and Python block
    dispatch simultaneously.
    """

    _STRUCTURED_OPS = frozenset(
        {
            Op.LOADK,
            Op.MOVE,
            Op.LOCAL,
            Op.ADD_I,
            Op.ADD_F,
            Op.SUB_I,
            Op.SUB_F,
            Op.MUL_I,
            Op.MUL_F,
            Op.NOT,
            Op.TOBOOL,
            Op.CALL,
        }
    )

    _STRUCTURED_LEAF_OPS = frozenset(
        {
            Op.LOADK,
            Op.MOVE,
            Op.LOCAL,
            Op.ADD_I,
            Op.ADD_F,
            Op.SUB_I,
            Op.SUB_F,
            Op.MUL_I,
            Op.MUL_F,
            Op.NOT,
            Op.TOBOOL,
            Op.RETURN,
        }
    )

    def _compile_loop(self, frame, start_pc: int, backedge_pc: int):
        if frame.proto.jit_fully_typed:
            compiled = self._compile_structured_typed_loop(
                frame, start_pc, backedge_pc
            )
            if compiled is not None:
                return compiled
        return super()._compile_loop(frame, start_pc, backedge_pc)

    @staticmethod
    def _i64_lines(dest: str, expression: str, tag: str, indent: str) -> list[str]:
        tmp = f"_i64_{tag}"
        return [
            f"{indent}{tmp} = ({expression}) & _MASK64",
            f"{indent}{dest} = {tmp} - _TWO64 if {tmp} & _SIGN64 else {tmp}",
        ]

    @staticmethod
    def _leaf_sequence(closure: Closure):
        proto = closure.proto
        if (
            not proto.jit_fully_typed
            or proto.is_vararg
            or proto.upvalues
            or proto.children
        ):
            return None
        sequence = []
        for pc, ins in enumerate(proto.code):
            if ins.op not in StructuredTypedLoopJITMixin._STRUCTURED_LEAF_OPS:
                return None
            sequence.append((pc, ins))
            if ins.op is Op.RETURN:
                break
        if not sequence or sequence[-1][1].op is not Op.RETURN:
            return None
        if len(sequence) > 24:
            return None
        return tuple(sequence)

    @staticmethod
    def _writes_register(ins, reg: int) -> bool:
        if ins.op in (
            Op.LOADK,
            Op.MOVE,
            Op.LOCAL,
            Op.ADD_I,
            Op.ADD_F,
            Op.SUB_I,
            Op.SUB_F,
            Op.MUL_I,
            Op.MUL_F,
            Op.NOT,
            Op.TOBOOL,
        ):
            return ins.a == reg
        if ins.op is Op.CALL and ins.e > 0:
            return ins.a <= reg < ins.a + ins.e
        return False

    def _compile_structured_typed_loop(self, frame, start_pc: int, backedge_pc: int):
        proto = frame.proto
        body = proto.code[start_pc:backedge_pc]
        if not body or any(ins.op not in self._STRUCTURED_OPS for ins in body):
            return None
        loop_ins = proto.code[backedge_pc]
        if loop_ins.op not in (Op.FORLOOP, Op.JFORLOOP) or loop_ins.d != start_pc:
            return None

        captured = self._captured_registers(proto)
        lowered = tuple(IRInstruction(start_pc + i, ins) for i, ins in enumerate(body))
        ir = IRLoop(
            start_pc,
            backedge_pc,
            backedge_pc + 1,
            IRBlock(start_pc, backedge_pc, lowered),
            loop_ins,
        )
        registers = self._used_registers(ir)
        if captured.intersection(registers):
            return None

        calls: dict[int, tuple[Closure, tuple[tuple[int, object], ...]]] = {}
        extra_cost = 0
        for pc, ins in zip(range(start_pc, backedge_pc), body):
            if ins.op is not Op.CALL:
                continue
            fn = frame.regs[ins.b]
            if not isinstance(fn, Closure):
                return None
            sequence = self._leaf_sequence(fn)
            if sequence is None:
                return None
            # Hoisting the callee guard is valid only if the loop body cannot
            # overwrite the register containing the function object.
            if any(self._writes_register(other, ins.b) for other in body):
                return None
            calls[pc] = (fn, sequence)
            extra_cost += len(sequence)

        iteration_cost = len(body) + 1 + extra_cost
        lines = [
            "def _jit_structured_loop(vm, frame, budget):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
            "    used = 0",
        ]
        for reg in registers:
            lines.append(f"    _r{reg} = regs[{reg}]")

        namespace: dict[str, object] = {
            "_Closure": Closure,
            "_MASK64": _MASK64,
            "_SIGN64": _SIGN64,
            "_TWO64": _TWO64,
            "_INT_MIN": _INT_MIN,
            "_INT_MAX": _INT_MAX,
            "_type_matches": type_matches,
        }
        for pc, (closure, _sequence) in calls.items():
            expected = f"_expected_proto_{pc}"
            # A local function declaration creates a fresh Closure every time
            # the top-level Proto runs.  For the leaf shapes admitted here there
            # are no upvalues/children, so the executable semantics are fully
            # determined by the immutable Proto.  Guarding the Proto rather than
            # the transient Closure keeps the compiled loop valid across runs.
            namespace[expected] = closure.proto
            function_reg = body[pc - start_pc].b
            lines.append(
                f"    if not isinstance(_r{function_reg}, _Closure) or _r{function_reg}.proto is not {expected}:"
            )
            lines.extend(self._spill_lines(registers, "        "))
            lines.extend(
                [
                    f"        frame.pc = {start_pc}",
                    "        return 0, False",
                ]
            )
            for arg_index in range(min(closure.proto.param_count, body[pc - start_pc].d)):
                expected_type = closure.proto.param_types[arg_index].name
                source_reg = body[pc - start_pc].c + arg_index
                lines.append(
                    f"    if not _type_matches({expected_type!r}, _r{source_reg}):"
                )
                lines.extend(self._spill_lines(registers, "        "))
                lines.extend(
                    [
                        f"        frame.pc = {start_pc}",
                        "        return 0, False",
                    ]
                )
            namespace[f"_consts_{pc}"] = closure.proto.constants

        lines.append(f"    while budget - used >= {iteration_cost}:")
        indent = "        "

        for offset, ins in enumerate(body):
            pc = start_pc + offset
            a, b, c = f"_r{ins.a}", f"_r{ins.b}", f"_r{ins.c}"
            if ins.op is Op.LOADK:
                lines.append(f"{indent}{a} = consts[{ins.b}]")
            elif ins.op in (Op.MOVE, Op.LOCAL):
                lines.append(f"{indent}{a} = {b}")
            elif ins.op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                symbol = {Op.ADD_I: "+", Op.SUB_I: "-", Op.MUL_I: "*"}[ins.op]
                lines.extend(
                    self._i64_lines(a, f"{b} {symbol} {c}", str(pc), indent)
                )
            elif ins.op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                symbol = {Op.ADD_F: "+", Op.SUB_F: "-", Op.MUL_F: "*"}[ins.op]
                lines.append(f"{indent}{a} = float({b} {symbol} {c})")
            elif ins.op is Op.NOT:
                lines.append(f"{indent}{a} = ({b} is None or {b} is False)")
            elif ins.op is Op.TOBOOL:
                lines.append(f"{indent}{a} = not ({b} is None or {b} is False)")
            elif ins.op is Op.CALL:
                closure, sequence = calls[pc]
                prefix = f"_inl_{pc}_"

                def r(index: int) -> str:
                    return f"{prefix}r{index}"

                for arg_index in range(closure.proto.param_count):
                    value = (
                        f"_r{ins.c + arg_index}"
                        if arg_index < ins.d
                        else "None"
                    )
                    lines.append(f"{indent}{r(arg_index)} = {value}")
                returned: list[str] | None = None
                for child_pc, child_ins in sequence:
                    ca, cb, cc = r(child_ins.a), r(child_ins.b), r(child_ins.c)
                    if child_ins.op is Op.LOADK:
                        lines.append(
                            f"{indent}{ca} = _consts_{pc}[{child_ins.b}]"
                        )
                    elif child_ins.op in (Op.MOVE, Op.LOCAL):
                        lines.append(f"{indent}{ca} = {cb}")
                    elif child_ins.op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                        symbol = {
                            Op.ADD_I: "+",
                            Op.SUB_I: "-",
                            Op.MUL_I: "*",
                        }[child_ins.op]
                        lines.extend(
                            self._i64_lines(
                                ca,
                                f"{cb} {symbol} {cc}",
                                f"inl_{pc}_{child_pc}",
                                indent,
                            )
                        )
                    elif child_ins.op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                        symbol = {
                            Op.ADD_F: "+",
                            Op.SUB_F: "-",
                            Op.MUL_F: "*",
                        }[child_ins.op]
                        lines.append(f"{indent}{ca} = float({cb} {symbol} {cc})")
                    elif child_ins.op is Op.NOT:
                        lines.append(f"{indent}{ca} = ({cb} is None or {cb} is False)")
                    elif child_ins.op is Op.TOBOOL:
                        lines.append(
                            f"{indent}{ca} = not ({cb} is None or {cb} is False)"
                        )
                    elif child_ins.op is Op.RETURN:
                        returned = [r(child_ins.a + i) for i in range(child_ins.b)]
                        break
                    else:
                        return None
                if returned is None:
                    return None
                if ins.e > 0:
                    for result_index in range(ins.e):
                        value = (
                            returned[result_index]
                            if result_index < len(returned)
                            else "None"
                        )
                        lines.append(f"{indent}_r{ins.a + result_index} = {value}")
            else:
                return None

        idx, limit, step = (
            f"_r{loop_ins.a}",
            f"_r{loop_ins.b}",
            f"_r{loop_ins.c}",
        )
        lines.append(f"{indent}used += {iteration_cost}")
        lines.extend(
            [
                f"{indent}if type({idx}) is int and type({limit}) is int and type({step}) is int:",
                f"{indent}    _next = {idx} + {step}",
                f"{indent}    if _next < _INT_MIN or _next > _INT_MAX or ({step} > 0 and _next > {limit}) or ({step} < 0 and _next < {limit}):",
            ]
        )
        lines.extend(self._spill_lines(registers, indent + "        "))
        lines.extend(
            [
                f"{indent}        frame.pc = {ir.exit_pc}",
                f"{indent}        return used, True",
                f"{indent}    {idx} = _next",
                f"{indent}    continue",
                f"{indent}_next = float({idx}) + float({step})",
                f"{indent}if ({step} > 0 and _next > {limit}) or ({step} < 0 and _next < {limit}):",
            ]
        )
        lines.extend(self._spill_lines(registers, indent + "    "))
        lines.extend(
            [
                f"{indent}    frame.pc = {ir.exit_pc}",
                f"{indent}    return used, True",
                f"{indent}{idx} = _next",
            ]
        )

        lines.extend(self._spill_lines(registers, "    "))
        lines.extend(
            [
                f"    frame.pc = {start_pc}",
                "    return used, used > 0",
            ]
        )

        tree = ast.parse("\n".join(lines))
        ast.fix_missing_locations(tree)
        exec(compile(tree, "<luapyre-structured-loop>", "exec"), namespace)
        runner: FunctionType = namespace["_jit_structured_loop"]
        return CompiledLoop(ir, iteration_cost, runner)
