from __future__ import annotations

import ast
from dataclasses import dataclass
from types import FunctionType

from .ast_backend import JumpListLayout, inline_expression_helper, inline_local_jump_list
from .bytecode import Closure, Ins, Op, Proto
from .errors import LuaRuntimeError
from .jit import CompiledLoop, IRBlock, IRInstruction, IRLoop
from .jit_policy import TYPED_JIT_LOOP_BODY_OPS
from .opdispatch import _float_divide, _float_modulo, _float_power, _shift, _to_lua_string
from .region_jit import RegionPythonJIT
from .table import LuaTable
from .values import coerce_lua_integer, lua_equal, static_value_type, type_matches


_INT_MIN = -(1 << 63)
_INT_MAX = (1 << 63) - 1
_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63
_TWO64 = 1 << 64

_BACKEDGE_STATE = -1
_DEOPT_STATE = -2

_CONTROL_OPS = frozenset(
    (
        Op.JMP,
        Op.JMPIF,
        Op.JMPIFNOT,
        Op.JMPIFNIL,
        Op.FORPREP,
        Op.FORLOOP,
        Op.JFORLOOP,
    )
)


@dataclass(frozen=True, slots=True)
class _AstBlock:
    start: int
    end: int
    instructions: tuple[IRInstruction, ...]


class AstPythonJIT(RegionPythonJIT):
    """0.15 typed super-region backend.

    Fully typed hot loops are promoted from Lua register-list traffic to Python
    locals.  Their CFG is first represented as local jump-list block functions;
    an AST pass then inlines those block functions into one match dispatcher so
    the executable fast path pays neither Python helper-call nor jump-list-call
    overhead.  Straight-line/ordinary source can still use the 0.13/0.14
    backends, and every unsupported dynamic shape fails closed to Tier 0.
    """

    def _compile_loop(self, frame, start_pc: int, backedge_pc: int):
        if frame.proto.jit_fully_typed:
            compiled = self._compile_ast_loop(frame, start_pc, backedge_pc)
            if compiled is not None:
                return compiled
        return super()._compile_loop(frame, start_pc, backedge_pc)

    @staticmethod
    def _reg(index: int) -> str:
        return f"_r{index}"

    @staticmethod
    def _i64_lines(dest: str, expression: str, tag: str, indent: str) -> list[str]:
        tmp = f"_i64_{tag}"
        return [
            f"{indent}{tmp} = ({expression}) & _MASK64",
            f"{indent}{dest} = {tmp} - _TWO64 if {tmp} & _SIGN64 else {tmp}",
        ]

    @staticmethod
    def _successor_target(ins: Ins, fallthrough: int) -> tuple[int, ...]:
        if ins.op is Op.JMP:
            return (ins.a,)
        if ins.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
            return (ins.a, fallthrough)
        if ins.op is Op.FORPREP:
            return (ins.d, fallthrough)
        if ins.op in (Op.FORLOOP, Op.JFORLOOP):
            return (ins.d, fallthrough)
        return (fallthrough,)

    def _lower_ast_region(
        self, frame, start_pc: int, backedge_pc: int
    ) -> tuple[IRLoop, tuple[_AstBlock, ...]] | None:
        proto = frame.proto
        body = proto.code[start_pc:backedge_pc]
        if not body or any(ins.op not in TYPED_JIT_LOOP_BODY_OPS for ins in body):
            return None
        loop_ins = proto.code[backedge_pc]
        if loop_ins.op not in (Op.FORLOOP, Op.JFORLOOP) or loop_ins.d != start_pc:
            return None

        lowered: list[IRInstruction] = []
        leaders = {start_pc, backedge_pc}
        for offset, ins in enumerate(body):
            pc = start_pc + offset
            specialization = None
            if ins.op in (Op.ADD, Op.SUB, Op.MUL, Op.MOD, Op.EQ, Op.LT, Op.LE):
                specialization = self._profile_binary(frame.regs, ins)
            lowered.append(IRInstruction(pc, ins, specialization))

            # Calls are block boundaries even though they do not alter the CFG:
            # a direct child consumes a dynamic amount of fuel and may suspend
            # back into Tier 0, so no later instruction may be prepaid.
            if ins.op is Op.CALL:
                if pc + 1 <= backedge_pc:
                    leaders.add(pc + 1)
                continue
            if ins.op in _CONTROL_OPS:
                targets = self._successor_target(ins, pc + 1)
                for target in targets:
                    if target < start_pc or target > backedge_pc:
                        return None
                    leaders.add(target)
                if pc + 1 <= backedge_pc:
                    leaders.add(pc + 1)

        ordered = sorted(leaders)
        by_pc = {item.pc: item for item in lowered}
        blocks: list[_AstBlock] = []
        for index, leader in enumerate(ordered[:-1]):
            end = ordered[index + 1]
            instructions = tuple(by_pc[pc] for pc in range(leader, end))
            if not instructions:
                continue
            if any(item.ins.op in _CONTROL_OPS for item in instructions[:-1]):
                return None
            if any(item.ins.op is Op.CALL for item in instructions[:-1]):
                return None
            blocks.append(_AstBlock(leader, end, instructions))

        starts = {block.start for block in blocks}
        starts.add(backedge_pc)
        for block in blocks:
            terminal = block.instructions[-1].ins
            if terminal.op in _CONTROL_OPS:
                for target in self._successor_target(terminal, block.end):
                    if target not in starts:
                        return None

        ir = IRLoop(
            start_pc,
            backedge_pc,
            backedge_pc + 1,
            IRBlock(start_pc, backedge_pc, tuple(lowered)),
            loop_ins,
        )
        return ir, tuple(blocks)

    @staticmethod
    def _used_registers(ir: IRLoop) -> tuple[int, ...]:
        regs: set[int] = {
            ir.loop_ins.a,
            ir.loop_ins.b,
            ir.loop_ins.c,
        }
        for item in ir.body.instructions:
            ins, op = item.ins, item.ins.op
            if op is Op.LOADK:
                regs.add(ins.a)
            elif op in (Op.MOVE, Op.LOCAL, Op.NEG, Op.NOT, Op.TOBOOL, Op.LEN, Op.BNOT):
                regs.update((ins.a, ins.b))
            elif op in (
                Op.ADD, Op.ADD_I, Op.ADD_F,
                Op.SUB, Op.SUB_I, Op.SUB_F,
                Op.MUL, Op.MUL_I, Op.MUL_F,
                Op.DIV, Op.IDIV, Op.MOD, Op.POW,
                Op.BAND, Op.BOR, Op.BXOR, Op.SHL, Op.SHR,
                Op.CONCAT, Op.EQ, Op.LT, Op.LE,
                Op.GETTABLE,
            ):
                regs.update((ins.a, ins.b, ins.c))
            elif op is Op.SETTABLE:
                regs.update((ins.a, ins.b, ins.c))
            elif op is Op.NEWTABLE:
                regs.add(ins.a)
            elif op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
                regs.add(ins.b)
            elif op in (Op.FORPREP, Op.FORLOOP, Op.JFORLOOP):
                regs.update((ins.a, ins.b, ins.c))
            elif op is Op.GUARD:
                regs.add(ins.a)
            elif op is Op.CALL:
                regs.update((ins.a, ins.b))
                regs.update(range(ins.c, ins.c + ins.d))
                if ins.e > 0:
                    regs.update(range(ins.a, ins.a + ins.e))
        return tuple(sorted(regs))

    @staticmethod
    def _captured_registers(proto: Proto) -> set[int]:
        return {
            desc.index
            for child in proto.children
            for desc in child.upvalues
            if desc.kind == "local"
        }

    @staticmethod
    def _spill_lines(registers: tuple[int, ...], indent: str) -> list[str]:
        return [f"{indent}regs[{reg}] = _r{reg}" for reg in registers]

    def _deopt_lines(
        self,
        registers: tuple[int, ...],
        pc: int,
        indent: str,
    ) -> list[str]:
        return [
            *self._spill_lines(registers, indent),
            f"{indent}frame.pc = {pc}",
            f"{indent}return _DEOPT_STATE",
        ]

    def _emit_forprep(
        self,
        lines: list[str],
        ins: Ins,
        *,
        true_state: int,
        false_state: int,
        indent: str,
    ) -> None:
        idx, limit, step = self._reg(ins.a), self._reg(ins.b), self._reg(ins.c)
        lines.extend(
            [
                f"{indent}if not (type({idx}) in _NUM_TYPES and type({limit}) in _NUM_TYPES and type({step}) in _NUM_TYPES):",
                f"{indent}    raise _LuaRuntimeError(\"'for' limit must be a number\")",
                f"{indent}if {step} == 0:",
                f"{indent}    raise _LuaRuntimeError(\"'for' step is zero\")",
                f"{indent}if type({idx}) is float or type({limit}) is float or type({step}) is float:",
                f"{indent}    {idx} = float({idx})",
                f"{indent}    {limit} = float({limit})",
                f"{indent}    {step} = float({step})",
                f"{indent}if ({idx} <= {limit} if {step} > 0 else {idx} >= {limit}):",
                f"{indent}    return {true_state}",
                f"{indent}return {false_state}",
            ]
        )

    def _emit_forloop(
        self,
        lines: list[str],
        ins: Ins,
        *,
        loop_state: int,
        exit_state: int,
        indent: str,
    ) -> None:
        idx, limit, step = self._reg(ins.a), self._reg(ins.b), self._reg(ins.c)
        tag = f"for_{ins.a}_{ins.d}"
        lines.extend(
            [
                f"{indent}if type({idx}) is int and type({limit}) is int and type({step}) is int:",
                f"{indent}    _next_{tag} = {idx} + {step}",
                f"{indent}    if _next_{tag} < _INT_MIN or _next_{tag} > _INT_MAX:",
                f"{indent}        return {exit_state}",
                f"{indent}    if ({step} > 0 and _next_{tag} > {limit}) or ({step} < 0 and _next_{tag} < {limit}):",
                f"{indent}        return {exit_state}",
                f"{indent}    {idx} = _next_{tag}",
                f"{indent}    return {loop_state}",
                f"{indent}_next_{tag} = float({idx}) + float({step})",
                f"{indent}if ({step} > 0 and _next_{tag} > {limit}) or ({step} < 0 and _next_{tag} < {limit}):",
                f"{indent}    return {exit_state}",
                f"{indent}{idx} = _next_{tag}",
                f"{indent}return {loop_state}",
            ]
        )

    def _inline_leaf_call(
        self,
        frame,
        item: IRInstruction,
        registers: tuple[int, ...],
        expected_name: str,
        const_name: str,
        indent: str,
    ) -> tuple[list[str], Closure, int] | None:
        ins = item.ins
        fn = frame.regs[ins.b]
        if not isinstance(fn, Closure) or not fn.proto.jit_fully_typed:
            return None
        proto = fn.proto
        if proto.is_vararg or proto.upvalues or proto.children:
            return None
        allowed = frozenset(
            {
                Op.LOADK, Op.MOVE, Op.LOCAL,
                Op.ADD_I, Op.ADD_F, Op.SUB_I, Op.SUB_F, Op.MUL_I, Op.MUL_F,
                Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD,
                Op.NOT, Op.TOBOOL, Op.EQ, Op.LT, Op.LE, Op.GUARD,
                Op.RETURN,
            }
        )
        sequence: list[tuple[int, Ins]] = []
        for pc, child_ins in enumerate(proto.code):
            if child_ins.op not in allowed:
                return None
            sequence.append((pc, child_ins))
            if child_ins.op is Op.RETURN:
                break
        if not sequence or sequence[-1][1].op is not Op.RETURN:
            return None
        # Straight-line means no control-flow op appeared; the allowed set makes
        # that true. Keep inlining bounded so one caller cannot explode Python
        # bytecode size.
        if len(sequence) > 24:
            return None

        prefix = f"_inl_{item.pc}_"
        r = lambda index: f"{prefix}r{index}"
        lines: list[str] = [
            f"{indent}if {self._reg(ins.b)} is not {expected_name}:",
            *[f"{indent}    {line.strip()}" for line in self._deopt_lines(registers, item.pc, "")],
        ]
        for arg_index in range(min(proto.param_count, ins.d)):
            expected = proto.param_types[arg_index].name
            source = self._reg(ins.c + arg_index)
            lines.extend(
                [
                    f"{indent}if not _type_matches({expected!r}, {source}):",
                    *[f"{indent}    {line.strip()}" for line in self._deopt_lines(registers, item.pc, "")],
                ]
            )
        for arg_index in range(proto.param_count):
            source = self._reg(ins.c + arg_index) if arg_index < ins.d else "None"
            lines.append(f"{indent}{r(arg_index)} = {source}")

        child_cost = 0
        returned: list[str] | None = None
        for child_pc, child_ins in sequence:
            op = child_ins.op
            child_cost += 1
            lines.append(f"{indent}used += 1")
            if op is Op.LOADK:
                lines.append(f"{indent}{r(child_ins.a)} = {const_name}[{child_ins.b}]")
            elif op in (Op.MOVE, Op.LOCAL):
                lines.append(f"{indent}{r(child_ins.a)} = {r(child_ins.b)}")
            elif op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                symbol = {Op.ADD_I: "+", Op.SUB_I: "-", Op.MUL_I: "*"}[op]
                lines.extend(
                    self._i64_lines(
                        r(child_ins.a),
                        f"{r(child_ins.b)} {symbol} {r(child_ins.c)}",
                        f"inline_{item.pc}_{child_pc}",
                        indent,
                    )
                )
            elif op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                symbol = {Op.ADD_F: "+", Op.SUB_F: "-", Op.MUL_F: "*"}[op]
                lines.append(
                    f"{indent}{r(child_ins.a)} = float({r(child_ins.b)} {symbol} {r(child_ins.c)})"
                )
            elif op in (Op.ADD, Op.SUB, Op.MUL):
                symbol = {Op.ADD: "+", Op.SUB: "-", Op.MUL: "*"}[op]
                left, right = r(child_ins.b), r(child_ins.c)
                tmp = f"{prefix}v{child_pc}"
                lines.append(f"{indent}{tmp} = {left} {symbol} {right}")
                lines.append(f"{indent}if type({left}) is int and type({right}) is int:")
                lines.extend(
                    self._i64_lines(
                        r(child_ins.a), tmp, f"inline_generic_{item.pc}_{child_pc}", indent + "    "
                    )
                )
                lines.append(f"{indent}else:")
                lines.append(f"{indent}    {r(child_ins.a)} = {tmp}")
            elif op is Op.DIV:
                lines.append(
                    f"{indent}{r(child_ins.a)} = _float_divide({r(child_ins.b)}, {r(child_ins.c)})"
                )
            elif op is Op.MOD:
                left, right = r(child_ins.b), r(child_ins.c)
                lines.append(f"{indent}if type({left}) is int and type({right}) is int:")
                lines.append(f"{indent}    if {right} == 0:")
                lines.append(f"{indent}        raise _LuaRuntimeError(\"attempt to perform 'n%0'\")")
                lines.extend(
                    self._i64_lines(
                        r(child_ins.a), f"{left} % {right}", f"inline_mod_{item.pc}_{child_pc}", indent + "    "
                    )
                )
                lines.append(f"{indent}else:")
                lines.append(
                    f"{indent}    {r(child_ins.a)} = _float_modulo({left}, {right})"
                )
            elif op is Op.NOT:
                lines.append(f"{indent}{r(child_ins.a)} = not _truthy({r(child_ins.b)})")
            elif op is Op.TOBOOL:
                lines.append(f"{indent}{r(child_ins.a)} = _truthy({r(child_ins.b)})")
            elif op is Op.GUARD:
                lines.append(
                    f"{indent}if not _type_matches({const_name}[{child_ins.b}], {r(child_ins.a)}):"
                )
                lines.append(
                    f"{indent}    raise _LuaRuntimeError(f\"expected {{{const_name}[{child_ins.b}]!s}}, got {{_static_value_type({r(child_ins.a)}).name}}\")"
                )
            elif op is Op.EQ:
                lines.append(
                    f"{indent}{r(child_ins.a)} = _lua_equal({r(child_ins.b)}, {r(child_ins.c)})"
                )
            elif op in (Op.LT, Op.LE):
                symbol = "<" if op is Op.LT else "<="
                lines.append(
                    f"{indent}{r(child_ins.a)} = {r(child_ins.b)} {symbol} {r(child_ins.c)}"
                )
            elif op is Op.RETURN:
                returned = [r(child_ins.a + i) for i in range(child_ins.b)]
                break
            else:
                return None

        if returned is None:
            return None
        # CALL itself is an instruction in addition to every child instruction.
        # The caller emits its own +1; this returned cost is used only for the
        # preflight budget check.
        if ins.e > 0:
            for index in range(ins.e):
                value = returned[index] if index < len(returned) else "None"
                lines.append(f"{indent}{self._reg(ins.a + index)} = {value}")
        return lines, fn, child_cost

    def _emit_instruction(
        self,
        frame,
        item: IRInstruction,
        *,
        registers: tuple[int, ...],
        captured: set[int],
        indent: str,
        expected_calls: dict[int, tuple[str, str]],
    ) -> list[str] | None:
        ins, pc, op = item.ins, item.pc, item.ins.op
        a, b, c = self._reg(ins.a), self._reg(ins.b), self._reg(ins.c)
        out: list[str] = []

        def deopt(condition: str) -> None:
            out.append(f"{indent}if {condition}:")
            out.extend(
                f"{indent}    {line.strip()}"
                for line in self._deopt_lines(registers, pc, "")
            )

        if op is Op.LOADK:
            out.extend([f"{indent}used += 1", f"{indent}{a} = consts[{ins.b}]"])
        elif op is Op.MOVE:
            out.extend([f"{indent}used += 1", f"{indent}{a} = {b}"])
        elif op is Op.LOCAL:
            out.extend([f"{indent}used += 1", f"{indent}{a} = {b}"])
            if ins.a in captured:
                out.extend(
                    [
                        f"{indent}if {ins.a} in cells:",
                        f"{indent}    cells[{ins.a}].value = {a}",
                    ]
                )
        elif op is Op.NEWTABLE:
            out.extend([f"{indent}used += 1", f"{indent}{a} = _LuaTable()"])
        elif op is Op.GETTABLE:
            deopt(f"not isinstance({b}, _LuaTable) or {b}.metatable is not None")
            out.extend([f"{indent}used += 1", f"{indent}{a} = {b}.rawget({c})"])
        elif op is Op.SETTABLE:
            deopt(f"not isinstance({a}, _LuaTable) or {a}.metatable is not None")
            deopt(f"{b} is None or (type({b}) is float and _isnan({b}))")
            out.extend([f"{indent}used += 1", f"{indent}{a}.rawset({b}, {c})"])
        elif op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
            symbol = {Op.ADD_I: "+", Op.SUB_I: "-", Op.MUL_I: "*"}[op]
            out.append(f"{indent}used += 1")
            out.extend(self._i64_lines(a, f"{b} {symbol} {c}", str(pc), indent))
        elif op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
            symbol = {Op.ADD_F: "+", Op.SUB_F: "-", Op.MUL_F: "*"}[op]
            out.extend(
                [
                    f"{indent}used += 1",
                    f"{indent}{a} = float({b} {symbol} {c})",
                ]
            )
        elif op in (Op.ADD, Op.SUB, Op.MUL):
            symbol = {Op.ADD: "+", Op.SUB: "-", Op.MUL: "*"}[op]
            deopt(f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES")
            tmp = f"_arith_{pc}"
            out.extend([f"{indent}used += 1", f"{indent}{tmp} = {b} {symbol} {c}"])
            out.append(f"{indent}if type({b}) is int and type({c}) is int:")
            out.extend(self._i64_lines(a, tmp, f"generic_{pc}", indent + "    "))
            out.append(f"{indent}else:")
            out.append(f"{indent}    {a} = {tmp}")
        elif op is Op.DIV:
            deopt(f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES")
            out.extend(
                [
                    f"{indent}used += 1",
                    f"{indent}{a} = _float_divide({b}, {c})",
                ]
            )
        elif op is Op.IDIV:
            deopt(f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES")
            out.append(f"{indent}used += 1")
            out.append(f"{indent}if {c} == 0:")
            out.append(f"{indent}    if type({b}) is int and type({c}) is int:")
            out.append(f"{indent}        raise _LuaRuntimeError(\"attempt to divide by zero\")")
            out.append(f"{indent}    {a} = _float_divide({b}, {c})")
            out.append(f"{indent}elif type({b}) is int and type({c}) is int:")
            out.extend(self._i64_lines(a, f"{b} // {c}", f"idiv_{pc}", indent + "    "))
            out.append(f"{indent}else:")
            out.append(f"{indent}    {a} = float(_floor(float({b}) / float({c})))")
        elif op is Op.MOD:
            deopt(f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES")
            out.append(f"{indent}used += 1")
            out.append(f"{indent}if type({b}) is int and type({c}) is int:")
            out.append(f"{indent}    if {c} == 0:")
            out.append(f"{indent}        raise _LuaRuntimeError(\"attempt to perform 'n%0'\")")
            out.extend(self._i64_lines(a, f"{b} % {c}", f"mod_{pc}", indent + "    "))
            out.append(f"{indent}else:")
            out.append(f"{indent}    {a} = _float_modulo({b}, {c})")
        elif op is Op.POW:
            deopt(f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES")
            out.extend([f"{indent}used += 1", f"{indent}{a} = _float_power({b}, {c})"])
        elif op in (Op.BAND, Op.BOR, Op.BXOR, Op.SHL, Op.SHR):
            left = f"_bit_l_{pc}"
            right = f"_bit_r_{pc}"
            out.extend(
                [
                    f"{indent}{left} = _coerce_lua_integer({b})",
                    f"{indent}{right} = _coerce_lua_integer({c})",
                ]
            )
            deopt(f"{left} is None or {right} is None")
            out.append(f"{indent}used += 1")
            if op in (Op.SHL, Op.SHR):
                out.append(
                    f"{indent}{a} = _shift({left}, {right}, left={op is Op.SHL})"
                )
            else:
                symbol = {Op.BAND: "&", Op.BOR: "|", Op.BXOR: "^"}[op]
                out.extend(self._i64_lines(a, f"{left} {symbol} {right}", f"bit_{pc}", indent))
        elif op is Op.BNOT:
            value = f"_bit_{pc}"
            out.append(f"{indent}{value} = _coerce_lua_integer({b})")
            deopt(f"{value} is None")
            out.append(f"{indent}used += 1")
            out.extend(self._i64_lines(a, f"~{value}", f"bnot_{pc}", indent))
        elif op is Op.NEG:
            deopt(f"type({b}) not in _NUM_TYPES")
            out.append(f"{indent}used += 1")
            out.append(f"{indent}if type({b}) is int:")
            out.extend(self._i64_lines(a, f"-{b}", f"neg_{pc}", indent + "    "))
            out.append(f"{indent}else:")
            out.append(f"{indent}    {a} = -{b}")
        elif op is Op.CONCAT:
            deopt(
                f"not isinstance({b}, (bytes, int, float)) or not isinstance({c}, (bytes, int, float)) or type({b}) is bool or type({c}) is bool"
            )
            out.extend(
                [
                    f"{indent}used += 1",
                    f"{indent}{a} = _to_lua_string({b}) + _to_lua_string({c})",
                ]
            )
        elif op is Op.LEN:
            out.append(f"{indent}if isinstance({b}, bytes):")
            out.append(f"{indent}    used += 1")
            out.append(f"{indent}    {a} = len({b})")
            out.append(f"{indent}elif isinstance({b}, _LuaTable) and {b}.metatable is None:")
            out.append(f"{indent}    used += 1")
            out.append(f"{indent}    {a} = {b}.rawlen()")
            out.append(f"{indent}else:")
            out.extend(
                f"{indent}    {line.strip()}"
                for line in self._deopt_lines(registers, pc, "")
            )
        elif op is Op.NOT:
            out.extend([f"{indent}used += 1", f"{indent}{a} = not _truthy({b})"])
        elif op is Op.TOBOOL:
            out.extend([f"{indent}used += 1", f"{indent}{a} = _truthy({b})"])
        elif op is Op.EQ:
            # Equality on two unequal metatable-bearing tables can invoke __eq.
            deopt(
                f"isinstance({b}, _LuaTable) and isinstance({c}, _LuaTable) and not _lua_equal({b}, {c}) and ({b}.metatable is not None or {c}.metatable is not None)"
            )
            out.extend([f"{indent}used += 1", f"{indent}{a} = _lua_equal({b}, {c})"])
        elif op in (Op.LT, Op.LE):
            symbol = "<" if op is Op.LT else "<="
            deopt(
                f"not ((type({b}) in _NUM_TYPES and type({c}) in _NUM_TYPES) or (isinstance({b}, bytes) and isinstance({c}, bytes)))"
            )
            out.extend([f"{indent}used += 1", f"{indent}{a} = {b} {symbol} {c}"])
        elif op is Op.GUARD:
            deopt(f"not _type_matches(consts[{ins.b}], {a})")
            out.append(f"{indent}used += 1")
        elif op is Op.CALL:
            call_key = expected_calls.get(pc)
            if call_key is None:
                return None
            expected_name, const_name = call_key
            inlined = self._inline_leaf_call(
                frame,
                item,
                registers,
                expected_name,
                const_name,
                indent,
            )
            if inlined is None:
                return None
            call_lines, _closure, child_cost = inlined
            out.append(f"{indent}if budget - used < {1 + child_cost}:")
            out.extend(
                f"{indent}    {line.strip()}"
                for line in self._deopt_lines(registers, pc, "")
            )
            out.append(f"{indent}used += 1")
            out.extend(call_lines)
        else:
            return None
        return out

    def _compile_ast_loop(self, frame, start_pc: int, backedge_pc: int):
        lowered = self._lower_ast_region(frame, start_pc, backedge_pc)
        if lowered is None:
            return None
        ir, blocks = lowered
        registers = self._used_registers(ir)
        captured = self._captured_registers(frame.proto)
        if captured.intersection(registers):
            # Captured register replacement has subtle open-cell identity rules.
            # Leave those loops on the proven 0.14/Tier-0 paths for now.
            return None

        block_index = {block.start: index for index, block in enumerate(blocks)}

        def state_for(pc: int) -> int:
            if pc == backedge_pc:
                return _BACKEDGE_STATE
            return block_index[pc]

        expected_calls: dict[int, tuple[str, str]] = {}
        expected_values: dict[str, object] = {}
        for block in blocks:
            for item in block.instructions:
                if item.ins.op is not Op.CALL:
                    continue
                fn = frame.regs[item.ins.b]
                if not isinstance(fn, Closure):
                    return None
                expected_name = f"_expected_call_{item.pc}"
                const_name = f"_call_consts_{item.pc}"
                expected_calls[item.pc] = (expected_name, const_name)
                expected_values[expected_name] = fn
                expected_values[const_name] = fn.proto.constants

        lines = [
            "def _jit_ast_loop(vm, frame, budget):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
            "    cells = frame.cells",
            "    used = 0",
        ]
        for reg in registers:
            lines.append(f"    _r{reg} = regs[{reg}]")
        lines.append("    _state = 0")

        block_names = tuple(f"_jump_{index}" for index in range(len(blocks)))
        for index, block in enumerate(blocks):
            lines.append(f"    def {block_names[index]}():")
            indent = "        "
            # A block must be all-or-nothing with respect to the quota check.
            # Calls additionally perform their own child-cost preflight.
            block_cost = len(block.instructions)
            lines.append(f"{indent}if budget - used < {block_cost}:")
            for spill in self._spill_lines(registers, indent + "    "):
                lines.append(spill)
            lines.append(f"{indent}    frame.pc = {block.start}")
            lines.append(f"{indent}    return _DEOPT_STATE")

            terminal = block.instructions[-1]
            control = terminal.ins.op in _CONTROL_OPS
            ordinary = block.instructions[:-1] if control else block.instructions
            for item in ordinary:
                emitted = self._emit_instruction(
                    frame,
                    item,
                    registers=registers,
                    captured=captured,
                    indent=indent,
                    expected_calls=expected_calls,
                )
                if emitted is None:
                    return None
                lines.extend(emitted)

            if control:
                ins = terminal.ins
                lines.append(f"{indent}used += 1")
                if ins.op is Op.JMP:
                    lines.append(f"{indent}return {state_for(ins.a)}")
                elif ins.op in (Op.JMPIF, Op.JMPIFNOT):
                    condition = f"_truthy({self._reg(ins.b)})"
                    if ins.op is Op.JMPIFNOT:
                        condition = f"not ({condition})"
                    lines.append(
                        f"{indent}return {state_for(ins.a)} if {condition} else {state_for(block.end)}"
                    )
                elif ins.op is Op.JMPIFNIL:
                    lines.append(
                        f"{indent}return {state_for(ins.a)} if {self._reg(ins.b)} is None else {state_for(block.end)}"
                    )
                elif ins.op is Op.FORPREP:
                    self._emit_forprep(
                        lines,
                        ins,
                        true_state=state_for(block.end),
                        false_state=state_for(ins.d),
                        indent=indent,
                    )
                elif ins.op in (Op.FORLOOP, Op.JFORLOOP):
                    self._emit_forloop(
                        lines,
                        ins,
                        loop_state=state_for(ins.d),
                        exit_state=state_for(block.end),
                        indent=indent,
                    )
                else:
                    return None
            else:
                lines.append(f"{indent}return {state_for(block.end)}")

        lines.append(
            "    _jump_list = (" + ", ".join(block_names) + ",)"
        )
        lines.extend(
            [
                "    while True:",
                f"        _state = {state_for(start_pc)}",
                "        while _state >= 0:",
                "            _state = _jump_list[_state]()",
                "        if _state == _DEOPT_STATE:",
                "            return used, used > 0",
                "        if budget - used < 1:",
            ]
        )
        lines.extend(self._spill_lines(registers, "            "))
        lines.extend(
            [
                f"            frame.pc = {backedge_pc}",
                "            return used, used > 0",
                "        used += 1",
            ]
        )

        outer = ir.loop_ins
        idx, limit, step = self._reg(outer.a), self._reg(outer.b), self._reg(outer.c)
        tag = f"outer_{outer.a}_{backedge_pc}"
        lines.extend(
            [
                f"        if type({idx}) is int and type({limit}) is int and type({step}) is int:",
                f"            _next_{tag} = {idx} + {step}",
                f"            if _next_{tag} < _INT_MIN or _next_{tag} > _INT_MAX or ({step} > 0 and _next_{tag} > {limit}) or ({step} < 0 and _next_{tag} < {limit}):",
            ]
        )
        lines.extend(self._spill_lines(registers, "                "))
        lines.extend(
            [
                f"                frame.pc = {ir.exit_pc}",
                "                return used, True",
                f"            {idx} = _next_{tag}",
                "            continue",
                f"        _next_{tag} = float({idx}) + float({step})",
                f"        if ({step} > 0 and _next_{tag} > {limit}) or ({step} < 0 and _next_{tag} < {limit}):",
            ]
        )
        lines.extend(self._spill_lines(registers, "            "))
        lines.extend(
            [
                f"            frame.pc = {ir.exit_pc}",
                "            return used, True",
                f"        {idx} = _next_{tag}",
            ]
        )

        source = "\n".join(lines)
        tree = ast.parse(source)
        tree = inline_expression_helper(
            tree,
            "def _truthy(value):\n    return not (value is None or value is False)\n",
        )
        tree = inline_local_jump_list(
            tree,
            JumpListLayout("_jump_list", "_state", block_names),
        )
        ast.fix_missing_locations(tree)

        namespace = {
            "_BACKEDGE_STATE": _BACKEDGE_STATE,
            "_DEOPT_STATE": _DEOPT_STATE,
            "_INT_MIN": _INT_MIN,
            "_INT_MAX": _INT_MAX,
            "_MASK64": _MASK64,
            "_SIGN64": _SIGN64,
            "_TWO64": _TWO64,
            "_NUM_TYPES": (int, float),
            "_LuaRuntimeError": LuaRuntimeError,
            "_LuaTable": LuaTable,
            "_coerce_lua_integer": coerce_lua_integer,
            "_float_divide": _float_divide,
            "_float_modulo": _float_modulo,
            "_float_power": _float_power,
            "_floor": __import__("math").floor,
            "_isnan": __import__("math").isnan,
            "_lua_equal": lua_equal,
            "_shift": _shift,
            "_static_value_type": static_value_type,
            "_to_lua_string": _to_lua_string,
            "_type_matches": type_matches,
            **expected_values,
        }
        exec(compile(tree, "<luapyre-ast-super-region>", "exec"), namespace)
        raw_runner: FunctionType = namespace["_jit_ast_loop"]

        # The fixed value is statistical only. Exact fuel is the dynamic `used`
        # count returned by the runner.
        return CompiledLoop(ir, max(1, len(ir.body.instructions) + 1), raw_runner)
