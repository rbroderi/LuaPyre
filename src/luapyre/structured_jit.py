from __future__ import annotations

import ast
from types import FunctionType

from .ast_backend import inline_type_guards, optimize_semantic_helpers
from .bytecode import Closure, Op
from .jit import CompiledLoop, IRBlock, IRInstruction, IRLoop
from .opdispatch import _float_divide
from .range_analysis import analyze_integer_ranges
from .table import LuaTable, _ABSENT, _hash_key
from .typed_ir import TypedIRCompiler
from .values import lua_equal, static_value_type, type_matches


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
            Op.GETUPVAL,
            Op.GETTABLE,
            Op.SETTABLE,
            Op.ADD,
            Op.ADD_I,
            Op.ADD_F,
            Op.SUB,
            Op.SUB_I,
            Op.SUB_F,
            Op.MUL,
            Op.MUL_I,
            Op.MUL_F,
            Op.DIV,
            Op.GUARD,
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
            Op.DIV,
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

    def _compile_structured_typed_diamond_loop(
        self, frame, start_pc: int, backedge_pc: int
    ):
        """Lower one forward diamond inside a numeric loop to a Python ``if``.

        The general CFG backend uses a state/match dispatcher. A very common
        typed shape is a single if/else whose arms immediately rejoin at the
        numeric backedge. Keeping that shape as Python control flow removes two
        state dispatches per Lua iteration while retaining exact path fuel.
        """
        proto = frame.proto
        loop_ins = proto.code[backedge_pc]
        if loop_ins.op not in (Op.FORLOOP, Op.JFORLOOP) or loop_ins.d != start_pc:
            return None
        body = proto.code[start_pc:backedge_pc]
        branches = [
            (start_pc + offset, ins)
            for offset, ins in enumerate(body)
            if ins.op in (Op.JMPIF, Op.JMPIFNOT)
        ]
        if len(branches) != 1:
            return None
        branch_pc, branch = branches[0]
        if not (branch_pc + 1 < branch.a < backedge_pc):
            return None
        first = list(enumerate(proto.code[branch_pc + 1:branch.a], branch_pc + 1))
        second = list(enumerate(proto.code[branch.a:backedge_pc], branch.a))
        if not first or first[-1][1].op is not Op.JMP or first[-1][1].a != backedge_pc:
            return None
        first_body = first[:-1]
        if any(ins.op in (Op.JMP, Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL, Op.CALL) for _, ins in (*first_body, *second)):
            return None
        prefix = list(enumerate(proto.code[start_pc:branch_pc], start_pc))
        allowed = {
            Op.LOADK, Op.MOVE, Op.LOCAL, Op.ADD_I, Op.SUB_I, Op.MUL_I,
            Op.ADD_F, Op.SUB_F, Op.MUL_F, Op.MOD, Op.EQ, Op.LT, Op.LE,
            Op.NOT, Op.TOBOOL, Op.GUARD,
        }
        if any(ins.op not in allowed for _, ins in (*prefix, *first_body, *second)):
            return None

        ranges = analyze_integer_ranges(proto)
        lowered = tuple(
            IRInstruction(pc, ins, overflow_free=ranges.overflow_free(pc))
            for pc, ins in enumerate(proto.code[start_pc:backedge_pc], start_pc)
        )
        ir = IRLoop(
            start_pc, backedge_pc, backedge_pc + 1,
            IRBlock(start_pc, backedge_pc, lowered), loop_ins,
        )
        registers = self._used_registers(ir)
        if self._captured_registers(proto).intersection(registers):
            return None
        lines = [
            "def _jit_structured_cfg_loop(vm, frame, budget):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
            "    used = 0",
        ]
        for reg in registers:
            lines.append(f"    _r{reg} = regs[{reg}]")
        lines.append(f"    _integer_loop = type(_r{loop_ins.a}) is int and type(_r{loop_ins.c}) is int")

        def spill(indent: str):
            lines.extend(self._spill_lines(registers, indent))

        max_cost = len(prefix) + 1 + max(len(first), len(second)) + 1

        def deopt(condition, pc, cost, indent):
            lines.append(f"{indent}if {condition}:")
            spill(indent + "    ")
            lines.extend([
                f"{indent}    frame.pc = {pc}",
                f"{indent}    return used + {cost}, used + {cost} > 0",
            ])

        def emit(sequence, indent: str, entry_types, start_cost=0):
            # Only facts established earlier on this very path are trusted.
            # No profile or previous iteration can prove a dynamic value type.
            known_types = dict(entry_types)
            numeric = {"integer", "integer_lua", "float", "number"}
            for offset, (pc, ins) in enumerate(sequence):
                cost = start_cost + offset
                a, b, c = f"_r{ins.a}", f"_r{ins.b}", f"_r{ins.c}"
                left, right = known_types.get(ins.b), known_types.get(ins.c)
                result_type = None
                if ins.op is Op.LOADK:
                    lines.append(f"{indent}{a} = consts[{ins.b}]")
                    result_type = static_value_type(proto.constants[ins.b]).name
                elif ins.op in (Op.MOVE, Op.LOCAL):
                    lines.append(f"{indent}{a} = {b}")
                    result_type = left
                elif ins.op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                    symbol = {Op.ADD_I: "+", Op.SUB_I: "-", Op.MUL_I: "*"}[ins.op]
                    expression = f"{b} {symbol} {c}"
                    if ranges.overflow_free(pc):
                        lines.append(f"{indent}{a} = {expression}")
                    else:
                        lines.extend(self._i64_lines(a, expression, f"cfg_{pc}", indent))
                    result_type = "integer"
                elif ins.op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                    symbol = {Op.ADD_F: "+", Op.SUB_F: "-", Op.MUL_F: "*"}[ins.op]
                    lines.append(f"{indent}{a} = float({b} {symbol} {c})")
                    result_type = "float"
                elif ins.op is Op.MOD:
                    checks = []
                    if left not in ("integer", "integer_lua"):
                        checks.append(f"type({b}) is not int")
                    if right not in ("integer", "integer_lua"):
                        checks.append(f"type({c}) is not int")
                    checks.append(f"{c} == 0")
                    deopt(" or ".join(checks), pc, cost, indent)
                    lines.append(f"{indent}{a} = {b} % {c}")
                    result_type = "integer"
                elif ins.op is Op.EQ:
                    if left in numeric and right in numeric:
                        lines.append(f"{indent}{a} = {b} == {c}")
                    elif left == right == "boolean":
                        lines.append(f"{indent}{a} = {b} is {c}")
                    else:
                        deopt(f"isinstance({b}, _LuaTable) and isinstance({c}, _LuaTable) and {b} is not {c} and ({b}.metatable is not None or {c}.metatable is not None)", pc, cost, indent)
                        lines.append(f"{indent}{a} = _lua_equal({b}, {c})")
                    result_type = "boolean"
                elif ins.op in (Op.LT, Op.LE):
                    if not (left in numeric and right in numeric or left == right == "string"):
                        deopt(f"not ((type({b}) in _NUM_TYPES and type({c}) in _NUM_TYPES) or (isinstance({b}, bytes) and isinstance({c}, bytes)))", pc, cost, indent)
                    symbol = "<" if ins.op is Op.LT else "<="
                    lines.append(f"{indent}{a} = {b} {symbol} {c}")
                    result_type = "boolean"
                elif ins.op is Op.NOT:
                    lines.append(f"{indent}{a} = ({b} is None or {b} is False)")
                    result_type = "boolean"
                elif ins.op is Op.TOBOOL:
                    lines.append(f"{indent}{a} = not ({b} is None or {b} is False)")
                    result_type = "boolean"
                elif ins.op is Op.GUARD:
                    expected = proto.constants[ins.b]
                    deopt(f"not _type_matches({expected!r}, {a})", pc, cost, indent)
                    result_type = expected
                else:
                    raise AssertionError(ins.op)
                known_types[ins.a] = result_type
            return known_types

        idx, limit, step = f"_r{loop_ins.a}", f"_r{loop_ins.b}", f"_r{loop_ins.c}"
        range_proven = all(
            ranges.range_at(start_pc, reg) is not None
            for reg in (loop_ins.a, loop_ins.b, loop_ins.c)
        ) and not any(
            ins.op is not Op.GUARD
            and ins.a in (loop_ins.a, loop_ins.b, loop_ins.c)
            for _pc, ins in (*prefix, *first_body, *second)
        )
        entry_types = (
            {
                loop_ins.a: "integer",
                loop_ins.b: "integer",
                loop_ins.c: "integer",
            }
            if range_proven
            else {}
        )

        def emit_body(indent: str):
            prefix_types = emit(prefix, indent, entry_types)
            condition = f"not (_r{branch.b} is None or _r{branch.b} is False)"
            if branch.op is Op.JMPIFNOT:
                condition = f"not ({condition})"
            lines.append(f"{indent}if {condition}:")
            emit(second, indent + "    ", prefix_types, len(prefix) + 1)
            lines.append(
                f"{indent}    used += {len(prefix) + 1 + len(second) + 1}"
            )
            lines.append(f"{indent}else:")
            emit(first_body, indent + "    ", prefix_types, len(prefix) + 1)
            # Charge the executed prefix, branch, arm, optional JMP, and
            # backedge once per path. Side exits include the exact partial cost.
            lines.append(
                f"{indent}    used += {len(prefix) + 1 + len(first) + 1}"
            )

        if range_proven:
            lines.extend(
                [
                    f"    _total = (({limit} - {idx}) // {step} + 1) if {step} > 0 else (({idx} - {limit}) // -{step} + 1)",
                    f"    if _integer_loop and _total > 0 and budget >= _total * {max_cost}:",
                    f"        for _loop_value in range({idx}, {idx} + _total * {step}, {step}):",
                    f"            {idx} = _loop_value",
                ]
            )
            emit_body("            ")
            spill("        ")
            lines.extend(
                [
                    f"        frame.pc = {backedge_pc + 1}",
                    "        return used, True",
                ]
            )

        lines.append(f"    while budget - used >= {max_cost}:")
        emit_body("        ")
        lines.extend([
            f"        _next = {idx} + {step}",
            f"        if (_integer_loop and (_next < _INT_MIN or _next > _INT_MAX)) or ({step} > 0 and _next > {limit}) or ({step} < 0 and _next < {limit}):",
        ])
        spill("            ")
        lines.extend([
            f"            frame.pc = {backedge_pc + 1}",
            "            return used, True",
            f"        {idx} = _next",
        ])
        spill("    ")
        lines.extend([f"    frame.pc = {start_pc}", "    return used, used > 0"])
        tree = optimize_semantic_helpers(ast.parse("\n".join(lines)))
        ast.fix_missing_locations(tree)
        namespace = {
            "_INT_MIN": _INT_MIN, "_INT_MAX": _INT_MAX,
            "_MASK64": _MASK64, "_SIGN64": _SIGN64, "_TWO64": _TWO64,
            "_NUM_TYPES": (int, float), "_LuaTable": LuaTable,
            "_type_matches": type_matches, "_lua_equal": lua_equal,
        }
        exec(compile(tree, "<luapyre-structured-cfg-loop>", "exec"), namespace)
        return CompiledLoop(ir, max_cost, namespace["_jit_structured_cfg_loop"])

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
        if any(ins.op is Op.DIV for _, ins in sequence):
            plan = TypedIRCompiler(proto).compile((tuple(sequence),))
            numeric = ("integer", "integer_lua", "float", "number")
            for pc, ins in sequence:
                if ins.op is Op.DIV:
                    site = plan.instruction(pc)
                    if (site.value_for(ins.b).type_name not in numeric
                            or site.value_for(ins.c).type_name not in numeric):
                        return None
        return tuple(sequence)

    @staticmethod
    def _writes_register(ins, reg: int) -> bool:
        if ins.op in (
            Op.LOADK,
            Op.MOVE,
            Op.LOCAL,
            Op.GETUPVAL,
            Op.ADD,
            Op.ADD_I,
            Op.ADD_F,
            Op.SUB_I,
            Op.SUB_F,
            Op.MUL_I,
            Op.MUL_F,
            Op.SUB,
            Op.MUL,
            Op.DIV,
            Op.NOT,
            Op.TOBOOL,
            Op.GETTABLE,
        ):
            return ins.a == reg
        if ins.op is Op.CALL and ins.e > 0:
            return ins.a <= reg < ins.a + ins.e
        return False

    def _compile_structured_typed_loop(self, frame, start_pc: int, backedge_pc: int):
        proto = frame.proto
        body = proto.code[start_pc:backedge_pc]
        if not body or any(ins.op not in self._STRUCTURED_OPS for ins in body):
            return self._compile_structured_typed_diamond_loop(
                frame, start_pc, backedge_pc
            )
        loop_ins = proto.code[backedge_pc]
        if loop_ins.op not in (Op.FORLOOP, Op.JFORLOOP) or loop_ins.d != start_pc:
            return None

        captured = self._captured_registers(proto)
        ranges = analyze_integer_ranges(proto)
        lowered = tuple(
            IRInstruction(
                start_pc + i,
                ins,
                overflow_free=ranges.overflow_free(start_pc + i),
            )
            for i, ins in enumerate(body)
        )
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
        call_upvalues: dict[int, int] = {}
        child_ranges = {}
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
            producers = [
                other for other in body
                if self._writes_register(other, ins.b)
            ]
            if any(other.op is not Op.GETUPVAL for other in producers):
                return None
            if producers:
                upvalue_indexes = {other.b for other in producers}
                if len(upvalue_indexes) != 1:
                    return None
                call_upvalues[pc] = next(iter(upvalue_indexes))
            calls[pc] = (fn, sequence)
            child_ranges[pc] = analyze_integer_ranges(fn.proto)
            extra_cost += len(sequence)

        iteration_cost = len(body) + 1 + extra_cost
        lines = [
            "def _jit_structured_loop(vm, frame, budget):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
            "    upvalues = frame.closure.upvalues",
            "    used = 0",
        ]
        for reg in registers:
            lines.append(f"    _r{reg} = regs[{reg}]")

        namespace: dict[str, object] = {
            "_Closure": Closure,
            "_LuaTable": LuaTable,
            "_float_divide": _float_divide,
            "_MASK64": _MASK64,
            "_SIGN64": _SIGN64,
            "_TWO64": _TWO64,
            "_INT_MIN": _INT_MIN,
            "_INT_MAX": _INT_MAX,
            "_type_matches": type_matches,
            "_ABSENT": _ABSENT,
        }
        table_arrays: dict[int, str] = {}
        for pc, ins in zip(range(start_pc, backedge_pc), body):
            if ins.op not in (Op.GETTABLE, Op.SETTABLE):
                continue
            table_reg = ins.b if ins.op is Op.GETTABLE else ins.a
            if any(self._writes_register(other, table_reg) for other in body):
                return None
            if table_reg in table_arrays:
                continue
            array_name = f"_array_r{table_reg}"
            table_arrays[table_reg] = array_name
            lines.append(
                f"    if not isinstance(_r{table_reg}, _LuaTable) or _r{table_reg}.metatable is not None:"
            )
            lines.extend(self._spill_lines(registers, "        "))
            lines.extend(
                [
                    f"        frame.pc = {start_pc}",
                    "        return 0, False",
                    f"    {array_name} = _r{table_reg}.array",
                ]
            )
        for pc, (closure, _sequence) in calls.items():
            expected = f"_expected_proto_{pc}"
            # A local function declaration creates a fresh Closure every time
            # the top-level Proto runs.  For the leaf shapes admitted here there
            # are no upvalues/children, so the executable semantics are fully
            # determined by the immutable Proto.  Guarding the Proto rather than
            # the transient Closure keeps the compiled loop valid across runs.
            namespace[expected] = closure.proto
            function_reg = body[pc - start_pc].b
            if pc in call_upvalues:
                lines.append(
                    f"    _r{function_reg} = upvalues[{call_upvalues[pc]}].value"
                )
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
            namespace[f"_consts_{pc}"] = closure.proto.constants

        if calls:
            lines.append(
                "    if vm._active_frames is None or len(vm._active_frames) >= vm.max_frames:"
            )
            lines.extend(self._spill_lines(registers, "        "))
            lines.extend(
                [
                    f"        frame.pc = {start_pc}",
                    "        return 0, False",
                ]
            )

        # Plain table reads cannot re-enter Lua. Without writes or calls, their
        # constant fields remain stable for this invocation of the loop runner.
        # Refresh on every entry; aliases and changes between runs stay visible.
        invariant_reads = not any(ins.op in (Op.SETTABLE, Op.CALL) for ins in body)
        hoisted_reads: list[str] = []
        loop_entry = len(lines)
        integer_loop = all(
            ranges.range_at(start_pc, reg) is not None
            for reg in (loop_ins.a, loop_ins.b, loop_ins.c)
        )
        idx, limit, step = (
            f"_r{loop_ins.a}",
            f"_r{loop_ins.b}",
            f"_r{loop_ins.c}",
        )
        batched_can_exit = integer_loop and any(
            ins.op in (Op.GUARD, Op.CALL, Op.SETTABLE, Op.DIV) for ins in body
        )
        if integer_loop:
            lines.extend(
                [
                    f"    _total = (({limit} - {idx}) // {step} + 1) if {step} > 0 else (({idx} - {limit}) // -{step} + 1)",
                    f"    _take = min(_total, budget // {iteration_cost})",
                ]
            )
            if batched_can_exit:
                lines.append("    _completed = 0")
            lines.extend(
                [
                    f"    for _loop_value in range({idx}, {idx} + _take * {step}, {step}):",
                    f"        {idx} = _loop_value",
                ]
            )
            indent = "        "
        else:
            lines.append(f"    while budget - used >= {iteration_cost}:")
            indent = "        "

        prior_child_cost = 0

        def guard(condition, pc, cost):
            lines.append(f"{indent}if {condition}:")
            lines.extend(self._spill_lines(registers, indent + "    "))
            completed_cost = f"_completed * {iteration_cost}" if integer_loop else "used"
            lines.extend([
                f"{indent}    frame.pc = {pc}",
                f"{indent}    return {completed_cost} + {cost}, False",
            ])

        typed_plan = TypedIRCompiler(proto).compile((tuple(enumerate(proto.code)),))
        known_constants: dict[int, object] = {}
        known_types: dict[int, str] = {}
        numeric_types = {"integer", "integer_lua", "float"}
        for offset, ins in enumerate(body):
            pc = start_pc + offset
            cost = offset + prior_child_cost
            a, b, c = f"_r{ins.a}", f"_r{ins.b}", f"_r{ins.c}"
            if self._writes_register(ins, ins.a) and ins.op not in (
                Op.LOADK, Op.MOVE, Op.LOCAL, Op.CALL
            ):
                known_constants.pop(ins.a, None)
            if ins.op is Op.LOADK:
                lines.append(f"{indent}{a} = consts[{ins.b}]")
                known_constants[ins.a] = proto.constants[ins.b]
                known_types[ins.a] = static_value_type(proto.constants[ins.b]).name
            elif ins.op in (Op.MOVE, Op.LOCAL):
                lines.append(f"{indent}{a} = {b}")
                if ins.b in known_constants:
                    known_constants[ins.a] = known_constants[ins.b]
                else:
                    known_constants.pop(ins.a, None)
                source_type = known_types.get(ins.b)
                if source_type is not None:
                    known_types[ins.a] = source_type
                else:
                    known_types.pop(ins.a, None)
            elif ins.op is Op.GETUPVAL:
                lines.append(f"{indent}{a} = upvalues[{ins.b}].value")
                known_constants.pop(ins.a, None)
                known_types.pop(ins.a, None)
            elif ins.op is Op.GETTABLE:
                array_name = table_arrays[ins.b]
                key = known_constants.get(ins.c, _ABSENT)
                if type(key) is float and key.is_integer():
                    key = int(key)
                if key is not _ABSENT:
                    read_start = len(lines)
                    target = a
                    if invariant_reads:
                        a = f"_field_{pc}"
                    token_name = f"_key_token_{pc}"
                    namespace[token_name] = _hash_key(key)
                    if type(key) is int and key >= 1:
                        lines.append(f"{indent}if {key} <= len({array_name}):")
                        lines.append(f"{indent}    {a} = {array_name}[{key - 1}]")
                        lines.append(f"{indent}else:")
                        lines.append(f"{indent}    _item_{pc} = {b}.hash.get({token_name}, _ABSENT)")
                        lines.append(f"{indent}    {a} = None if _item_{pc} is _ABSENT else _item_{pc}[1]")
                    else:
                        lines.append(f"{indent}_item_{pc} = {b}.hash.get({token_name}, _ABSENT)")
                        lines.append(f"{indent}{a} = None if _item_{pc} is _ABSENT else _item_{pc}[1]")
                    if invariant_reads:
                        hoisted_reads.extend("    " + line[len(indent):] for line in lines[read_start:])
                        del lines[read_start:]
                        lines.append(f"{indent}{target} = {a}")
                    a = target
                else:
                    lines.append(f"{indent}_key_{pc} = {c}")
                    lines.append(
                        f"{indent}if type(_key_{pc}) is int and 1 <= _key_{pc} <= len({array_name}):"
                    )
                    lines.append(f"{indent}    {a} = {array_name}[_key_{pc} - 1]")
                    lines.append(f"{indent}else:")
                    lines.append(f"{indent}    {a} = {b}.rawget(_key_{pc})")
                known_constants.pop(ins.a, None)
                known_types.pop(ins.a, None)
            elif ins.op is Op.SETTABLE:
                guard(f"{b} is None or (type({b}) is float and {b} != {b})", pc, cost)
                lines.append(f"{indent}{a}.rawset({b}, {c})")
            elif ins.op is Op.GUARD:
                guard(f"not _type_matches({proto.constants[ins.b]!r}, {a})", pc, cost)
                known_types[ins.a] = str(proto.constants[ins.b])
            elif ins.op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                symbol = {Op.ADD_I: "+", Op.SUB_I: "-", Op.MUL_I: "*"}[ins.op]
                expression = f"{b} {symbol} {c}"
                if ranges.overflow_free(pc):
                    lines.append(f"{indent}{a} = {expression}")
                else:
                    lines.extend(self._i64_lines(a, expression, str(pc), indent))
                known_types[ins.a] = "integer"
            elif ins.op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                symbol = {Op.ADD_F: "+", Op.SUB_F: "-", Op.MUL_F: "*"}[ins.op]
                lines.append(f"{indent}{a} = float({b} {symbol} {c})")
                known_types[ins.a] = "float"
            elif ins.op in (Op.ADD, Op.SUB, Op.MUL):
                site = typed_plan.instruction(pc)
                left_type = known_types.get(ins.b, site.value_for(ins.b).type_name)
                right_type = known_types.get(ins.c, site.value_for(ins.c).type_name)
                if left_type not in numeric_types or right_type not in numeric_types:
                    return None
                symbol = {Op.ADD: "+", Op.SUB: "-", Op.MUL: "*"}[ins.op]
                expression = f"{b} {symbol} {c}"
                if left_type in ("integer", "integer_lua") and right_type in ("integer", "integer_lua"):
                    if ranges.overflow_free(pc):
                        lines.append(f"{indent}{a} = {expression}")
                    else:
                        lines.extend(self._i64_lines(a, expression, str(pc), indent))
                    known_types[ins.a] = "integer"
                else:
                    lines.append(f"{indent}{a} = float({expression})")
                    known_types[ins.a] = "float"
            elif ins.op is Op.DIV:
                guard(f"type({b}) not in (int, float) or type({c}) not in (int, float)", pc, cost)
                lines.append(f"{indent}{a} = _float_divide({b}, {c})")
                known_types[ins.a] = "float"
            elif ins.op is Op.NOT:
                lines.append(f"{indent}{a} = ({b} is None or {b} is False)")
                known_types[ins.a] = "boolean"
            elif ins.op is Op.TOBOOL:
                lines.append(f"{indent}{a} = not ({b} is None or {b} is False)")
                known_types[ins.a] = "boolean"
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
                    expected_type = closure.proto.param_types[arg_index].name
                    actual_type = known_types.get(ins.c + arg_index)
                    accepts = (
                        actual_type == expected_type
                        or expected_type == "number" and actual_type in numeric_types
                        or expected_type in ("integer", "integer_lua")
                        and actual_type in ("integer", "integer_lua")
                    )
                    if not accepts:
                        guard(f"not _type_matches({expected_type!r}, {value})", pc, cost)
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
                        expression = f"{cb} {symbol} {cc}"
                        if child_ranges[pc].overflow_free(child_pc):
                            lines.append(f"{indent}{ca} = {expression}")
                        else:
                            lines.extend(
                                self._i64_lines(
                                    ca,
                                    expression,
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
                    elif child_ins.op is Op.DIV:
                        lines.append(f"{indent}{ca} = _float_divide({cb}, {cc})")
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
                        known_constants.pop(ins.a + result_index, None)
                        if result_index < len(closure.proto.return_types):
                            known_types[ins.a + result_index] = closure.proto.return_types[result_index].name
                        else:
                            known_types.pop(ins.a + result_index, None)
                prior_child_cost += len(sequence)
            else:
                return None

        if batched_can_exit:
            lines.append(f"{indent}_completed += 1")

        if integer_loop:
            completed = "_completed" if batched_can_exit else "_take"
            lines.append(f"    used = {completed} * {iteration_cost}")
            lines.append("    if _take == _total and _take:")
            lines.extend(self._spill_lines(registers, "        "))
            lines.extend(
                [
                    f"        frame.pc = {ir.exit_pc}",
                    "        return used, True",
                    "    if _take:",
                    f"        {idx} += {step}",
                ]
            )
        else:
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

        lines[loop_entry:loop_entry] = hoisted_reads
        tree = ast.parse("\n".join(lines))
        tree = inline_type_guards(tree)
        ast.fix_missing_locations(tree)
        exec(compile(tree, "<luapyre-structured-loop>", "exec"), namespace)
        runner: FunctionType = namespace["_jit_structured_loop"]
        return CompiledLoop(ir, iteration_cost, runner)
