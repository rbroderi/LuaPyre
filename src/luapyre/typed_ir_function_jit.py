from __future__ import annotations

import ast
import math

from .ast_backend import JumpListLayout, inline_expression_helper, inline_local_jump_list
from .bytecode import Op, Proto
from .errors import LuaRuntimeError
from .function_jit import (
    CompiledAstFunction,
    _FUNCTION_CONTROL,
    _FUNC_RETURN,
    _FUNC_SUSPEND,
)
from .opdispatch import _float_divide, _float_modulo
from .table import LuaTable, _ABSENT, _hash_key
from .typed_ir import IRValue, IRValueKind, TypedIRCompiler, TypedIRPlan
from .typed_ir_passes import eliminate_rematerializable_dead_defs
from .values import lua_equal, static_value_type, type_matches


_INT_MIN = -(1 << 63)
_INT_MAX = (1 << 63) - 1
_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63
_TWO64 = 1 << 64
_NUM_TYPES = (int, float)


class TypedIRFunctionJITMixin:
    """Whole-function backend for the small statically typed optimizer IR.

    This backend intentionally handles only functions whose dynamic call edges do
    not need the existing recursive/direct-call machinery. Unsupported shapes
    fall through to ``TypedFunctionJITMixin``. That keeps the semantic frame and
    fuel model unchanged while moving table/global optimization, value
    propagation, DSE, and deopt rematerialization into the shared IR.
    """

    @staticmethod
    def _function_ir_expr(value: IRValue) -> str:
        if value.kind is IRValueKind.CONSTANT:
            return f"consts[{value.index}]"
        if value.kind is IRValueKind.UPVALUE:
            return f"upvalues[{value.index}].value"
        return f"_r{value.index}"

    @staticmethod
    def _function_constant_array_index(value: object) -> int | None:
        if type(value) is int and value >= 1:
            return value
        if type(value) is float and value >= 1 and value.is_integer():
            return int(value)
        return None

    def _compile_ast_function(self, proto: Proto) -> CompiledAstFunction | None:
        blocks = self._function_blocks(proto)
        if blocks is None or proto.children:
            return super()._compile_ast_function(proto)

        # Direct/recursive calls already have exact shared-meter semantics in the
        # proven 0.15 function backend. Keep those there until CALL is represented
        # explicitly in the typed IR rather than duplicating call semantics here.
        if any(ins.op is Op.CALL for ins in proto.code):
            return super()._compile_ast_function(proto)

        block_code = tuple(block.instructions for block in blocks)
        plan = eliminate_rematerializable_dead_defs(
            TypedIRCompiler(proto).compile(block_code), block_code
        )
        interesting = any(
            item.dead_definition
            or item.specialization
            in ("global_get", "global_set", "table_get_const", "table_set_const")
            for item in plan.instructions
        )
        if not interesting:
            return super()._compile_ast_function(proto)

        registers = tuple(range(max(1, proto.register_count)))
        block_index = {block.start: index for index, block in enumerate(blocks)}

        def state_for(pc: int) -> int:
            if pc == len(proto.code):
                return _FUNC_RETURN
            return block_index[pc]

        def spill(pc: int, indent: str) -> list[str]:
            state = plan.virtual_state(pc)
            out: list[str] = []
            for reg in registers:
                value = state.get(reg)
                expr = (
                    self._function_ir_expr(value)
                    if value is not None
                    else f"_r{reg}"
                )
                out.append(f"{indent}regs[{reg}] = {expr}")
            return out

        def suspend(pc: int, indent: str) -> list[str]:
            return [
                *spill(pc, indent),
                f"{indent}frame.pc = {pc}",
                f"{indent}return _FUNC_SUSPEND",
            ]

        def deopt(out: list[str], condition: str, pc: int, indent: str) -> None:
            out.append(f"{indent}if {condition}:")
            out.extend(
                f"{indent}    {line.strip()}" for line in suspend(pc, "")
            )

        def i64(dest: str, expression: str, tag: str, indent: str) -> list[str]:
            tmp = f"_i64_{tag}"
            return [
                f"{indent}{tmp} = ({expression}) & _MASK64",
                f"{indent}{dest} = {tmp} - _TWO64 if {tmp} & _SIGN64 else {tmp}",
            ]

        def emit_forprep(
            out: list[str], ins, true_state: int, false_state: int, indent: str
        ) -> None:
            a, b, c = f"_r{ins.a}", f"_r{ins.b}", f"_r{ins.c}"
            out.extend(
                [
                    f"{indent}if not (type({a}) in _NUM_TYPES and type({b}) in _NUM_TYPES and type({c}) in _NUM_TYPES):",
                    f"{indent}    raise _LuaRuntimeError(\"'for' limit must be a number\")",
                    f"{indent}if {c} == 0:",
                    f"{indent}    raise _LuaRuntimeError(\"'for' step is zero\")",
                    f"{indent}if type({a}) is float or type({b}) is float or type({c}) is float:",
                    f"{indent}    {a} = float({a})",
                    f"{indent}    {b} = float({b})",
                    f"{indent}    {c} = float({c})",
                    f"{indent}return {true_state} if ({a} <= {b} if {c} > 0 else {a} >= {b}) else {false_state}",
                ]
            )

        def emit_forloop(
            out: list[str], ins, loop_state: int, exit_state: int, pc: int, indent: str
        ) -> None:
            a, b, c = f"_r{ins.a}", f"_r{ins.b}", f"_r{ins.c}"
            tag = f"f_{pc}_{ins.a}"
            out.extend(
                [
                    f"{indent}if type({a}) is int and type({b}) is int and type({c}) is int:",
                    f"{indent}    _next_{tag} = {a} + {c}",
                    f"{indent}    if _next_{tag} < _INT_MIN or _next_{tag} > _INT_MAX or ({c} > 0 and _next_{tag} > {b}) or ({c} < 0 and _next_{tag} < {b}):",
                    f"{indent}        return {exit_state}",
                    f"{indent}    {a} = _next_{tag}",
                    f"{indent}    return {loop_state}",
                    f"{indent}_next_{tag} = float({a}) + float({c})",
                    f"{indent}if ({c} > 0 and _next_{tag} > {b}) or ({c} < 0 and _next_{tag} < {b}):",
                    f"{indent}    return {exit_state}",
                    f"{indent}{a} = _next_{tag}",
                    f"{indent}return {loop_state}",
                ]
            )

        namespace: dict[str, object] = {
            "_ABSENT": _ABSENT,
            "_FUNC_RETURN": _FUNC_RETURN,
            "_FUNC_SUSPEND": _FUNC_SUSPEND,
            "_INT_MIN": _INT_MIN,
            "_INT_MAX": _INT_MAX,
            "_MASK64": _MASK64,
            "_SIGN64": _SIGN64,
            "_TWO64": _TWO64,
            "_NUM_TYPES": _NUM_TYPES,
            "_LuaRuntimeError": LuaRuntimeError,
            "_LuaTable": LuaTable,
            "_float_divide": _float_divide,
            "_float_modulo": _float_modulo,
            "_isnan": math.isnan,
            "_lua_equal": lua_equal,
            "_static_value_type": static_value_type,
            "_type_matches": type_matches,
        }

        for item in plan.instructions:
            if item.specialization not in (
                "global_get",
                "global_set",
                "table_get_const",
                "table_set_const",
            ):
                continue
            ins = item.ins
            key_reg = ins.c if ins.op is Op.GETTABLE else ins.b
            key_value = item.value_for(key_reg)
            if key_value.kind is IRValueKind.CONSTANT:
                namespace[f"_key_token_{item.pc}"] = _hash_key(
                    proto.constants[key_value.index]
                )

        lines = [
            "def _jit_ir_function(vm, frames, frame, budget, meter):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
            "    upvalues = frame.closure.upvalues",
            "    used = 0",
            "    _result = ()",
        ]
        for reg in registers:
            lines.append(f"    _r{reg} = regs[{reg}]")

        # The IR may prove a root _ENV read invariant across the whole compiled
        # function. Hoist only raw-table reads. A metatable/dynamic receiver
        # fails closed before any Lua instruction executes, so Tier 0 resumes at
        # pc 0 without duplicate side effects or fuel consumption.
        for pc in plan.invariant_sites:
            site = plan.instruction(pc)
            if site is None:
                return super()._compile_ast_function(proto)
            ins = site.ins
            table_value = site.value_for(ins.b)
            key_value = site.value_for(ins.c)
            if key_value.kind is not IRValueKind.CONSTANT:
                return super()._compile_ast_function(proto)
            table_expr = self._function_ir_expr(table_value)
            key = proto.constants[key_value.index]
            array_index = self._function_constant_array_index(key)
            table_tmp = f"_licm_{pc}_table"
            item_tmp = f"_licm_{pc}_item"
            token = f"_key_token_{pc}"
            lines.extend(
                [
                    f"    {table_tmp} = {table_expr}",
                    f"    if not isinstance({table_tmp}, _LuaTable) or {table_tmp}.metatable is not None:",
                    "        frame.pc = 0",
                    "        return _FUNC_SUSPEND, None",
                ]
            )
            if array_index is not None:
                lines.extend(
                    [
                        f"    if {array_index} <= len({table_tmp}.array):",
                        f"        _licm_{pc}_value = {table_tmp}.array[{array_index - 1}]",
                        "    else:",
                        f"        {item_tmp} = {table_tmp}.hash.get({token}, _ABSENT)",
                        f"        _licm_{pc}_value = None if {item_tmp} is _ABSENT else {item_tmp}[1]",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"    {item_tmp} = {table_tmp}.hash.get({token}, _ABSENT)",
                        f"    _licm_{pc}_value = None if {item_tmp} is _ABSENT else {item_tmp}[1]",
                    ]
                )

        for pc in plan.cache_sites:
            lines.extend(
                [
                    f"    _ic_{pc}_table = None",
                    f"    _ic_{pc}_version = -1",
                    f"    _ic_{pc}_value = None",
                ]
            )

        block_names = tuple(f"_jump_{index}" for index in range(len(blocks)))
        for block_no, block in enumerate(blocks):
            lines.append(f"    def {block_names[block_no]}():")
            indent = "        "
            cost = len(block.instructions)
            lines.append(f"{indent}if budget - meter[0] - used < {cost}:")
            lines.extend(
                f"{indent}    {line.strip()}"
                for line in suspend(block.start, "")
            )

            terminal_pc, terminal_ins = block.instructions[-1]
            terminal_control = terminal_ins.op in _FUNCTION_CONTROL or terminal_ins.op in (
                Op.RETURN,
                Op.HALT,
            )
            ordinary = block.instructions[:-1] if terminal_control else block.instructions

            for pc, ins in ordinary:
                site = plan.instruction(pc)
                if site is None:
                    return super()._compile_ast_function(proto)
                op = ins.op
                a, b, c = f"_r{ins.a}", f"_r{ins.b}", f"_r{ins.c}"

                if site.dead_definition:
                    lines.append(f"{indent}used += 1")
                    continue

                if op is Op.LOADK:
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = consts[{ins.b}]"])
                elif op in (Op.MOVE, Op.LOCAL):
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = {b}"])
                elif op is Op.GETUPVAL:
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = upvalues[{ins.b}].value"])
                elif op is Op.SETUPVAL:
                    lines.extend([f"{indent}used += 1", f"{indent}upvalues[{ins.a}].value = {b}"])
                elif op is Op.NEWTABLE:
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = _LuaTable()"])
                elif op is Op.GETTABLE and site.specialization in (
                    "global_get",
                    "table_get_const",
                ):
                    if pc in plan.invariant_sites:
                        lines.extend(
                            [
                                f"{indent}used += 1",
                                f"{indent}{a} = _licm_{pc}_value",
                            ]
                        )
                        continue
                    table_value = site.value_for(ins.b)
                    key_value = site.value_for(ins.c)
                    if key_value.kind is not IRValueKind.CONSTANT:
                        return super()._compile_ast_function(proto)
                    table_expr = self._function_ir_expr(table_value)
                    key = proto.constants[key_value.index]
                    array_index = self._function_constant_array_index(key)
                    table_tmp = f"_ir_table_{pc}"
                    item_tmp = f"_ir_item_{pc}"
                    cache_table = f"_ic_{pc}_table"
                    cache_version = f"_ic_{pc}_version"
                    cache_value = f"_ic_{pc}_value"
                    token = f"_key_token_{pc}"
                    lines.append(f"{indent}{table_tmp} = {table_expr}")
                    deopt(
                        lines,
                        f"not isinstance({table_tmp}, _LuaTable) or {table_tmp}.metatable is not None",
                        pc,
                        indent,
                    )
                    lines.extend(
                        [
                            f"{indent}used += 1",
                            f"{indent}if {table_tmp} is {cache_table} and {table_tmp}.version == {cache_version}:",
                            f"{indent}    {a} = {cache_value}",
                            f"{indent}else:",
                        ]
                    )
                    if array_index is not None:
                        lines.extend(
                            [
                                f"{indent}    if {array_index} <= len({table_tmp}.array):",
                                f"{indent}        {a} = {table_tmp}.array[{array_index - 1}]",
                                f"{indent}    else:",
                                f"{indent}        {item_tmp} = {table_tmp}.hash.get({token}, _ABSENT)",
                                f"{indent}        {a} = None if {item_tmp} is _ABSENT else {item_tmp}[1]",
                            ]
                        )
                    else:
                        lines.extend(
                            [
                                f"{indent}    {item_tmp} = {table_tmp}.hash.get({token}, _ABSENT)",
                                f"{indent}    {a} = None if {item_tmp} is _ABSENT else {item_tmp}[1]",
                            ]
                        )
                    lines.extend(
                        [
                            f"{indent}    {cache_table} = {table_tmp}",
                            f"{indent}    {cache_version} = {table_tmp}.version",
                            f"{indent}    {cache_value} = {a}",
                        ]
                    )
                elif op is Op.SETTABLE and site.specialization in (
                    "global_set",
                    "table_set_const",
                ):
                    table_value = site.value_for(ins.a)
                    key_value = site.value_for(ins.b)
                    if key_value.kind is not IRValueKind.CONSTANT:
                        return super()._compile_ast_function(proto)
                    table_expr = self._function_ir_expr(table_value)
                    key_expr = self._function_ir_expr(key_value)
                    key = proto.constants[key_value.index]
                    array_index = self._function_constant_array_index(key)
                    table_tmp = f"_ir_table_{pc}"
                    token = f"_key_token_{pc}"
                    lines.append(f"{indent}{table_tmp} = {table_expr}")
                    deopt(
                        lines,
                        f"not isinstance({table_tmp}, _LuaTable) or {table_tmp}.metatable is not None",
                        pc,
                        indent,
                    )
                    lines.append(f"{indent}used += 1")
                    if array_index is None:
                        lines.extend(
                            [
                                f"{indent}{table_tmp}.version += 1",
                                f"{indent}if {c} is None:",
                                f"{indent}    {table_tmp}.hash.pop({token}, None)",
                                f"{indent}else:",
                                f"{indent}    {table_tmp}.hash[{token}] = ({key_expr}, {c})",
                            ]
                        )
                    else:
                        lines.append(f"{indent}{table_tmp}.rawset({key_expr}, {c})")
                elif op is Op.GETTABLE:
                    deopt(
                        lines,
                        f"not isinstance({b}, _LuaTable) or {b}.metatable is not None",
                        pc,
                        indent,
                    )
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = {b}.rawget({c})"])
                elif op is Op.SETTABLE:
                    deopt(
                        lines,
                        f"not isinstance({a}, _LuaTable) or {a}.metatable is not None",
                        pc,
                        indent,
                    )
                    deopt(
                        lines,
                        f"{b} is None or (type({b}) is float and _isnan({b}))",
                        pc,
                        indent,
                    )
                    lines.extend([f"{indent}used += 1", f"{indent}{a}.rawset({b}, {c})"])
                elif op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                    symbol = {Op.ADD_I: "+", Op.SUB_I: "-", Op.MUL_I: "*"}[op]
                    lines.append(f"{indent}used += 1")
                    lines.extend(i64(a, f"{b} {symbol} {c}", str(pc), indent))
                elif op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                    symbol = {Op.ADD_F: "+", Op.SUB_F: "-", Op.MUL_F: "*"}[op]
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = float({b} {symbol} {c})"])
                elif op in (Op.ADD, Op.SUB, Op.MUL):
                    symbol = {Op.ADD: "+", Op.SUB: "-", Op.MUL: "*"}[op]
                    deopt(
                        lines,
                        f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES",
                        pc,
                        indent,
                    )
                    tmp = f"_arith_{pc}"
                    lines.extend([f"{indent}used += 1", f"{indent}{tmp} = {b} {symbol} {c}"])
                    lines.append(f"{indent}if type({b}) is int and type({c}) is int:")
                    lines.extend(i64(a, tmp, f"g_{pc}", indent + "    "))
                    lines.append(f"{indent}else:")
                    lines.append(f"{indent}    {a} = {tmp}")
                elif op is Op.DIV:
                    deopt(
                        lines,
                        f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES",
                        pc,
                        indent,
                    )
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = _float_divide({b}, {c})"])
                elif op is Op.MOD:
                    deopt(
                        lines,
                        f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES",
                        pc,
                        indent,
                    )
                    lines.append(f"{indent}used += 1")
                    lines.append(f"{indent}if type({b}) is int and type({c}) is int:")
                    lines.append(f"{indent}    if {c} == 0:")
                    lines.append(f"{indent}        raise _LuaRuntimeError(\"attempt to perform 'n%0'\")")
                    lines.extend(i64(a, f"{b} % {c}", f"m_{pc}", indent + "    "))
                    lines.append(f"{indent}else:")
                    lines.append(f"{indent}    {a} = _float_modulo({b}, {c})")
                elif op is Op.NOT:
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = not _truthy({b})"])
                elif op is Op.TOBOOL:
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = _truthy({b})"])
                elif op is Op.EQ:
                    deopt(
                        lines,
                        f"isinstance({b}, _LuaTable) and isinstance({c}, _LuaTable) and not _lua_equal({b}, {c}) and ({b}.metatable is not None or {c}.metatable is not None)",
                        pc,
                        indent,
                    )
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = _lua_equal({b}, {c})"])
                elif op in (Op.LT, Op.LE):
                    symbol = "<" if op is Op.LT else "<="
                    deopt(
                        lines,
                        f"not ((type({b}) in _NUM_TYPES and type({c}) in _NUM_TYPES) or (isinstance({b}, bytes) and isinstance({c}, bytes)))",
                        pc,
                        indent,
                    )
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = {b} {symbol} {c}"])
                elif op is Op.GUARD:
                    lines.append(f"{indent}used += 1")
                    lines.append(f"{indent}if not _type_matches(consts[{ins.b}], {a}):")
                    lines.append(
                        f"{indent}    raise _LuaRuntimeError(f\"expected {{consts[{ins.b}]!s}}, got {{_static_value_type({a}).name}}\")"
                    )
                else:
                    return super()._compile_ast_function(proto)

            if terminal_control:
                ins = terminal_ins
                pc = terminal_pc
                lines.append(f"{indent}used += 1")
                if ins.op is Op.JMP:
                    lines.append(f"{indent}return {state_for(ins.a)}")
                elif ins.op in (Op.JMPIF, Op.JMPIFNOT):
                    condition = f"_truthy(_r{ins.b})"
                    if ins.op is Op.JMPIFNOT:
                        condition = f"not ({condition})"
                    lines.append(
                        f"{indent}return {state_for(ins.a)} if {condition} else {state_for(block.end)}"
                    )
                elif ins.op is Op.JMPIFNIL:
                    lines.append(
                        f"{indent}return {state_for(ins.a)} if _r{ins.b} is None else {state_for(block.end)}"
                    )
                elif ins.op is Op.FORPREP:
                    emit_forprep(lines, ins, state_for(block.end), state_for(ins.d), indent)
                elif ins.op in (Op.FORLOOP, Op.JFORLOOP):
                    emit_forloop(
                        lines,
                        ins,
                        state_for(ins.d),
                        state_for(block.end),
                        pc,
                        indent,
                    )
                elif ins.op is Op.RETURN:
                    values = ", ".join(f"_r{ins.a + i}" for i in range(ins.b))
                    if ins.b == 1:
                        values += ","
                    lines.append(f"{indent}_result = ({values})")
                    lines.append(f"{indent}return _FUNC_RETURN")
                elif ins.op is Op.HALT:
                    lines.append(f"{indent}_result = ()")
                    lines.append(f"{indent}return _FUNC_RETURN")
                else:
                    return super()._compile_ast_function(proto)
            else:
                lines.append(f"{indent}return {state_for(block.end)}")

        lines.append("    _jump_list = (" + ", ".join(block_names) + ",)")
        lines.extend(
            [
                "    try:",
                "        _state = 0",
                "        while _state >= 0:",
                "            _state = _jump_list[_state]()",
                "        if _state == _FUNC_RETURN:",
                "            return _FUNC_RETURN, _result",
                "        return _FUNC_SUSPEND, None",
                "    finally:",
                "        meter[0] += used",
            ]
        )

        tree = ast.parse("\n".join(lines))
        tree = inline_expression_helper(
            tree,
            "def _truthy(value):\n    return not (value is None or value is False)\n",
        )
        tree = inline_local_jump_list(
            tree,
            JumpListLayout("_jump_list", "_state", block_names),
        )
        ast.fix_missing_locations(tree)
        exec(compile(tree, "<luapyre-ir-function>", "exec"), namespace)
        return CompiledAstFunction(proto, namespace["_jit_ir_function"])
