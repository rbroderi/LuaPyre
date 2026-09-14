from __future__ import annotations

import ast
from dataclasses import replace
from types import FunctionType

from .ast_backend import JumpListLayout, inline_expression_helper, inline_local_jump_list
from .ast_jit import (
    _BACKEDGE_STATE,
    _CONTROL_OPS,
    _DEOPT_STATE,
    _INT_MAX,
    _INT_MIN,
    _MASK64,
    _SIGN64,
    _TWO64,
    AstPythonJIT,
)
from .bytecode import Closure, Op
from .errors import LuaRuntimeError
from .jit import CompiledLoop
from .opdispatch import _float_divide, _float_modulo, _float_power, _shift, _to_lua_string
from .table import LuaTable, _ABSENT, _hash_key
from .typed_ir import IRValue, IRValueKind, TypedIRCompiler, TypedIRPlan
from .values import coerce_lua_integer, lua_equal, static_value_type, type_matches


class TypedIRLoopJITMixin:
    """Compile the small typed IR to optimized Python AST.

    The IR owns value/type propagation, loop-invariance, and deoptimization
    facts. This mixin is a Python backend only: it chooses concrete inline-cache
    layouts and emits the Python control flow. Keeping that boundary explicit is
    the 0.16 direction: LuaPyre optimizations target the IR rather than growing
    Python-specific peepholes in the semantic compiler.
    """

    @staticmethod
    def _ir_expr(value: IRValue) -> str:
        if value.kind is IRValueKind.CONSTANT:
            return f"consts[{value.index}]"
        if value.kind is IRValueKind.UPVALUE:
            return f"upvalues[{value.index}].value"
        return f"_r{value.index}"

    @classmethod
    def _ir_spill_lines(
        cls,
        registers: tuple[int, ...],
        state: dict[int, IRValue],
        indent: str,
    ) -> list[str]:
        lines = []
        for reg in registers:
            value = state.get(reg)
            expression = cls._ir_expr(value) if value is not None else f"_r{reg}"
            lines.append(f"{indent}regs[{reg}] = {expression}")
        return lines

    @classmethod
    def _ir_deopt_lines(
        cls,
        registers: tuple[int, ...],
        plan: TypedIRPlan,
        pc: int,
        indent: str,
    ) -> list[str]:
        return [
            *cls._ir_spill_lines(registers, plan.virtual_state(pc), indent),
            f"{indent}frame.pc = {pc}",
            f"{indent}return _DEOPT_STATE",
        ]

    @staticmethod
    def _constant_array_index(value: object) -> int | None:
        if type(value) is int and value >= 1:
            return value
        if type(value) is float and value >= 1 and value.is_integer():
            return int(value)
        return None

    def _emit_typed_ir_instruction(
        self,
        frame,
        item,
        *,
        plan: TypedIRPlan,
        registers: tuple[int, ...],
        captured: set[int],
        indent: str,
        expected_calls: dict[int, tuple[str, str]],
        sparse_tables: dict[int, int] | None = None,
    ) -> list[str] | None:
        site = plan.instruction(item.pc)
        if site is None:
            return AstPythonJIT._emit_instruction(
                self,
                frame,
                item,
                registers=registers,
                captured=captured,
                indent=indent,
                expected_calls=expected_calls,
                sparse_tables=sparse_tables,
            )

        ins, pc, op = item.ins, item.pc, item.ins.op
        if frame.proto.jit_fully_typed and op in (Op.MOD, Op.EQ, Op.LT, Op.LE):
            left = site.value_for(ins.b).type_name
            right = site.value_for(ins.c).type_name
            integers = ("integer", "integer_lua")
            numeric = (*integers, "float", "number")
            proven = (
                "int" if left in integers and right in integers
                else "number" if left in numeric and right in numeric
                else "bytes" if left == right == "string" and op is not Op.MOD
                else None
            )
            if proven is not None:
                item = replace(item, specialization=proven, types_proven=True)
        if site.dead_definition:
            # The original Lua instruction still consumes one unit of fuel. Its
            # register value is virtual and is reconstructed from IR state on a
            # guard/deopt side exit.
            return [f"{indent}used += 1"]

        if op is Op.CONCAT and frame.proto.jit_fully_typed:
            left = site.value_for(ins.b).type_name
            right = site.value_for(ins.c).type_name
            if left == right == "string" or item.specialization == "bytes":
                out = []
                if left != "string" or right != "string":
                    out.append(
                        f"{indent}if not isinstance(_r{ins.b}, bytes) or not isinstance(_r{ins.c}, bytes):"
                    )
                    out.extend(
                        f"{indent}    {line.strip()}"
                        for line in self._ir_deopt_lines(registers, plan, pc, "")
                    )
                out.extend(
                    [
                        f"{indent}used += 1",
                        f"{indent}_r{ins.a} = _r{ins.b} + _r{ins.c}",
                    ]
                )
                return out

        if op is Op.GETUPVAL:
            # Non-virtual GETUPVAL shapes are not part of the first IR backend.
            # Fail closed so the prior tiers/interpreter keep exact behavior.
            return None

        if op is Op.GETTABLE and site.specialization in (
            "global_get",
            "table_get_const",
        ):
            if pc in plan.invariant_sites:
                # IR proved that nothing in this loop can mutate/re-enter _ENV.
                # Keep the Lua fuel charge at the original bytecode position but
                # use the value loaded once before the generated Python loop.
                return [
                    f"{indent}used += 1",
                    f"{indent}_r{ins.a} = _licm_{pc}_value",
                ]

            table_value = site.value_for(ins.b)
            key_value = site.value_for(ins.c)
            if key_value.kind is not IRValueKind.CONSTANT:
                return None
            table_expr = self._ir_expr(table_value)
            key_expr = self._ir_expr(key_value)
            key = frame.proto.constants[key_value.index]
            array_index = self._constant_array_index(key)
            table_tmp = f"_ir_table_{pc}"
            item_tmp = f"_ir_item_{pc}"
            cache_table = f"_ic_{pc}_table"
            cache_version = f"_ic_{pc}_version"
            cache_value = f"_ic_{pc}_value"
            token = f"_key_token_{pc}"
            out = [f"{indent}{table_tmp} = {table_expr}"]
            out.append(
                f"{indent}if not isinstance({table_tmp}, _LuaTable) or {table_tmp}.metatable is not None:"
            )
            out.extend(
                f"{indent}    {line.strip()}"
                for line in self._ir_deopt_lines(registers, plan, pc, "")
            )
            out.extend(
                [
                    f"{indent}used += 1",
                    f"{indent}if {table_tmp} is {cache_table} and {table_tmp}.version == {cache_version}:",
                    f"{indent}    _r{ins.a} = {cache_value}",
                    f"{indent}else:",
                ]
            )
            if array_index is not None:
                out.extend(
                    [
                        f"{indent}    if {array_index} <= len({table_tmp}.array):",
                        f"{indent}        _r{ins.a} = {table_tmp}.array[{array_index - 1}]",
                        f"{indent}    else:",
                        f"{indent}        {item_tmp} = {table_tmp}.hash.get({token}, _ABSENT)",
                        f"{indent}        _r{ins.a} = None if {item_tmp} is _ABSENT else {item_tmp}[1]",
                    ]
                )
            else:
                out.extend(
                    [
                        f"{indent}    {item_tmp} = {table_tmp}.hash.get({token}, _ABSENT)",
                        f"{indent}    _r{ins.a} = None if {item_tmp} is _ABSENT else {item_tmp}[1]",
                    ]
                )
            out.extend(
                [
                    f"{indent}    {cache_table} = {table_tmp}",
                    f"{indent}    {cache_version} = {table_tmp}.version",
                    f"{indent}    {cache_value} = _r{ins.a}",
                ]
            )
            return out

        if op is Op.SETTABLE and site.specialization in (
            "global_set",
            "table_set_const",
        ):
            table_value = site.value_for(ins.a)
            key_value = site.value_for(ins.b)
            if key_value.kind is not IRValueKind.CONSTANT:
                return None
            table_expr = self._ir_expr(table_value)
            key_expr = self._ir_expr(key_value)
            key = frame.proto.constants[key_value.index]
            array_index = self._constant_array_index(key)
            table_tmp = f"_ir_table_{pc}"
            token = f"_key_token_{pc}"
            value = f"_r{ins.c}"
            out = [f"{indent}{table_tmp} = {table_expr}"]
            out.append(
                f"{indent}if not isinstance({table_tmp}, _LuaTable) or {table_tmp}.metatable is not None:"
            )
            out.extend(
                f"{indent}    {line.strip()}"
                for line in self._ir_deopt_lines(registers, plan, pc, "")
            )
            out.append(f"{indent}used += 1")
            if array_index is None:
                # Constant hash keys need neither _hash_key nor rawset at run
                # time. The token is prepared once when the Python function is
                # compiled. Version changes still invalidate every read cache.
                out.extend(
                    [
                        f"{indent}if {value} is None:",
                        f"{indent}    {table_tmp}.rawset_prehashed({key_expr}, {token}, None)",
                        f"{indent}else:",
                        f"{indent}    {table_tmp}.version += 1",
                        f"{indent}    {table_tmp}.hash[{token}] = ({key_expr}, {value})",
                    ]
                )
            else:
                # Array writes have migration/trailing-nil rules; retain the
                # proven rawset implementation for this first IR tranche.
                out.append(f"{indent}{table_tmp}.rawset({key_expr}, {value})")
            return out

        return AstPythonJIT._emit_instruction(
            self,
            frame,
            item,
            registers=registers,
            captured=captured,
            indent=indent,
            expected_calls=expected_calls,
            sparse_tables=sparse_tables,
        )

    def _compile_ast_loop(self, frame, start_pc: int, backedge_pc: int):
        lowered = self._lower_ast_region(frame, start_pc, backedge_pc)
        if lowered is None:
            return None
        ir, blocks = lowered

        block_code = tuple(
            tuple((item.pc, item.ins) for item in block.instructions)
            for block in blocks
        )
        plan = TypedIRCompiler(frame.proto).compile(block_code)

        register_set = set(self._used_registers(ir))
        for item in ir.body.instructions:
            if item.ins.op is Op.GETUPVAL:
                register_set.add(item.ins.a)
        registers = tuple(sorted(register_set))
        captured = self._captured_registers(frame.proto)
        sparse_tables = self._sparse_table_registers(frame, ir)
        if captured.intersection(registers):
            return None

        block_index = {block.start: index for index, block in enumerate(blocks)}

        def state_for(pc: int) -> int:
            if pc == backedge_pc:
                return _BACKEDGE_STATE
            return block_index[pc]

        expected_calls: dict[int, tuple[str, str]] = {}
        expected_values: dict[str, object] = {"_ABSENT": _ABSENT}
        for block in blocks:
            for item in block.instructions:
                if item.ins.op is Op.CALL:
                    fn = frame.regs[item.ins.b]
                    if not isinstance(fn, Closure):
                        return None
                    expected_name = f"_expected_call_{item.pc}"
                    const_name = f"_call_consts_{item.pc}"
                    expected_calls[item.pc] = (expected_name, const_name)
                    expected_values[expected_name] = fn
                    expected_values[const_name] = fn.proto.constants

                site = plan.instruction(item.pc)
                if site is None or site.specialization not in (
                    "global_get",
                    "global_set",
                    "table_get_const",
                    "table_set_const",
                ):
                    continue
                key_reg = item.ins.c if item.ins.op is Op.GETTABLE else item.ins.b
                key_value = site.value_for(key_reg)
                if key_value.kind is IRValueKind.CONSTANT:
                    expected_values[f"_key_token_{item.pc}"] = _hash_key(
                        frame.proto.constants[key_value.index]
                    )

        lines = [
            "def _jit_ast_loop(vm, frame, budget):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
            "    cells = frame.cells",
            "    upvalues = frame.closure.upvalues",
            "    used = 0",
        ]
        for reg in registers:
            lines.append(f"    _r{reg} = regs[{reg}]")
        for owner in sorted(set(sparse_tables.values())):
            table = self._reg(owner)
            sparse = f"_sparse_{owner}"
            lines.extend(
                [
                    f"    {sparse} = {table}._sparse_int",
                    f"    if {sparse} is None and not {table}.array and not {table}.hash and {table}._deleted_successors is None:",
                    f"        {sparse} = {{}}",
                    f"        {table}._sparse_int = {sparse}",
                    f"        if {table}._gc_owner is not None:",
                    f"            {table}._gc_owner.account_bytes(_SPARSE_DICT_BYTES)",
                ]
            )

        # Loop-invariant global reads are an IR optimization, not a Python AST
        # peephole. The backend performs the proven access exactly once per JIT
        # runner invocation. If the receiver has become dynamic/metatable-backed,
        # fail closed before executing any Lua instruction and let Tier 0 resume
        # at the original loop start with untouched registers/fuel.
        for pc in plan.invariant_sites:
            site = plan.instruction(pc)
            if site is None:
                return None
            ins = site.ins
            table_value = site.value_for(ins.b)
            key_value = site.value_for(ins.c)
            if key_value.kind is not IRValueKind.CONSTANT:
                return None
            table_expr = self._ir_expr(table_value)
            key = frame.proto.constants[key_value.index]
            array_index = self._constant_array_index(key)
            table_tmp = f"_licm_{pc}_table"
            item_tmp = f"_licm_{pc}_item"
            token = f"_key_token_{pc}"
            lines.extend(
                [
                    f"    {table_tmp} = {table_expr}",
                    f"    if not isinstance({table_tmp}, _LuaTable) or {table_tmp}.metatable is not None:",
                    f"        frame.pc = {ir.start_pc}",
                    "        return 0, False",
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
        lines.append("    _state = 0")

        block_names = tuple(f"_jump_{index}" for index in range(len(blocks)))
        for index, block in enumerate(blocks):
            lines.append(f"    def {block_names[index]}():")
            indent = "        "
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
                emitted = self._emit_typed_ir_instruction(
                    frame,
                    item,
                    plan=plan,
                    registers=registers,
                    captured=captured,
                    indent=indent,
                    expected_calls=expected_calls,
                    sparse_tables=sparse_tables,
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

        lines.append("    _jump_list = (" + ", ".join(block_names) + ",)")
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

        namespace = {
            "_BACKEDGE_STATE": _BACKEDGE_STATE,
            "_DEOPT_STATE": _DEOPT_STATE,
            "_INT_MIN": _INT_MIN,
            "_INT_MAX": _INT_MAX,
            "_MASK64": _MASK64,
            "_SIGN64": _SIGN64,
            "_SPARSE_DICT_BYTES": __import__("sys").getsizeof({}),
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
        exec(compile(tree, "<luapyre-typed-ir-region>", "exec"), namespace)
        raw_runner: FunctionType = namespace["_jit_ast_loop"]
        return CompiledLoop(ir, max(1, len(ir.body.instructions) + 1), raw_runner)
