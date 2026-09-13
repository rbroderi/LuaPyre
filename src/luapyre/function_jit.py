from __future__ import annotations

import ast
from dataclasses import dataclass, field
from types import FunctionType

from .ast_backend import JumpListLayout, inline_expression_helper, inline_local_jump_list
from .bytecode import Closure, Ins, Op, Proto
from .errors import LuaRuntimeError
from .opdispatch import _float_divide, _float_modulo
from .range_analysis import analyze_integer_ranges
from .table import LuaTable, _ABSENT, _hash_key
from .values import lua_equal, static_value_type, type_matches


_FUNC_RETURN = -3
_FUNC_SUSPEND = -2

_FUNCTION_CONTROL = frozenset(
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

_FUNCTION_OPS = frozenset(
    {
        Op.LOADK,
        Op.MOVE,
        Op.LOCAL,
        Op.GETUPVAL,
        Op.SETUPVAL,
        Op.NEWTABLE,
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
        Op.MOD,
        Op.EQ,
        Op.LT,
        Op.LE,
        Op.NOT,
        Op.TOBOOL,
        Op.GUARD,
        Op.JMP,
        Op.JMPIF,
        Op.JMPIFNOT,
        Op.JMPIFNIL,
        Op.FORPREP,
        Op.FORLOOP,
        Op.JFORLOOP,
        Op.CALL,
        Op.RETURN,
        Op.HALT,
    }
)


@dataclass(frozen=True, slots=True)
class CompiledAstFunction:
    proto: Proto
    runner: FunctionType
    frame_pool: list[object] = field(default_factory=list, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class _FunctionBlock:
    start: int
    end: int
    instructions: tuple[tuple[int, Ins], ...]


class TypedFunctionJITMixin:
    """Whole-function AST compilation for certified fully typed closures.

    The VM keeps real ``Frame`` objects on its stack while generated Python runs.
    A guard miss therefore does not restart a function: promoted locals are
    spilled, ``frame.pc`` is set to the exact instruction, and execution simply
    resumes in Tier 0. Nested compiled calls use the same rule, which makes
    direct recursion safe and keeps traceback/unwind state available.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._function_hot: dict[int, tuple[Proto, int]] = {}
        self._function_cache: dict[int, tuple[Proto, CompiledAstFunction | None]] = {}
        self._virtual_frame_cache: dict[int, tuple[Proto, FunctionType | None]] = {}
        self.function_compiles = 0
        self.function_executions = 0
        self.function_suspends = 0

    @staticmethod
    def _function_target(ins: Ins, fallthrough: int) -> tuple[int, ...]:
        if ins.op is Op.JMP:
            return (ins.a,)
        if ins.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
            return (ins.a, fallthrough)
        if ins.op is Op.FORPREP:
            return (ins.d, fallthrough)
        if ins.op in (Op.FORLOOP, Op.JFORLOOP):
            return (ins.d, fallthrough)
        return (fallthrough,)

    @staticmethod
    def _function_writes_register(ins: Ins, reg: int) -> bool:
        if ins.op is Op.CALL:
            return ins.e > 0 and ins.a <= reg < ins.a + ins.e
        if ins.op in (Op.SETUPVAL, Op.SETTABLE):
            return False
        return ins.op not in _FUNCTION_CONTROL and ins.op not in (
            Op.RETURN, Op.HALT
        ) and ins.a == reg

    def _function_blocks(self, proto: Proto) -> tuple[_FunctionBlock, ...] | None:
        if not proto.code or any(ins.op not in _FUNCTION_OPS for ins in proto.code):
            return None
        leaders = {0, len(proto.code)}
        for pc, ins in enumerate(proto.code):
            if ins.op is Op.CALL:
                leaders.add(pc + 1)
                continue
            if ins.op in _FUNCTION_CONTROL:
                for target in self._function_target(ins, pc + 1):
                    if target < 0 or target > len(proto.code):
                        return None
                    leaders.add(target)
                leaders.add(pc + 1)
            elif ins.op in (Op.RETURN, Op.HALT):
                leaders.add(pc + 1)

        ordered = sorted(leaders)
        blocks: list[_FunctionBlock] = []
        for index, start in enumerate(ordered[:-1]):
            end = ordered[index + 1]
            instructions = tuple((pc, proto.code[pc]) for pc in range(start, end))
            if not instructions:
                continue
            for _pc, ins in instructions[:-1]:
                if ins.op in _FUNCTION_CONTROL or ins.op in (Op.CALL, Op.RETURN, Op.HALT):
                    return None
            blocks.append(_FunctionBlock(start, end, instructions))

        starts = {block.start for block in blocks}
        starts.add(len(proto.code))
        for block in blocks:
            terminal = block.instructions[-1][1]
            if terminal.op in _FUNCTION_CONTROL:
                for target in self._function_target(terminal, block.end):
                    if target not in starts:
                        return None
        return tuple(blocks)

    def get_compiled_function(self, closure: Closure) -> CompiledAstFunction | None:
        proto = closure.proto
        if not proto.jit_fully_typed or proto.is_vararg:
            return None
        ident = id(proto)
        cached = self._function_cache.get(ident)
        if cached is not None and cached[0] is proto:
            return cached[1]
        compiled = self._compile_ast_function(proto)
        self._function_cache[ident] = (proto, compiled)
        if compiled is not None:
            self.function_compiles += 1
        return compiled

    def maybe_function(self, closure: Closure) -> CompiledAstFunction | None:
        proto = closure.proto
        if not proto.jit_fully_typed or proto.is_vararg:
            return None
        ident = id(proto)
        cached = self._function_cache.get(ident)
        if cached is not None and cached[0] is proto:
            return cached[1]
        hot_entry = self._function_hot.get(ident)
        hot = hot_entry[1] if hot_entry is not None and hot_entry[0] is proto else 0
        hot += 1
        self._function_hot[ident] = (proto, hot)
        if hot < self.threshold:
            return None
        return self.get_compiled_function(closure)

    def run_compiled_child(
        self,
        vm,
        frames,
        parent,
        closure: Closure,
        args: tuple,
        dest: int,
        want: int,
        budget: int,
        meter: list[int],
        compiled: CompiledAstFunction,
    ):
        if len(frames) >= vm.max_frames:
            raise LuaRuntimeError("stack overflow")
        ident = id(closure.proto)
        virtual_entry = self._virtual_frame_cache.get(ident)
        if virtual_entry is not None and virtual_entry[0] is closure.proto:
            virtual = virtual_entry[1]
        else:
            # Import lazily: virtual_frame depends on this module's status
            # constants.  The compiled result, including refusal, is permanent
            # for an immutable Proto.
            from .virtual_frame import compile_virtual_frame

            virtual = compile_virtual_frame(closure.proto)
            self._virtual_frame_cache[ident] = (closure.proto, virtual)
        if virtual is not None:
            return virtual(vm, frames, closure, args, dest, want, budget, meter)
        acquire = getattr(vm, "_acquire_compiled_frame", None)
        child = (
            acquire(compiled, closure, args, dest, want)
            if acquire is not None
            else vm._new_frame(closure, list(args), dest, want)
        )
        frames.append(child)
        status, values = compiled.runner(vm, frames, child, budget, meter)
        if status == _FUNC_RETURN:
            if not frames or frames[-1] is not child:
                raise RuntimeError("compiled function stack mismatch")
            frames.pop()
            self.function_executions += 1
            release = getattr(vm, "_release_compiled_frame", None)
            if release is not None:
                release(compiled, child)
            return _FUNC_RETURN, values
        self.function_suspends += 1
        return _FUNC_SUSPEND, None

    def _compile_ast_function(self, proto: Proto) -> CompiledAstFunction | None:
        blocks = self._function_blocks(proto)
        if blocks is None:
            return None
        # Functions that create child closures need exact open-cell identity.
        # Those stay in Tier 0 until the closure/cell lowering tranche.
        if proto.children:
            return None

        registers = tuple(range(max(1, proto.register_count)))
        ranges = analyze_integer_ranges(proto)
        block_index = {block.start: index for index, block in enumerate(blocks)}

        def state_for(pc: int) -> int:
            if pc == len(proto.code):
                return _FUNC_RETURN
            return block_index[pc]

        def spill(indent: str) -> list[str]:
            return [f"{indent}regs[{reg}] = _r{reg}" for reg in registers]

        def suspend(pc: int, indent: str) -> list[str]:
            return [
                *spill(indent),
                f"{indent}frame.pc = {pc}",
                f"{indent}return _FUNC_SUSPEND",
            ]

        lines = [
            "def _jit_ast_function(vm, frames, frame, budget, meter):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
            "    upvalues = frame.closure.upvalues",
            "    used = 0",
            "    _result = ()",
        ]
        for reg in registers:
            lines.append(f"    _r{reg} = regs[{reg}]")
        lines.append("    _state = 0")

        block_names = tuple(f"_jump_{index}" for index in range(len(blocks)))

        def i64(
            dest: str, expression: str, tag: str, indent: str, *, overflow_free=False
        ) -> list[str]:
            if overflow_free:
                return [f"{indent}{dest} = {expression}"]
            tmp = f"_i64_{tag}"
            return [
                f"{indent}{tmp} = ({expression}) & _MASK64",
                f"{indent}{dest} = {tmp} - _TWO64 if {tmp} & _SIGN64 else {tmp}",
            ]

        def deopt(out: list[str], condition: str, pc: int, indent: str) -> None:
            out.append(f"{indent}if {condition}:")
            out.extend(f"{indent}    {line.strip()}" for line in suspend(pc, ""))

        def emit_forprep(out: list[str], ins: Ins, true_state: int, false_state: int, indent: str):
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

        def emit_forloop(out: list[str], ins: Ins, loop_state: int, exit_state: int, pc: int, indent: str):
            a, b, c = f"_r{ins.a}", f"_r{ins.b}", f"_r{ins.c}"
            tag = f"f_{pc}_{ins.a}"
            if all(ranges.range_at(pc, reg) is not None for reg in (ins.a, ins.b, ins.c)):
                out.extend(
                    [
                        f"{indent}_next_{tag} = {a} + {c}",
                        f"{indent}if _next_{tag} < _INT_MIN or _next_{tag} > _INT_MAX or ({c} > 0 and _next_{tag} > {b}) or ({c} < 0 and _next_{tag} < {b}):",
                        f"{indent}    return {exit_state}",
                        f"{indent}{a} = _next_{tag}",
                        f"{indent}return {loop_state}",
                    ]
                )
                return
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

        constant_tokens: dict[str, object] = {}
        call_target_caches: dict[str, list[object | None]] = {}
        for block_no, block in enumerate(blocks):
            lines.append(f"    def {block_names[block_no]}():")
            indent = "        "
            known_constants: dict[int, object] = {}
            cost = len(block.instructions)
            lines.append(f"{indent}if budget - meter[0] - used < {cost}:")
            lines.extend(f"{indent}    {line.strip()}" for line in suspend(block.start, ""))

            terminal_pc, terminal_ins = block.instructions[-1]
            terminal_control = terminal_ins.op in _FUNCTION_CONTROL or terminal_ins.op in (
                Op.RETURN,
                Op.HALT,
            )
            ordinary = block.instructions[:-1] if terminal_control else block.instructions

            for pc, ins in ordinary:
                op = ins.op
                a, b, c = f"_r{ins.a}", f"_r{ins.b}", f"_r{ins.c}"
                if op is Op.LOADK:
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = consts[{ins.b}]"])
                    known_constants[ins.a] = proto.constants[ins.b]
                elif op in (Op.MOVE, Op.LOCAL):
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = {b}"])
                    if ins.b in known_constants:
                        known_constants[ins.a] = known_constants[ins.b]
                    else:
                        known_constants.pop(ins.a, None)
                elif op is Op.GETUPVAL:
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = upvalues[{ins.b}].value"])
                elif op is Op.SETUPVAL:
                    lines.extend([
                        f"{indent}used += 1",
                        f"{indent}vm.gc.write_barrier(upvalues[{ins.a}], {b})",
                        f"{indent}upvalues[{ins.a}].value = {b}",
                    ])
                elif op is Op.NEWTABLE:
                    lines.append(f"{indent}used += 1")
                    lines.extend(spill(indent))
                    lines.append(f"{indent}{a} = vm._new_table()")
                elif op is Op.GETTABLE:
                    deopt(lines, f"not isinstance({b}, _LuaTable) or {b}.metatable is not None", pc, indent)
                    key = known_constants.get(ins.c, _ABSENT)
                    if type(key) is float and key.is_integer():
                        key = int(key)
                    if key is _ABSENT:
                        lines.extend([f"{indent}used += 1", f"{indent}{a} = {b}.rawget({c})"])
                    else:
                        token_name = f"_key_token_{pc}"
                        constant_tokens[token_name] = _hash_key(key)
                        lines.append(f"{indent}used += 1")
                        if type(key) is int and key >= 1:
                            lines.extend([
                                f"{indent}if {key} <= len({b}.array):",
                                f"{indent}    {a} = {b}.array[{key - 1}]",
                                f"{indent}else:",
                                f"{indent}    _item_{pc} = {b}.hash.get({token_name}, _ABSENT)",
                                f"{indent}    {a} = None if _item_{pc} is _ABSENT else _item_{pc}[1]",
                            ])
                        else:
                            lines.extend([
                                f"{indent}_item_{pc} = {b}.hash.get({token_name}, _ABSENT)",
                                f"{indent}{a} = None if _item_{pc} is _ABSENT else _item_{pc}[1]",
                            ])
                    known_constants.pop(ins.a, None)
                elif op is Op.SETTABLE:
                    deopt(lines, f"not isinstance({a}, _LuaTable) or {a}.metatable is not None", pc, indent)
                    deopt(lines, f"{b} is None or (type({b}) is float and _isnan({b}))", pc, indent)
                    lines.extend([f"{indent}used += 1", f"{indent}{a}.rawset({b}, {c})"])
                elif op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                    symbol = {Op.ADD_I: "+", Op.SUB_I: "-", Op.MUL_I: "*"}[op]
                    lines.append(f"{indent}used += 1")
                    lines.extend(
                        i64(
                            a, f"{b} {symbol} {c}", str(pc), indent,
                            overflow_free=ranges.overflow_free(pc),
                        )
                    )
                elif op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                    symbol = {Op.ADD_F: "+", Op.SUB_F: "-", Op.MUL_F: "*"}[op]
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = float({b} {symbol} {c})"])
                elif op in (Op.ADD, Op.SUB, Op.MUL):
                    symbol = {Op.ADD: "+", Op.SUB: "-", Op.MUL: "*"}[op]
                    deopt(lines, f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES", pc, indent)
                    tmp = f"_arith_{pc}"
                    lines.extend([f"{indent}used += 1", f"{indent}{tmp} = {b} {symbol} {c}"])
                    lines.append(f"{indent}if type({b}) is int and type({c}) is int:")
                    lines.extend(i64(a, tmp, f"g_{pc}", indent + "    "))
                    lines.append(f"{indent}else:")
                    lines.append(f"{indent}    {a} = {tmp}")
                elif op is Op.DIV:
                    deopt(lines, f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES", pc, indent)
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = _float_divide({b}, {c})"])
                elif op is Op.MOD:
                    deopt(lines, f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES", pc, indent)
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
                    lines.append(f"{indent}if not _type_matches({proto.constants[ins.b]!r}, {a}):")
                    lines.append(
                        f"{indent}    raise _LuaRuntimeError(f\"expected {{consts[{ins.b}]!s}}, got {{_static_value_type({a}).name}}\")"
                    )
                elif op is Op.CALL:
                    # Do not execute a dynamic/host call inside a partially
                    # compiled function. Suspend before the CALL so Tier 0 can
                    # perform every Lua metamethod/callability check exactly.
                    lines.append(f"{indent}_fn_{pc} = {b}")
                    cache_name = f"_call_target_{pc}"
                    call_target_caches.setdefault(cache_name, [None, None])
                    lines.extend([
                        f"{indent}_target_cache_{pc} = {cache_name}",
                        f"{indent}if _fn_{pc} is _target_cache_{pc}[0]:",
                        f"{indent}    _compiled_{pc} = _target_cache_{pc}[1]",
                        f"{indent}else:",
                        f"{indent}    _compiled_{pc} = vm.jit.get_compiled_function(_fn_{pc}) if isinstance(_fn_{pc}, _Closure) else None",
                        f"{indent}    _target_cache_{pc}[0] = _fn_{pc}",
                        f"{indent}    _target_cache_{pc}[1] = _compiled_{pc}",
                    ])
                    lines.append(f"{indent}if _compiled_{pc} is None:")
                    lines.extend(f"{indent}    {line.strip()}" for line in suspend(pc, ""))
                    lines.append(f"{indent}used += 1")
                    # Flush the parent's exact instruction count before the
                    # child runs. The child's own finally block uses the same
                    # meter, so exceptions and recursive suspension preserve
                    # fuel exactly without a per-op mutable-list increment.
                    lines.append(f"{indent}meter[0] += used")
                    lines.append(f"{indent}used = 0")
                    lines.extend(spill(indent))
                    lines.append(f"{indent}frame.pc = {pc + 1}")
                    args = ", ".join(f"_r{ins.c + i}" for i in range(ins.d))
                    if ins.d == 1:
                        args += ","
                    lines.append(
                        f"{indent}_status_{pc}, _values_{pc} = vm.jit.run_compiled_child(vm, frames, frame, _fn_{pc}, ({args}), {ins.a}, {ins.e}, budget, meter, _compiled_{pc})"
                    )
                    lines.append(f"{indent}if _status_{pc} == _FUNC_SUSPEND:")
                    lines.append(f"{indent}    return _FUNC_SUSPEND")
                    if ins.e > 0:
                        for value_index in range(ins.e):
                            lines.append(
                                f"{indent}_r{ins.a + value_index} = _values_{pc}[{value_index}] if {value_index} < len(_values_{pc}) else None"
                            )
                else:
                    return None

                if op is Op.CALL and ins.e > 0:
                    for result_reg in range(ins.a, ins.a + ins.e):
                        known_constants.pop(result_reg, None)
                elif op not in (Op.LOADK, Op.MOVE, Op.LOCAL, Op.GETTABLE) and self._function_writes_register(ins, ins.a):
                    known_constants.pop(ins.a, None)

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
                    emit_forloop(lines, ins, state_for(ins.d), state_for(block.end), pc, indent)
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
                    return None
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
        namespace = {
            "_FUNC_RETURN": _FUNC_RETURN,
            "_FUNC_SUSPEND": _FUNC_SUSPEND,
            "_INT_MIN": -(1 << 63),
            "_INT_MAX": (1 << 63) - 1,
            "_MASK64": (1 << 64) - 1,
            "_SIGN64": 1 << 63,
            "_TWO64": 1 << 64,
            "_NUM_TYPES": (int, float),
            "_Closure": Closure,
            "_LuaRuntimeError": LuaRuntimeError,
            "_LuaTable": LuaTable,
            "_float_divide": _float_divide,
            "_float_modulo": _float_modulo,
            "_isnan": __import__("math").isnan,
            "_lua_equal": lua_equal,
            "_static_value_type": static_value_type,
            "_type_matches": type_matches,
            "_ABSENT": _ABSENT,
            **constant_tokens,
            **call_target_caches,
        }
        exec(compile(tree, "<luapyre-ast-function>", "exec"), namespace)
        return CompiledAstFunction(proto, namespace["_jit_ast_function"])
