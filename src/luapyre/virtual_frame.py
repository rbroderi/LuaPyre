from __future__ import annotations

import ast
from types import FunctionType

from .ast_backend import inline_type_guards
from .bytecode import Op, Proto
from .errors import LuaRuntimeError
from .function_jit import _FUNC_RETURN, _FUNC_SUSPEND
from .opdispatch import _float_divide
from .range_analysis import analyze_integer_ranges
from .values import static_value_type, type_matches
from .vm import Frame


_CONTROL = frozenset({Op.JMP, Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL, Op.FORPREP, Op.FORLOOP, Op.JFORLOOP})
_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63
_TWO64 = 1 << 64


def compile_virtual_frame(
    proto: Proto,
    *,
    arg_count: int | None = None,
    trusted_args: bool = False,
) -> FunctionType | None:
    """Compile a proven non-escaping child without allocating its Frame."""

    ranges = analyze_integer_ranges(proto)
    leaders = {0, len(proto.code)}
    for pc, ins in enumerate(proto.code):
        if ins.op is Op.JMP:
            leaders.update((ins.a, pc + 1))
        elif ins.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
            leaders.update((ins.a, pc + 1))
        elif ins.op in (Op.FORPREP, Op.FORLOOP, Op.JFORLOOP):
            leaders.update((ins.d, pc + 1))
        elif ins.op in (Op.RETURN, Op.HALT):
            leaders.add(pc + 1)
    if any(pc < 0 or pc > len(proto.code) for pc in leaders):
        return None
    ordered = sorted(leaders)
    blocks = [(start, ordered[i + 1]) for i, start in enumerate(ordered[:-1]) if start < ordered[i + 1]]
    registers = tuple(range(max(1, proto.register_count)))
    lines = [
        "def _run(vm, frames, closure, args, dest, want, budget, meter):",
        "    if len(frames) >= vm.max_frames:",
        "        raise _LuaRuntimeError('stack overflow')",
    ]
    for index in range(proto.param_count):
        expected = proto.param_types[index].name
        if arg_count is None:
            value = f"args[{index}] if {index} < len(args) else None"
        else:
            value = f"args[{index}]" if index < arg_count else "None"
        lines.append(f"    _r{index} = {value}")
        if not trusted_args:
            lines.extend([
                f"    if not _type_matches({expected!r}, _r{index}):",
                f"        raise _LuaRuntimeError(f'argument {index + 1}: expected {expected}, got {{_static_value_type(_r{index}).name}}')",
            ])
    for reg in registers[proto.param_count:]:
        lines.append(f"    _r{reg} = None")
    lines.extend(["    consts = closure.proto.constants", "    used = 0", "    _pc = 0", "    _active_pc = 0", "    try:", "        while True:"])

    def materialize(pc: int, indent: str) -> None:
        values = ", ".join(f"_r{reg}" for reg in registers)
        lines.extend([
            f"{indent}vm.jit.stats.virtual_frame_materializations += 1",
            f"{indent}vm.jit.function_suspends += 1",
            f"{indent}frames.append(_Frame(closure, [{values}], {pc}, dest, want))",
            f"{indent}return _FUNC_SUSPEND, None",
        ])

    for block_index, (start, end) in enumerate(blocks):
        prefix = "if" if block_index == 0 else "elif"
        lines.append(f"            {prefix} _pc == {start}:")
        indent = "                "
        lines.append(f"{indent}_active_pc = {start}")
        lines.append(f"{indent}if budget - meter[0] - used < {end - start}:")
        materialize(start, indent + "    ")
        terminal = proto.code[end - 1]
        is_terminal = terminal.op in _CONTROL or terminal.op in (Op.RETURN, Op.HALT)
        ordinary_end = end - 1 if is_terminal else end
        pending_cost = 0
        for pc in range(start, ordinary_end):
            ins = proto.code[pc]
            # The whole block has already passed its fuel preflight.  Keep
            # pure operations in a local batch, flushing only before an
            # instruction that can raise a Lua error and therefore needs an
            # exact diagnostic PC and consumed prefix.
            if ins.op is Op.GUARD:
                if pending_cost:
                    lines.append(f"{indent}used += {pending_cost}")
                    pending_cost = 0
                lines.append(f"{indent}_active_pc = {pc}")
            a, b, c = f"_r{ins.a}", f"_r{ins.b}", f"_r{ins.c}"
            if ins.op is Op.LOADK:
                lines.append(f"{indent}{a} = consts[{ins.b}]")
            elif ins.op in (Op.MOVE, Op.LOCAL):
                lines.append(f"{indent}{a} = {b}")
            elif ins.op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                symbol = {Op.ADD_I: "+", Op.SUB_I: "-", Op.MUL_I: "*"}[ins.op]
                if ranges.overflow_free(pc):
                    lines.append(f"{indent}{a} = {b} {symbol} {c}")
                else:
                    lines.extend([f"{indent}_wide_{pc} = ({b} {symbol} {c}) & _MASK64", f"{indent}{a} = _wide_{pc} - _TWO64 if _wide_{pc} & _SIGN64 else _wide_{pc}"])
            elif ins.op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                symbol = {Op.ADD_F: "+", Op.SUB_F: "-", Op.MUL_F: "*"}[ins.op]
                lines.append(f"{indent}{a} = float({b} {symbol} {c})")
            elif ins.op is Op.DIV:
                lines.append(f"{indent}{a} = _float_divide({b}, {c})")
            elif ins.op is Op.NOT:
                lines.append(f"{indent}{a} = ({b} is None or {b} is False)")
            elif ins.op is Op.TOBOOL:
                lines.append(f"{indent}{a} = not ({b} is None or {b} is False)")
            elif ins.op in (Op.LT, Op.LE):
                symbol = "<" if ins.op is Op.LT else "<="
                lines.append(f"{indent}{a} = {b} {symbol} {c}")
            elif ins.op is Op.GUARD:
                expected = proto.constants[ins.b]
                lines.extend([f"{indent}used += 1", f"{indent}if not _type_matches({expected!r}, {a}):", f"{indent}    raise _LuaRuntimeError(f'expected {expected!s}, got {{_static_value_type({a}).name}}')"])
            else:
                return None
            if ins.op is not Op.GUARD:
                pending_cost += 1

        if pending_cost:
            lines.append(f"{indent}used += {pending_cost}")

        if is_terminal:
            pc = end - 1
            ins = terminal
            lines.append(f"{indent}_active_pc = {pc}")
            lines.append(f"{indent}used += 1")
            if ins.op is Op.JMP:
                lines.extend([f"{indent}_pc = {ins.a}", f"{indent}continue"])
            elif ins.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
                if ins.op is Op.JMPIFNIL:
                    cond = f"_r{ins.b} is None"
                else:
                    cond = f"not (_r{ins.b} is None or _r{ins.b} is False)"
                    if ins.op is Op.JMPIFNOT:
                        cond = f"not ({cond})"
                lines.extend([f"{indent}_pc = {ins.a} if {cond} else {end}", f"{indent}continue"])
            elif ins.op is Op.FORPREP:
                a, b, c = f"_r{ins.a}", f"_r{ins.b}", f"_r{ins.c}"
                lines.extend([f"{indent}if {c} == 0:", f"{indent}    raise _LuaRuntimeError(\"'for' step is zero\")", f"{indent}_pc = {end} if ({a} <= {b} if {c} > 0 else {a} >= {b}) else {ins.d}", f"{indent}continue"])
            elif ins.op in (Op.FORLOOP, Op.JFORLOOP):
                a, b, c = f"_r{ins.a}", f"_r{ins.b}", f"_r{ins.c}"
                lines.extend([f"{indent}_next = {a} + {c}", f"{indent}if type({a}) is int and (_next < _INT_MIN or _next > _INT_MAX):", f"{indent}    _pc = {end}", f"{indent}else:", f"{indent}    {a} = _next", f"{indent}    _pc = {ins.d} if ({a} <= {b} if {c} > 0 else {a} >= {b}) else {end}", f"{indent}continue"])
            elif ins.op is Op.RETURN:
                values = ", ".join(f"_r{ins.a + i}" for i in range(ins.b))
                if ins.b == 1:
                    values += ","
                lines.extend([f"{indent}vm.jit.stats.virtual_frame_elisions += 1", f"{indent}vm.jit.function_executions += 1", f"{indent}return _FUNC_RETURN, ({values})"])
            elif ins.op is Op.HALT:
                lines.extend([f"{indent}vm.jit.stats.virtual_frame_elisions += 1", f"{indent}vm.jit.function_executions += 1", f"{indent}return _FUNC_RETURN, ()"])
            else:
                return None
        else:
            lines.extend([f"{indent}_pc = {end}", f"{indent}continue"])
    values = ", ".join(f"_r{reg}" for reg in registers)
    lines.extend([
        "            else:",
        "                raise RuntimeError('invalid virtual frame pc')",
        "    except _LuaRuntimeError:",
        "        vm.jit.stats.virtual_frame_materializations += 1",
        f"        frames.append(_Frame(closure, [{values}], _active_pc + 1, dest, want))",
        "        raise",
        "    finally:",
        "        meter[0] += used",
    ])
    namespace = {
        "_Frame": Frame, "_FUNC_RETURN": _FUNC_RETURN, "_FUNC_SUSPEND": _FUNC_SUSPEND,
        "_LuaRuntimeError": LuaRuntimeError, "_type_matches": type_matches,
        "_static_value_type": static_value_type, "_MASK64": _MASK64,
        "_float_divide": _float_divide,
        "_SIGN64": _SIGN64, "_TWO64": _TWO64,
        "_INT_MIN": -(1 << 63), "_INT_MAX": (1 << 63) - 1,
    }
    tree = inline_type_guards(ast.parse("\n".join(lines)))
    exec(compile(tree, "<luapyre-virtual-frame>", "exec"), namespace)
    return namespace["_run"]
