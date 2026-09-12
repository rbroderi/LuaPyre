from __future__ import annotations

import ast
from collections.abc import Callable

from .ast_backend import optimize_semantic_helpers
from .bytecode import Op
from .values import MASK64, SIGN64


_TWO64 = 1 << 64

Emitter = Callable[[list[str], object, int, bool], bool]


def _loop_guard(lines: list[str], condition: str, pc: int, offset: int) -> None:
    lines.append(f"        if not ({condition}):")
    lines.append(f"            frame.pc = {pc}")
    lines.append(f"            return used + {offset}, False")


def _leaf_guard(lines: list[str], condition: str) -> None:
    lines.append(f"    if not ({condition}): return _DEOPT")


def _loop_loadk(lines, item, offset, trusted):
    ins = item.ins
    lines.append(f"        regs[{ins.a}] = consts[{ins.b}]")
    return True


def _loop_move(lines, item, offset, trusted):
    ins = item.ins
    lines.append(f"        regs[{ins.a}] = regs[{ins.b}]")
    return True


def _loop_local(lines, item, offset, trusted):
    ins = item.ins
    lines.append(f"        _v = regs[{ins.b}]")
    lines.append(f"        regs[{ins.a}] = _v")
    lines.append(f"        if {ins.a} in cells:")
    lines.append(f"            cells[{ins.a}] = _Cell(_v)")
    return True


def _loop_newtable(lines, item, offset, trusted):
    ins = item.ins
    lines.append(f"        regs[{ins.a}] = _LuaTable()")
    return True


def _loop_gettable(lines, item, offset, trusted):
    ins, pc = item.ins, item.pc
    _loop_guard(
        lines,
        f"isinstance(regs[{ins.b}], _LuaTable) and regs[{ins.b}].metatable is None",
        pc,
        offset,
    )
    lines.append(f"        regs[{ins.a}] = regs[{ins.b}].rawget(regs[{ins.c}])")
    return True


def _loop_settable(lines, item, offset, trusted):
    ins, pc = item.ins, item.pc
    _loop_guard(
        lines,
        f"isinstance(regs[{ins.a}], _LuaTable) and regs[{ins.a}].metatable is None",
        pc,
        offset,
    )
    lines.append(f"        _key = regs[{ins.b}]")
    _loop_guard(
        lines,
        "_key is not None and not (type(_key) is float and _isnan(_key))",
        pc,
        offset,
    )
    lines.append(f"        regs[{ins.a}].rawset(_key, regs[{ins.c}])")
    return True


def _make_loop_int_binop(symbol: str) -> Emitter:
    def emit(lines, item, offset, trusted):
        ins, pc = item.ins, item.pc
        if not trusted:
            _loop_guard(
                lines,
                f"type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int",
                pc,
                offset,
            )
        lines.append(
            f"        regs[{ins.a}] = _i64(regs[{ins.b}] {symbol} regs[{ins.c}])"
        )
        return True

    return emit


def _make_loop_float_binop(symbol: str) -> Emitter:
    def emit(lines, item, offset, trusted):
        ins, pc = item.ins, item.pc
        if not trusted:
            _loop_guard(
                lines,
                f"type(regs[{ins.b}]) in _NUM_TYPES and type(regs[{ins.c}]) in _NUM_TYPES",
                pc,
                offset,
            )
        lines.append(
            f"        regs[{ins.a}] = float(regs[{ins.b}] {symbol} regs[{ins.c}])"
        )
        return True

    return emit


def _make_loop_generic_binop(symbol: str) -> Emitter:
    def emit(lines, item, offset, trusted):
        ins, pc, specialization = item.ins, item.pc, item.specialization
        if specialization == "int":
            _loop_guard(
                lines,
                f"type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int",
                pc,
                offset,
            )
            lines.append(
                f"        regs[{ins.a}] = _i64(regs[{ins.b}] {symbol} regs[{ins.c}])"
            )
            return True
        if specialization == "number":
            _loop_guard(
                lines,
                f"type(regs[{ins.b}]) in _NUM_TYPES and type(regs[{ins.c}]) in _NUM_TYPES",
                pc,
                offset,
            )
            lines.append(f"        _a = regs[{ins.b}]; _b = regs[{ins.c}]")
            lines.append(f"        _v = _a {symbol} _b")
            lines.append("        if type(_a) is int and type(_b) is int:")
            lines.append(f"            regs[{ins.a}] = _i64(_v)")
            lines.append("        else:")
            lines.append(f"            regs[{ins.a}] = _v")
            return True
        return False

    return emit


def _loop_mod(lines, item, offset, trusted):
    ins, pc = item.ins, item.pc
    if item.specialization != "int":
        return False
    _loop_guard(
        lines,
        f"type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int and regs[{ins.c}] != 0",
        pc,
        offset,
    )
    lines.append(f"        regs[{ins.a}] = _i64(regs[{ins.b}] % regs[{ins.c}])")
    return True


def _comparison_guard(ins, specialization: str | None) -> str | None:
    if specialization == "int":
        return f"type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int"
    if specialization == "number":
        return f"type(regs[{ins.b}]) in _NUM_TYPES and type(regs[{ins.c}]) in _NUM_TYPES"
    if specialization == "bytes":
        return f"isinstance(regs[{ins.b}], bytes) and isinstance(regs[{ins.c}], bytes)"
    return None


def _loop_eq(lines, item, offset, trusted):
    ins, pc = item.ins, item.pc
    guard = _comparison_guard(ins, item.specialization)
    if guard is None:
        return False
    _loop_guard(lines, guard, pc, offset)
    # The guard proves this is numeric or bytes equality, for which Python's
    # equality operator has exactly the Lua result. Avoid a helper call here.
    lines.append(f"        regs[{ins.a}] = regs[{ins.b}] == regs[{ins.c}]")
    return True


def _make_loop_ordered(symbol: str) -> Emitter:
    def emit(lines, item, offset, trusted):
        ins, pc = item.ins, item.pc
        guard = _comparison_guard(ins, item.specialization)
        if guard is None:
            return False
        _loop_guard(lines, guard, pc, offset)
        lines.append(f"        regs[{ins.a}] = regs[{ins.b}] {symbol} regs[{ins.c}]")
        return True

    return emit


def _loop_not(lines, item, offset, trusted):
    ins = item.ins
    lines.append(f"        _v = regs[{ins.b}]")
    lines.append(f"        regs[{ins.a}] = (_v is None or _v is False)")
    return True


def _loop_tobool(lines, item, offset, trusted):
    ins = item.ins
    lines.append(f"        _v = regs[{ins.b}]")
    lines.append(f"        regs[{ins.a}] = not (_v is None or _v is False)")
    return True


def _leaf_loadk(lines, item, offset, trusted):
    ins = item.ins
    lines.append(f"    regs[{ins.a}] = consts[{ins.b}]")
    return True


def _leaf_move(lines, item, offset, trusted):
    ins = item.ins
    lines.append(f"    regs[{ins.a}] = regs[{ins.b}]")
    return True


def _leaf_local(lines, item, offset, trusted):
    ins = item.ins
    lines.append(f"    _v = regs[{ins.b}]")
    lines.append(f"    regs[{ins.a}] = _v")
    lines.append(f"    if {ins.a} in cells:")
    lines.append(f"        cells[{ins.a}] = _Cell(_v)")
    return True


def _make_leaf_int_binop(symbol: str) -> Emitter:
    def emit(lines, item, offset, trusted):
        ins = item.ins
        if not trusted:
            _leaf_guard(
                lines,
                f"type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int",
            )
        lines.append(f"    regs[{ins.a}] = _i64(regs[{ins.b}] {symbol} regs[{ins.c}])")
        return True

    return emit


def _make_leaf_float_binop(symbol: str) -> Emitter:
    def emit(lines, item, offset, trusted):
        ins = item.ins
        if not trusted:
            _leaf_guard(
                lines,
                f"type(regs[{ins.b}]) in _NUM_TYPES and type(regs[{ins.c}]) in _NUM_TYPES",
            )
        lines.append(f"    regs[{ins.a}] = float(regs[{ins.b}] {symbol} regs[{ins.c}])")
        return True

    return emit


def _make_leaf_generic_binop(symbol: str) -> Emitter:
    def emit(lines, item, offset, trusted):
        ins = item.ins
        lines.append(f"    _a = regs[{ins.b}]; _b = regs[{ins.c}]")
        lines.append("    if type(_a) is int and type(_b) is int:")
        lines.append(f"        regs[{ins.a}] = _i64(_a {symbol} _b)")
        lines.append("    elif type(_a) in _NUM_TYPES and type(_b) in _NUM_TYPES:")
        lines.append(f"        regs[{ins.a}] = _a {symbol} _b")
        lines.append("    else:")
        lines.append("        return _DEOPT")
        return True

    return emit


def _leaf_mod(lines, item, offset, trusted):
    ins = item.ins
    lines.append(f"    _a = regs[{ins.b}]; _b = regs[{ins.c}]")
    lines.append("    if type(_a) is not int or type(_b) is not int or _b == 0: return _DEOPT")
    lines.append(f"    regs[{ins.a}] = _i64(_a % _b)")
    return True


def _leaf_eq(lines, item, offset, trusted):
    ins = item.ins
    lines.append(f"    _a = regs[{ins.b}]; _b = regs[{ins.c}]")
    lines.append("    if isinstance(_a, _LuaTable) and isinstance(_b, _LuaTable): return _DEOPT")
    lines.append(f"    regs[{ins.a}] = _lua_equal(_a, _b)")
    return True


def _make_leaf_ordered(symbol: str) -> Emitter:
    def emit(lines, item, offset, trusted):
        ins = item.ins
        lines.append(f"    _a = regs[{ins.b}]; _b = regs[{ins.c}]")
        lines.append(
            "    if not ((type(_a) in _NUM_TYPES and type(_b) in _NUM_TYPES) or (isinstance(_a, bytes) and isinstance(_b, bytes))): return _DEOPT"
        )
        lines.append(f"    regs[{ins.a}] = _a {symbol} _b")
        return True

    return emit


def _leaf_not(lines, item, offset, trusted):
    ins = item.ins
    lines.append(f"    _v = regs[{ins.b}]")
    lines.append(f"    regs[{ins.a}] = (_v is None or _v is False)")
    return True


def _leaf_tobool(lines, item, offset, trusted):
    ins = item.ins
    lines.append(f"    _v = regs[{ins.b}]")
    lines.append(f"    regs[{ins.a}] = not (_v is None or _v is False)")
    return True


def _leaf_guard_type(lines, item, offset, trusted):
    ins = item.ins
    lines.append(f"    if not _type_matches(consts[{ins.b}], regs[{ins.a}]): return _DEOPT")
    return True


def _leaf_return(lines, item, offset, trusted):
    ins = item.ins
    if ins.b <= 0:
        lines.append("    return ()")
    else:
        values = ", ".join(f"regs[{ins.a + i}]" for i in range(ins.b))
        if ins.b == 1:
            values += ","
        lines.append(f"    return ({values})")
    return True


def _dense_table(entries: dict[Op, Emitter]) -> tuple[Emitter | None, ...]:
    table: list[Emitter | None] = [None] * (max(op.value for op in Op) + 1)
    for op, emitter in entries.items():
        table[op.value] = emitter
    return tuple(table)


LOOP_EMITTERS = _dense_table(
    {
        Op.LOADK: _loop_loadk,
        Op.MOVE: _loop_move,
        Op.LOCAL: _loop_local,
        Op.NEWTABLE: _loop_newtable,
        Op.GETTABLE: _loop_gettable,
        Op.SETTABLE: _loop_settable,
        Op.ADD_I: _make_loop_int_binop("+"),
        Op.SUB_I: _make_loop_int_binop("-"),
        Op.MUL_I: _make_loop_int_binop("*"),
        Op.ADD_F: _make_loop_float_binop("+"),
        Op.SUB_F: _make_loop_float_binop("-"),
        Op.MUL_F: _make_loop_float_binop("*"),
        Op.ADD: _make_loop_generic_binop("+"),
        Op.SUB: _make_loop_generic_binop("-"),
        Op.MUL: _make_loop_generic_binop("*"),
        Op.MOD: _loop_mod,
        Op.EQ: _loop_eq,
        Op.LT: _make_loop_ordered("<"),
        Op.LE: _make_loop_ordered("<="),
        Op.NOT: _loop_not,
        Op.TOBOOL: _loop_tobool,
    }
)


LEAF_EMITTERS = _dense_table(
    {
        Op.LOADK: _leaf_loadk,
        Op.MOVE: _leaf_move,
        Op.LOCAL: _leaf_local,
        Op.ADD_I: _make_leaf_int_binop("+"),
        Op.SUB_I: _make_leaf_int_binop("-"),
        Op.MUL_I: _make_leaf_int_binop("*"),
        Op.ADD_F: _make_leaf_float_binop("+"),
        Op.SUB_F: _make_leaf_float_binop("-"),
        Op.MUL_F: _make_leaf_float_binop("*"),
        Op.ADD: _make_leaf_generic_binop("+"),
        Op.SUB: _make_leaf_generic_binop("-"),
        Op.MUL: _make_leaf_generic_binop("*"),
        Op.MOD: _leaf_mod,
        Op.EQ: _leaf_eq,
        Op.LT: _make_leaf_ordered("<"),
        Op.LE: _make_leaf_ordered("<="),
        Op.NOT: _leaf_not,
        Op.TOBOOL: _leaf_tobool,
        Op.GUARD: _leaf_guard_type,
        Op.RETURN: _leaf_return,
    }
)


class _InlineI64Assignments(ast.NodeTransformer):
    """Inline the tiny i64 helper after source lowering but before compile().

    The transform is deliberately statement-based, so the wrapped expression is
    evaluated exactly once. This avoids the side-effect duplication hazard of
    naive expression substitution while removing a Python helper call from the
    generated hot path.
    """

    def __init__(self) -> None:
        self.counter = 0

    def visit_Assign(self, node: ast.Assign):
        node = self.generic_visit(node)
        value = node.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "_i64"
            and len(value.args) == 1
            and not value.keywords
        ):
            return node

        temp = f"_i64v_{self.counter}"
        self.counter += 1
        temp_store = ast.Name(id=temp, ctx=ast.Store())
        temp_load = ast.Name(id=temp, ctx=ast.Load())
        masked = ast.Assign(
            targets=[temp_store],
            value=ast.BinOp(
                left=value.args[0],
                op=ast.BitAnd(),
                right=ast.Name(id="_MASK64", ctx=ast.Load()),
            ),
        )
        wrapped = ast.IfExp(
            test=ast.BinOp(
                left=temp_load,
                op=ast.BitAnd(),
                right=ast.Name(id="_SIGN64", ctx=ast.Load()),
            ),
            body=ast.BinOp(
                left=ast.Name(id=temp, ctx=ast.Load()),
                op=ast.Sub(),
                right=ast.Name(id="_TWO64", ctx=ast.Load()),
            ),
            orelse=ast.Name(id=temp, ctx=ast.Load()),
        )
        return [masked, ast.Assign(targets=node.targets, value=wrapped)]


def optimize_generated_ast(tree: ast.AST) -> ast.AST:
    tree = _InlineI64Assignments().visit(tree)
    tree = optimize_semantic_helpers(tree)
    ast.fix_missing_locations(tree)
    return tree


def generated_namespace(base: dict[str, object]) -> dict[str, object]:
    namespace = dict(base)
    namespace.update(
        {
            "_MASK64": MASK64,
            "_SIGN64": SIGN64,
            "_TWO64": _TWO64,
        }
    )
    return namespace
