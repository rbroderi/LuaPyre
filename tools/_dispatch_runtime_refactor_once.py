from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPDISPATCH = ROOT / "src/luapyre/opdispatch.py"
VM = ROOT / "src/luapyre/vm.py"
WORKFLOW = ROOT / ".github/workflows/_dispatch-runtime-refactor-once.yml"
SELF = Path(__file__)


def replace_top_level_function(source: str, name: str, replacement: str) -> str:
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    lines = source.splitlines()
    lines[node.lineno - 1:node.end_lineno] = replacement.rstrip().splitlines()
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


def remove_top_level_functions(source: str, names: set[str]) -> str:
    tree = ast.parse(source)
    nodes = [
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in names
    ]
    lines = source.splitlines()
    for node in sorted(nodes, key=lambda item: item.lineno, reverse=True):
        del lines[node.lineno - 1:node.end_lineno]
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


def remove_class_methods(source: str, class_name: str, names: set[str]) -> str:
    tree = ast.parse(source)
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name)
    nodes = [
        item for item in cls.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in names
    ]
    lines = source.splitlines()
    for node in sorted(nodes, key=lambda item: item.lineno, reverse=True):
        del lines[node.lineno - 1:node.end_lineno]
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")

op = OPDISPATCH.read_text(encoding="utf-8")

# Comparison handlers need equality semantics directly now that VM no longer
# receives an opcode and redispatches it.
op = op.replace(
    "    MultiValue, coerce_lua_integer, i64, parse_lua_number,\n    static_value_type, truthy, type_matches,\n",
    "    MultiValue, coerce_lua_integer, i64, lua_equal, parse_lua_number,\n    static_value_type, truthy, type_matches,\n",
    1,
)

arith = r'''def _int_add(vm, frames, frame, ins, regs, constants):
    a, b = _need_number(regs[ins.b]), _need_number(regs[ins.c])
    regs[ins.a] = i64(a + b)


def _int_sub(vm, frames, frame, ins, regs, constants):
    a, b = _need_number(regs[ins.b]), _need_number(regs[ins.c])
    regs[ins.a] = i64(a - b)


def _int_mul(vm, frames, frame, ins, regs, constants):
    a, b = _need_number(regs[ins.b]), _need_number(regs[ins.c])
    regs[ins.a] = i64(a * b)


def _float_add(vm, frames, frame, ins, regs, constants):
    a, b = _need_number(regs[ins.b]), _need_number(regs[ins.c])
    regs[ins.a] = float(a + b)


def _float_sub(vm, frames, frame, ins, regs, constants):
    a, b = _need_number(regs[ins.b]), _need_number(regs[ins.c])
    regs[ins.a] = float(a - b)


def _float_mul(vm, frames, frame, ins, regs, constants):
    a, b = _need_number(regs[ins.b]), _need_number(regs[ins.c])
    regs[ins.a] = float(a * b)


def _float_divide(a, b):
    a = float(a)
    b = float(b)
    if b != 0.0:
        return a / b
    if a == 0.0:
        return math.nan
    return math.copysign(math.inf, a * (1.0 if math.copysign(1.0, b) > 0 else -1.0))


def _float_modulo(a, b):
    a = float(a)
    b = float(b)
    try:
        value = math.fmod(a, b)
    except ValueError:
        return math.nan
    if (value > 0.0 and b < 0.0) or (value < 0.0 and b > 0.0):
        value += b
    return value


def _float_power(a, b):
    a = float(a)
    b = float(b)
    try:
        return math.pow(a, b)
    except ValueError:
        if a == 0.0 and b < 0.0:
            negative = (
                math.copysign(1.0, a) < 0.0
                and math.isfinite(b)
                and b.is_integer()
                and int(b) & 1
            )
            return -math.inf if negative else math.inf
        return math.nan
    except OverflowError:
        negative = a < 0.0 and math.isfinite(b) and b.is_integer() and int(b) & 1
        return -math.inf if negative else math.inf


def _arith_metamethod(vm, frames, frame, name, a, b, dest):
    tm = vm._first_tm(a, b, name)
    if tm is None:
        raise LuaRuntimeError(
            f"attempt to perform arithmetic on a {static_value_type(a).name} value"
        )
    vm._invoke(frames, frame, tm, [a, b], dest, 1)


def _add(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    na, nb = parse_lua_number(a), parse_lua_number(b)
    if na is None or nb is None:
        _arith_metamethod(vm, frames, frame, b"__add", a, b, ins.a)
        return
    value = na + nb
    regs[ins.a] = i64(value) if type(na) is int and type(nb) is int else value


def _sub(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    na, nb = parse_lua_number(a), parse_lua_number(b)
    if na is None or nb is None:
        _arith_metamethod(vm, frames, frame, b"__sub", a, b, ins.a)
        return
    value = na - nb
    regs[ins.a] = i64(value) if type(na) is int and type(nb) is int else value


def _mul(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    na, nb = parse_lua_number(a), parse_lua_number(b)
    if na is None or nb is None:
        _arith_metamethod(vm, frames, frame, b"__mul", a, b, ins.a)
        return
    value = na * nb
    regs[ins.a] = i64(value) if type(na) is int and type(nb) is int else value


def _div(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    na, nb = parse_lua_number(a), parse_lua_number(b)
    if na is None or nb is None:
        _arith_metamethod(vm, frames, frame, b"__div", a, b, ins.a)
        return
    regs[ins.a] = _float_divide(na, nb)


def _idiv(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    na, nb = parse_lua_number(a), parse_lua_number(b)
    if na is None or nb is None:
        _arith_metamethod(vm, frames, frame, b"__idiv", a, b, ins.a)
        return
    if nb == 0:
        if type(na) is int and type(nb) is int:
            raise LuaRuntimeError("attempt to divide by zero")
        regs[ins.a] = _float_divide(na, nb)
        return
    if type(na) is int and type(nb) is int:
        regs[ins.a] = i64(na // nb)
    else:
        regs[ins.a] = float(math.floor(float(na) / float(nb)))


def _mod(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    na, nb = parse_lua_number(a), parse_lua_number(b)
    if na is None or nb is None:
        _arith_metamethod(vm, frames, frame, b"__mod", a, b, ins.a)
        return
    if type(na) is int and type(nb) is int:
        if nb == 0:
            raise LuaRuntimeError("attempt to perform 'n%0'")
        regs[ins.a] = i64(na % nb)
    else:
        regs[ins.a] = _float_modulo(na, nb)


def _pow(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    na, nb = parse_lua_number(a), parse_lua_number(b)
    if na is None or nb is None:
        _arith_metamethod(vm, frames, frame, b"__pow", a, b, ins.a)
        return
    regs[ins.a] = _float_power(na, nb)


def _band(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    ai, bi = coerce_lua_integer(a), coerce_lua_integer(b)
    if ai is None or bi is None:
        _bitwise_error(vm, frames, frame, Op.BAND, a, b, ins.a)
        return
    regs[ins.a] = i64(ai & bi)


def _bor(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    ai, bi = coerce_lua_integer(a), coerce_lua_integer(b)
    if ai is None or bi is None:
        _bitwise_error(vm, frames, frame, Op.BOR, a, b, ins.a)
        return
    regs[ins.a] = i64(ai | bi)


def _bxor(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    ai, bi = coerce_lua_integer(a), coerce_lua_integer(b)
    if ai is None or bi is None:
        _bitwise_error(vm, frames, frame, Op.BXOR, a, b, ins.a)
        return
    regs[ins.a] = i64(ai ^ bi)


def _shl(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    ai, bi = coerce_lua_integer(a), coerce_lua_integer(b)
    if ai is None or bi is None:
        _bitwise_error(vm, frames, frame, Op.SHL, a, b, ins.a)
        return
    regs[ins.a] = _shift(ai, bi, left=True)


def _shr(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    ai, bi = coerce_lua_integer(a), coerce_lua_integer(b)
    if ai is None or bi is None:
        _bitwise_error(vm, frames, frame, Op.SHR, a, b, ins.a)
        return
    regs[ins.a] = _shift(ai, bi, left=False)
'''

# Replace the three redispatching arithmetic functions as one contiguous block.
tree = ast.parse(op)
functions = {
    node.name: node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in {"_int_arith", "_float_arith", "_generic_arith"}
}
start = min(node.lineno for node in functions.values()) - 1
end = max(node.end_lineno for node in functions.values())
op_lines = op.splitlines()
op_lines[start:end] = arith.rstrip().splitlines()
op = "\n".join(op_lines) + ("\n" if op.endswith("\n") else "")

compare = r'''def _eq(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    if lua_equal(a, b):
        regs[ins.a] = True
        return
    if not (isinstance(a, LuaTable) and isinstance(b, LuaTable)):
        regs[ins.a] = False
        return
    tm = vm._first_tm(a, b, b"__eq")
    if tm is None:
        regs[ins.a] = False
        return
    vm._invoke(frames, frame, tm, [a, b], ins.a, 1)


def _lt(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    if _is_number(a) and _is_number(b):
        regs[ins.a] = a < b
        return
    if isinstance(a, bytes) and isinstance(b, bytes):
        regs[ins.a] = a < b
        return
    tm = vm._first_tm(a, b, b"__lt")
    if tm is None:
        raise LuaRuntimeError("attempt to compare incompatible values")
    vm._invoke(frames, frame, tm, [a, b], ins.a, 1)


def _le(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    if _is_number(a) and _is_number(b):
        regs[ins.a] = a <= b
        return
    if isinstance(a, bytes) and isinstance(b, bytes):
        regs[ins.a] = a <= b
        return
    tm = vm._first_tm(a, b, b"__le")
    if tm is None:
        raise LuaRuntimeError("attempt to compare incompatible values")
    vm._invoke(frames, frame, tm, [a, b], ins.a, 1)
'''
op = replace_top_level_function(op, "_compare", compare)

calls = r'''def _call(vm, frames, frame, ins, regs, constants):
    fn = regs[ins.b]
    args = [regs[ins.c + i] for i in range(ins.d)]
    return vm._invoke(frames, frame, fn, args, ins.a, ins.e, tail=False)


def _callv(vm, frames, frame, ins, regs, constants):
    fn = regs[ins.b]
    args = [regs[ins.c + i] for i in range(ins.d)]
    mv = regs[ins.e]
    args.extend(mv.values if isinstance(mv, MultiValue) else (mv,))
    return vm._invoke(frames, frame, fn, args, ins.a, -1, tail=False)


def _tailcall(vm, frames, frame, ins, regs, constants):
    fn = regs[ins.b]
    args = [regs[ins.c + i] for i in range(ins.d)]
    returned = vm._invoke(frames, frame, fn, args, ins.a, -1, tail=True)
    return returned if returned is not None else None


def _tailcallv(vm, frames, frame, ins, regs, constants):
    fn = regs[ins.b]
    args = [regs[ins.c + i] for i in range(ins.d)]
    mv = regs[ins.e]
    args.extend(mv.values if isinstance(mv, MultiValue) else (mv,))
    returned = vm._invoke(frames, frame, fn, args, ins.a, -1, tail=True)
    return returned if returned is not None else None
'''
op = replace_top_level_function(op, "_call", calls)

old_table = '''    Op.ADD: _generic_arith,
    Op.ADD_I: _int_arith,
    Op.ADD_F: _float_arith,
    Op.SUB: _generic_arith,
    Op.SUB_I: _int_arith,
    Op.SUB_F: _float_arith,
    Op.MUL: _generic_arith,
    Op.MUL_I: _int_arith,
    Op.MUL_F: _float_arith,
    Op.DIV: _generic_arith,
    Op.IDIV: _generic_arith,
    Op.MOD: _generic_arith,
    Op.POW: _generic_arith,
    Op.BAND: _generic_arith,
    Op.BOR: _generic_arith,
    Op.BXOR: _generic_arith,
    Op.SHL: _generic_arith,
    Op.SHR: _generic_arith,'''
new_table = '''    Op.ADD: _add,
    Op.ADD_I: _int_add,
    Op.ADD_F: _float_add,
    Op.SUB: _sub,
    Op.SUB_I: _int_sub,
    Op.SUB_F: _float_sub,
    Op.MUL: _mul,
    Op.MUL_I: _int_mul,
    Op.MUL_F: _float_mul,
    Op.DIV: _div,
    Op.IDIV: _idiv,
    Op.MOD: _mod,
    Op.POW: _pow,
    Op.BAND: _band,
    Op.BOR: _bor,
    Op.BXOR: _bxor,
    Op.SHL: _shl,
    Op.SHR: _shr,'''
if old_table not in op:
    raise RuntimeError("arithmetic opcode table block changed")
op = op.replace(old_table, new_table, 1)

op = op.replace(
    "    Op.EQ: _compare,\n    Op.LT: _compare,\n    Op.LE: _compare,",
    "    Op.EQ: _eq,\n    Op.LT: _lt,\n    Op.LE: _le,",
    1,
)
op = op.replace(
    "    Op.CALL: _call,\n    Op.CALLV: _call,\n    Op.TAILCALL: _call,\n    Op.TAILCALLV: _call,",
    "    Op.CALL: _call,\n    Op.CALLV: _callv,\n    Op.TAILCALL: _tailcall,\n    Op.TAILCALLV: _tailcallv,",
    1,
)

ast.parse(op)
OPDISPATCH.write_text(op, encoding="utf-8")

vm = VM.read_text(encoding="utf-8")
vm = remove_class_methods(vm, "VM", {"_arith_primitive", "_generic_binary", "_compare"})
vm = remove_top_level_functions(
    vm,
    {"_need_number", "_to_int", "_float_div", "_float_mod", "_float_pow", "_shift_left", "_shift_right", "_to_lua_string"},
)
vm = vm.replace("import math\n", "", 1)
vm = vm.replace(
    "from .values import MultiValue, i64, lua_equal, static_value_type, truthy, type_matches\n",
    "from .values import MultiValue, i64, static_value_type, truthy, type_matches\n",
    1,
)
ast.parse(vm)
VM.write_text(vm, encoding="utf-8")

# Extend the architecture regression to prevent reintroducing second-level
# opcode switches inside already-dispatched hot handlers.
test_path = ROOT / "tests/test_dispatch_architecture.py"
existing = test_path.read_text(encoding="utf-8")
existing += r'''


def test_hot_runtime_opcodes_have_specialized_handlers():
    from luapyre.bytecode import Op
    from luapyre.opdispatch import OPCODE_HANDLERS

    groups = [
        (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.IDIV, Op.MOD, Op.POW),
        (Op.BAND, Op.BOR, Op.BXOR, Op.SHL, Op.SHR),
        (Op.ADD_I, Op.SUB_I, Op.MUL_I),
        (Op.ADD_F, Op.SUB_F, Op.MUL_F),
        (Op.EQ, Op.LT, Op.LE),
        (Op.CALL, Op.CALLV, Op.TAILCALL, Op.TAILCALLV),
    ]
    for group in groups:
        handlers = [OPCODE_HANDLERS[opcode] for opcode in group]
        assert len(set(handlers)) == len(handlers)
        for handler in handlers:
            assert "ins.op" not in inspect.getsource(handler)
'''
test_path.write_text(existing, encoding="utf-8")

SELF.unlink()
if WORKFLOW.exists():
    WORKFLOW.unlink()
