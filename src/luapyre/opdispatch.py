from __future__ import annotations

import math

from .bytecode import Cell, Closure, Op
from .errors import LuaRuntimeError
from .table import LuaTable
from .values import (
    MultiValue, coerce_lua_integer, i64, lua_equal, parse_lua_number,
    static_value_type, truthy, type_matches,
)


_UINT_MASK = (1 << 64) - 1
_INT_MIN = -(1 << 63)
_INT_MAX = (1 << 63) - 1


def _is_number(value):
    return type(value) in (int, float)


def _need_number(value):
    if not _is_number(value):
        raise LuaRuntimeError(
            f"attempt to perform arithmetic on a {static_value_type(value).name} value"
        )
    return value


def _to_int(value):
    integer = coerce_lua_integer(value)
    if integer is not None:
        return integer
    if type(value) is float and math.isinf(value):
        raise LuaRuntimeError("number (field 'huge') has no integer representation")
    raise LuaRuntimeError("number has no integer representation")


def _shift(value, count, *, left):
    """Lua logical shift without negating ``math.mininteger``."""
    value &= _UINT_MASK
    if count >= 64 or count <= -64:
        return 0
    if count < 0:
        left = not left
        count = -count
    result = (value << count) & _UINT_MASK if left else value >> count
    return i64(result)


def _bitwise_error(vm, frames, frame, op, a, b, dest):
    name = {
        Op.BAND: "band", Op.BOR: "bor", Op.BXOR: "bxor",
        Op.SHL: "shl", Op.SHR: "shr",
    }[op]
    tm = vm._first_tm(a, b, vm.ARITH_TM[op])
    if tm is not None:
        vm._invoke(frames, frame, tm, [a, b], dest, 1)
        return
    bad = a if coerce_lua_integer(a) is None else b
    if type(bad) is float:
        if math.isinf(bad):
            raise LuaRuntimeError("number (field 'huge') has no integer representation")
        raise LuaRuntimeError("number has no integer representation")
    raise LuaRuntimeError(
        f"attempt to perform '{name}' on a {static_value_type(bad).name} value"
    )


def _to_lua_string(value):
    if isinstance(value, bytes):
        return value
    if type(value) is int:
        return str(value).encode("ascii")
    if type(value) is float:
        return repr(value).encode("ascii")
    raise LuaRuntimeError("attempt to concatenate a non-string value")


def _loadk(vm, frames, frame, ins, regs, constants):
    regs[ins.a] = constants[ins.b]


def _move(vm, frames, frame, ins, regs, constants):
    regs[ins.a] = regs[ins.b]


def _local(vm, frames, frame, ins, regs, constants):
    value = regs[ins.b]
    regs[ins.a] = value
    if ins.a in frame.cells:
        frame.cells[ins.a] = Cell(value)


def _getcell(vm, frames, frame, ins, regs, constants):
    cell = frame.cells.get(ins.b)
    regs[ins.a] = regs[ins.b] if cell is None else cell.value


def _setcell(vm, frames, frame, ins, regs, constants):
    cell = frame.cells.get(ins.a)
    if cell is None:
        cell = frame.cells[ins.a] = Cell(regs[ins.a])
    cell.value = regs[ins.b]
    regs[ins.a] = regs[ins.b]


def _getupval(vm, frames, frame, ins, regs, constants):
    regs[ins.a] = frame.closure.upvalues[ins.b].value


def _setupval(vm, frames, frame, ins, regs, constants):
    frame.closure.upvalues[ins.a].value = regs[ins.b]


def _closure(vm, frames, frame, ins, regs, constants):
    child = frame.proto.children[ins.b]
    upvalues = []
    for desc in child.upvalues:
        if desc.kind == "local":
            cell = frame.cells.get(desc.index)
            if cell is None:
                cell = frame.cells[desc.index] = Cell(regs[desc.index])
            upvalues.append(cell)
        else:
            upvalues.append(frame.closure.upvalues[desc.index])
    regs[ins.a] = Closure(child, upvalues, frame.closure.env)


def _getglobal(vm, frames, frame, ins, regs, constants):
    regs[ins.a] = frame.closure.env.rawget(constants[ins.b])


def _setglobal(vm, frames, frame, ins, regs, constants):
    frame.closure.env.rawset(constants[ins.b], regs[ins.a])


def _newtable(vm, frames, frame, ins, regs, constants):
    regs[ins.a] = LuaTable()


def _gettable(vm, frames, frame, ins, regs, constants):
    vm._gettable(frames, frame, regs[ins.b], regs[ins.c], ins.a)


def _settable(vm, frames, frame, ins, regs, constants):
    vm._settable(frames, frame, regs[ins.a], regs[ins.b], regs[ins.c])


def _setlistv(vm, frames, frame, ins, regs, constants):
    table = regs[ins.a]
    mv = regs[ins.c]
    if not isinstance(table, LuaTable) or not isinstance(mv, MultiValue):
        raise LuaRuntimeError("invalid table list expansion")
    for offset, value in enumerate(mv.values):
        table.rawset(ins.b + offset, value)


def _int_add(vm, frames, frame, ins, regs, constants):
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


def _neg(vm, frames, frame, ins, regs, constants):
    value = regs[ins.b]
    number = parse_lua_number(value)
    if number is not None:
        regs[ins.a] = i64(-number) if type(number) is int else -number
        return
    tm = vm._tm(value, b"__unm")
    if tm is None:
        raise LuaRuntimeError(
            f"attempt to perform arithmetic on a {static_value_type(value).name} value"
        )
    vm._invoke(frames, frame, tm, [value], ins.a, 1)


def _bnot(vm, frames, frame, ins, regs, constants):
    value = regs[ins.b]
    integer = coerce_lua_integer(value)
    if integer is not None:
        regs[ins.a] = i64(~integer)
        return
    tm = vm._tm(value, b"__bnot")
    if tm is None:
        raise LuaRuntimeError(
            f"attempt to perform 'bnot' on a {static_value_type(value).name} value"
        )
    vm._invoke(frames, frame, tm, [value], ins.a, 1)


def _concat(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    try:
        regs[ins.a] = _to_lua_string(a) + _to_lua_string(b)
    except LuaRuntimeError:
        tm = vm._first_tm(a, b, b"__concat")
        if tm is None:
            raise
        vm._invoke(frames, frame, tm, [a, b], ins.a, 1)


def _len(vm, frames, frame, ins, regs, constants):
    value = regs[ins.b]
    if isinstance(value, bytes):
        regs[ins.a] = len(value)
        return
    if isinstance(value, LuaTable):
        tm = vm._tm(value, b"__len")
        if tm is None:
            regs[ins.a] = value.rawlen()
        else:
            vm._invoke(frames, frame, tm, [value], ins.a, 1)
        return
    tm = vm._tm(value, b"__len")
    if tm is None:
        raise LuaRuntimeError(
            f"attempt to get length of a {static_value_type(value).name} value"
        )
    vm._invoke(frames, frame, tm, [value], ins.a, 1)


def _not(vm, frames, frame, ins, regs, constants):
    regs[ins.a] = not truthy(regs[ins.b])


def _tobool(vm, frames, frame, ins, regs, constants):
    regs[ins.a] = truthy(regs[ins.b])


def _eq(vm, frames, frame, ins, regs, constants):
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


def _jmp(vm, frames, frame, ins, regs, constants):
    frame.pc = ins.a


def _jmpif(vm, frames, frame, ins, regs, constants):
    if truthy(regs[ins.b]):
        frame.pc = ins.a


def _jmpifnot(vm, frames, frame, ins, regs, constants):
    if not truthy(regs[ins.b]):
        frame.pc = ins.a


def _jmpifnil(vm, frames, frame, ins, regs, constants):
    if regs[ins.b] is None:
        frame.pc = ins.a


def _forprep(vm, frames, frame, ins, regs, constants):
    if not vm._forprep(regs, ins):
        frame.pc = ins.d


def _forloop(vm, frames, frame, ins, regs, constants):
    if vm._forloop(regs, ins):
        frame.pc = ins.d


def _pforprep(vm, frames, frame, ins, regs, constants):
    init, limit, step = regs[ins.a], regs[ins.a + 1], regs[ins.a + 2]
    if type(init) is int and type(step) is int:
        if step == 0:
            raise LuaRuntimeError("'for' step is zero")
        if type(limit) is int:
            ilimit = limit
        elif type(limit) is float and math.isfinite(limit):
            ilimit = math.ceil(limit) if step < 0 else math.floor(limit)
            if ilimit > _INT_MAX:
                if step < 0:
                    frame.pc = ins.d
                    return
                ilimit = _INT_MAX
            elif ilimit < _INT_MIN:
                if step > 0:
                    frame.pc = ins.d
                    return
                ilimit = _INT_MIN
        else:
            raise LuaRuntimeError("'for' limit must be a number")
        if (step > 0 and init > ilimit) or (step < 0 and init < ilimit):
            frame.pc = ins.d
            return
        if step > 0:
            count = ((ilimit & _UINT_MASK) - (init & _UINT_MASK)) & _UINT_MASK
            if step != 1:
                count //= step & _UINT_MASK
        else:
            count = ((init & _UINT_MASK) - (ilimit & _UINT_MASK)) & _UINT_MASK
            count //= -(step + 1) + 1
        regs[ins.a] = i64(count)
        regs[ins.a + 1] = step
        regs[ins.a + 2] = init
        return

    if not all(_is_number(value) for value in (init, limit, step)):
        raise LuaRuntimeError("'for' limit must be a number")
    init, limit, step = float(init), float(limit), float(step)
    if step == 0.0:
        raise LuaRuntimeError("'for' step is zero")
    if (step > 0 and init > limit) or (step < 0 and init < limit):
        frame.pc = ins.d
        return
    regs[ins.a] = limit
    regs[ins.a + 1] = step
    regs[ins.a + 2] = init


def _pforloop(vm, frames, frame, ins, regs, constants):
    if type(regs[ins.a + 1]) is int:
        count = regs[ins.a] & _UINT_MASK
        if count > 0:
            regs[ins.a] = i64(count - 1)
            regs[ins.a + 2] = i64(regs[ins.a + 2] + regs[ins.a + 1])
            frame.pc = ins.d
        return
    limit = float(regs[ins.a])
    step = float(regs[ins.a + 1])
    idx = float(regs[ins.a + 2]) + step
    if (step > 0 and idx <= limit) or (step < 0 and idx >= limit):
        regs[ins.a + 2] = idx
        frame.pc = ins.d


def _ptforprep(vm, frames, frame, ins, regs, constants):
    regs[ins.a + 2], regs[ins.a + 3] = regs[ins.a + 3], regs[ins.a + 2]
    value = regs[ins.a + 2]
    if value is not None and value is not False:
        if vm._tm(value, b"__close") is None:
            raise LuaRuntimeError("variable got a non-closable value")
        frame.puc_close_stack.append((ins.a + 2, value))
    frame.pc = ins.d


def _ptforloop(vm, frames, frame, ins, regs, constants):
    if regs[ins.a + 3] is not None:
        frame.pc = ins.d


def _guard(vm, frames, frame, ins, regs, constants):
    expected = constants[ins.b]
    if not type_matches(expected, regs[ins.a]):
        raise LuaRuntimeError(
            f"expected {expected}, got {static_value_type(regs[ins.a]).name}"
        )


def _checknil(vm, frames, frame, ins, regs, constants):
    if regs[ins.a] is None:
        return
    name = constants[ins.b]
    if isinstance(name, bytes):
        name = name.decode("utf-8", "replace")
    raise LuaRuntimeError(f"global '{name}' already has a value")


def _tbc(vm, frames, frame, ins, regs, constants):
    value = regs[ins.a]
    if value is None or value is False:
        return
    if vm._tm(value, b"__close") is None:
        raise LuaRuntimeError("variable got a non-closable value")
    frame.close_stack.append(value)


def _close(vm, frames, frame, ins, regs, constants):
    if ins.a < 0 or ins.a > len(frame.close_stack):
        raise LuaRuntimeError("invalid close depth")
    frame.pending_close_target = ins.a


def _ptbc(vm, frames, frame, ins, regs, constants):
    value = regs[ins.a]
    if value is None or value is False:
        return
    if vm._tm(value, b"__close") is None:
        raise LuaRuntimeError("variable got a non-closable value")
    frame.puc_close_stack.append((ins.a, value))


def _pclose(vm, frames, frame, ins, regs, constants):
    frame.pending_puc_close_reg = ins.a


def _native_varargs(frame, regs):
    named = frame.proto.vararg_name_reg
    if named < 0:
        return frame.varargs
    table = regs[named]
    if not isinstance(table, LuaTable):
        raise LuaRuntimeError("no proper 'n' field in vararg table")
    count = table.rawget(b"n")
    if type(count) is not int or count < 0 or count > 1_000_000:
        raise LuaRuntimeError("no proper 'n' field in vararg table")
    return tuple(table.rawget(index) for index in range(1, count + 1))


def _vararg(vm, frames, frame, ins, regs, constants):
    values = _native_varargs(frame, regs)
    if ins.b == -1:
        regs[ins.a] = MultiValue(tuple(values))
        return
    for i in range(ins.b):
        regs[ins.a + i] = values[i] if i < len(values) else None


def _pvararg(vm, frames, frame, ins, regs, constants):
    if ins.c >= 0:
        table = regs[ins.c]
        if not isinstance(table, LuaTable):
            raise LuaRuntimeError("invalid named vararg table")
        count = table.rawget(b"n")
        if type(count) is not int or count < 0:
            raise LuaRuntimeError("invalid named vararg count")
        values = tuple(table.rawget(i) for i in range(1, count + 1))
    else:
        values = frame.varargs
    if ins.b == -1:
        regs[ins.a] = MultiValue(tuple(values))
        return
    for i in range(ins.b):
        regs[ins.a + i] = values[i] if i < len(values) else None


def _pgetvarg(vm, frames, frame, ins, regs, constants):
    key = regs[ins.b]
    if isinstance(key, bytes) and key == b"n":
        regs[ins.a] = len(frame.varargs)
        return
    try:
        index = _to_int(key)
    except LuaRuntimeError:
        regs[ins.a] = None
        return
    regs[ins.a] = frame.varargs[index - 1] if 1 <= index <= len(frame.varargs) else None


def _unpack(vm, frames, frame, ins, regs, constants):
    mv = regs[ins.b]
    values = mv.values if isinstance(mv, MultiValue) else (mv,)
    for i in range(ins.c):
        regs[ins.a + i] = values[i] if i < len(values) else None


def _call(vm, frames, frame, ins, regs, constants):
    fn = regs[ins.b]
    args = [regs[ins.c + i] for i in range(ins.d)]
    return vm._invoke_site(frames, frame, fn, args, ins.a, ins.e, tail=False)


def _callv(vm, frames, frame, ins, regs, constants):
    fn = regs[ins.b]
    args = [regs[ins.c + i] for i in range(ins.d)]
    mv = regs[ins.e]
    args.extend(mv.values if isinstance(mv, MultiValue) else (mv,))
    return vm._invoke_site(frames, frame, fn, args, ins.a, -1, tail=False)


def _tailcall(vm, frames, frame, ins, regs, constants):
    fn = regs[ins.b]
    args = [regs[ins.c + i] for i in range(ins.d)]
    returned = vm._invoke_site(frames, frame, fn, args, ins.a, -1, tail=True)
    return returned if returned is not None else None


def _tailcallv(vm, frames, frame, ins, regs, constants):
    fn = regs[ins.b]
    args = [regs[ins.c + i] for i in range(ins.d)]
    mv = regs[ins.e]
    args.extend(mv.values if isinstance(mv, MultiValue) else (mv,))
    returned = vm._invoke_site(frames, frame, fn, args, ins.a, -1, tail=True)
    return returned if returned is not None else None


def _return(vm, frames, frame, ins, regs, constants):
    return vm._return(
        frames,
        frame,
        tuple(regs[ins.a + i] for i in range(ins.b)),
    )


def _returnv(vm, frames, frame, ins, regs, constants):
    values = [regs[ins.a + i] for i in range(ins.b)]
    mv = regs[ins.c]
    values.extend(mv.values if isinstance(mv, MultiValue) else (mv,))
    return vm._return(frames, frame, tuple(values))


def _halt(vm, frames, frame, ins, regs, constants):
    return vm._return(frames, frame, ())


OPCODE_HANDLERS = {
    Op.LOADK: _loadk,
    Op.MOVE: _move,
    Op.LOCAL: _local,
    Op.GETGLOBAL: _getglobal,
    Op.SETGLOBAL: _setglobal,
    Op.GETUPVAL: _getupval,
    Op.SETUPVAL: _setupval,
    Op.GETCELL: _getcell,
    Op.SETCELL: _setcell,
    Op.CLOSURE: _closure,
    Op.NEWTABLE: _newtable,
    Op.GETTABLE: _gettable,
    Op.SETTABLE: _settable,
    Op.SETLISTV: _setlistv,
    Op.LEN: _len,
    Op.ADD: _add,
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
    Op.SHR: _shr,
    Op.BNOT: _bnot,
    Op.CONCAT: _concat,
    Op.NEG: _neg,
    Op.NOT: _not,
    Op.TOBOOL: _tobool,
    Op.EQ: _eq,
    Op.LT: _lt,
    Op.LE: _le,
    Op.JMP: _jmp,
    Op.JMPIF: _jmpif,
    Op.JMPIFNOT: _jmpifnot,
    Op.JMPIFNIL: _jmpifnil,
    Op.FORPREP: _forprep,
    Op.FORLOOP: _forloop,
    Op.JFORLOOP: _forloop,
    Op.PFORPREP: _pforprep,
    Op.PFORLOOP: _pforloop,
    Op.PTFORPREP: _ptforprep,
    Op.PTFORLOOP: _ptforloop,
    Op.CALL: _call,
    Op.CALLV: _callv,
    Op.TAILCALL: _tailcall,
    Op.TAILCALLV: _tailcallv,
    Op.VARARG: _vararg,
    Op.PVARARG: _pvararg,
    Op.PGETVARG: _pgetvarg,
    Op.UNPACK: _unpack,
    Op.TBC: _tbc,
    Op.CLOSE: _close,
    Op.PTBC: _ptbc,
    Op.PCLOSE: _pclose,
    Op.CHECKNIL: _checknil,
    Op.RETURN: _return,
    Op.RETURNV: _returnv,
    Op.GUARD: _guard,
    Op.HALT: _halt,
}

_missing = set(Op).difference(OPCODE_HANDLERS)
_extra = set(OPCODE_HANDLERS).difference(Op)
if _missing or _extra:
    raise RuntimeError(
        f"opcode handler table mismatch: missing={sorted(_missing)} extra={sorted(_extra)}"
    )
