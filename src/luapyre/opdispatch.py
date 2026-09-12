from __future__ import annotations

import math

from .bytecode import Cell, Closure, Op
from .errors import LuaRuntimeError
from .table import LuaTable
from .values import (
    MultiValue, coerce_lua_integer, i64, parse_lua_number,
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


def _int_arith(vm, frames, frame, ins, regs, constants):
    a, b = _need_number(regs[ins.b]), _need_number(regs[ins.c])
    if ins.op is Op.ADD_I:
        value = a + b
    elif ins.op is Op.SUB_I:
        value = a - b
    else:
        value = a * b
    regs[ins.a] = i64(value)


def _float_arith(vm, frames, frame, ins, regs, constants):
    a, b = _need_number(regs[ins.b]), _need_number(regs[ins.c])
    if ins.op is Op.ADD_F:
        value = a + b
    elif ins.op is Op.SUB_F:
        value = a - b
    else:
        value = a * b
    regs[ins.a] = float(value)


def _generic_arith(vm, frames, frame, ins, regs, constants):
    a, b = regs[ins.b], regs[ins.c]
    if ins.op in (Op.BAND, Op.BOR, Op.BXOR, Op.SHL, Op.SHR):
        ai, bi = coerce_lua_integer(a), coerce_lua_integer(b)
        if ai is None or bi is None:
            _bitwise_error(vm, frames, frame, ins.op, a, b, ins.a)
            return
        if ins.op is Op.BAND:
            value = i64(ai & bi)
        elif ins.op is Op.BOR:
            value = i64(ai | bi)
        elif ins.op is Op.BXOR:
            value = i64(ai ^ bi)
        elif ins.op is Op.SHL:
            value = _shift(ai, bi, left=True)
        else:
            value = _shift(ai, bi, left=False)
        regs[ins.a] = value
        return

    na, nb = parse_lua_number(a), parse_lua_number(b)
    if na is not None and nb is not None:
        vm._generic_binary(frames, frame, ins.op, na, nb, ins.a)
    else:
        vm._generic_binary(frames, frame, ins.op, a, b, ins.a)


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


def _compare(vm, frames, frame, ins, regs, constants):
    vm._compare(frames, frame, ins.op, regs[ins.b], regs[ins.c], ins.a)


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
    tail = ins.op in (Op.TAILCALL, Op.TAILCALLV)
    fn = regs[ins.b]
    args = [regs[ins.c + i] for i in range(ins.d)]
    want = ins.e if ins.op is Op.CALL else -1
    if ins.op in (Op.CALLV, Op.TAILCALLV):
        mv = regs[ins.e]
        args.extend(mv.values if isinstance(mv, MultiValue) else (mv,))
    returned = vm._invoke(frames, frame, fn, args, ins.a, want, tail=tail)
    if tail and returned is not None:
        return returned
    return None


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
    Op.ADD: _generic_arith,
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
    Op.SHR: _generic_arith,
    Op.BNOT: _bnot,
    Op.CONCAT: _concat,
    Op.NEG: _neg,
    Op.NOT: _not,
    Op.TOBOOL: _tobool,
    Op.EQ: _compare,
    Op.LT: _compare,
    Op.LE: _compare,
    Op.JMP: _jmp,
    Op.JMPIF: _jmpif,
    Op.JMPIFNOT: _jmpifnot,
    Op.JMPIFNIL: _jmpifnil,
    Op.FORPREP: _forprep,
    Op.FORLOOP: _forloop,
    Op.PFORPREP: _pforprep,
    Op.PFORLOOP: _pforloop,
    Op.PTFORPREP: _ptforprep,
    Op.PTFORLOOP: _ptforloop,
    Op.CALL: _call,
    Op.CALLV: _call,
    Op.TAILCALL: _call,
    Op.TAILCALLV: _call,
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
