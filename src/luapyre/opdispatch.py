from __future__ import annotations

import math

from .bytecode import Cell, Closure, Op
from .errors import LuaRuntimeError
from .table import LuaTable
from .values import MultiValue, i64, static_value_type, truthy, type_matches


def _is_number(value):
    return type(value) in (int, float)


def _need_number(value):
    if not _is_number(value):
        raise LuaRuntimeError(
            f"attempt to perform arithmetic on a {static_value_type(value).name} value"
        )
    return value


def _to_int(value):
    if type(value) is int:
        return i64(value)
    if type(value) is float and math.isfinite(value) and value.is_integer():
        iv = int(value)
        if -(1 << 63) <= iv <= (1 << 63) - 1:
            return iv
    raise LuaRuntimeError("number has no integer representation")


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
    vm._generic_binary(frames, frame, ins.op, regs[ins.b], regs[ins.c], ins.a)


def _neg(vm, frames, frame, ins, regs, constants):
    value = regs[ins.b]
    if _is_number(value):
        regs[ins.a] = i64(-value) if type(value) is int else -value
        return
    tm = vm._tm(value, b"__unm")
    if tm is None:
        raise LuaRuntimeError(
            f"attempt to perform arithmetic on a {static_value_type(value).name} value"
        )
    vm._invoke(frames, frame, tm, [value], ins.a, 1)


def _bnot(vm, frames, frame, ins, regs, constants):
    value = regs[ins.b]
    try:
        regs[ins.a] = i64(~_to_int(value))
    except LuaRuntimeError:
        tm = vm._tm(value, b"__bnot")
        if tm is None:
            raise
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


def _vararg(vm, frames, frame, ins, regs, constants):
    if ins.b == -1:
        regs[ins.a] = MultiValue(frame.varargs)
        return
    for i in range(ins.b):
        regs[ins.a + i] = frame.varargs[i] if i < len(frame.varargs) else None


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
    Op.CALL: _call,
    Op.CALLV: _call,
    Op.TAILCALL: _call,
    Op.TAILCALLV: _call,
    Op.VARARG: _vararg,
    Op.UNPACK: _unpack,
    Op.TBC: _tbc,
    Op.CLOSE: _close,
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
