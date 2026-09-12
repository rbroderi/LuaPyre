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
        # PUC-Lua's diagnostic retains expression context for math.huge. LuaPyre
        # does not yet carry full value provenance through registers, so preserve
        # the observable standard-library field name for infinity conversions.
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
    if type(bad) is float and math.isinf(bad):
        raise LuaRuntimeError("number (field 'huge') has no integer representation")
    if type(bad) is float:
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

