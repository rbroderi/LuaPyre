from __future__ import annotations

from functools import cmp_to_key

from .errors import LuaRuntimeError
from .table import LuaTable
from .values import MultiValue
from .stdlib_support import need_bytes, need_integer, need_table, number_to_bytes, put


def install_table_library(globals_table: LuaTable, vm) -> LuaTable:
    tablelib = LuaTable()

    def concat(tab, sep=b"", i=1, j=None):
        tab = need_table(tab, 1, "concat")
        sep = need_bytes(sep, 2, "concat")
        i = need_integer(i, 3, "concat")
        j = tab.rawlen() if j is None else need_integer(j, 4, "concat")
        if i > j:
            return b""
        parts = []
        for index in range(i, j + 1):
            value = tab.rawget(index)
            if isinstance(value, bytes):
                parts.append(value)
            elif type(value) in (int, float):
                parts.append(number_to_bytes(value))
            else:
                raise LuaRuntimeError(
                    f"invalid value ({type(value).__name__}) at index {index} in table for 'concat'"
                )
        return sep.join(parts)

    def create(nseq, nrec=0):
        nseq = need_integer(nseq, 1, "create")
        nrec = need_integer(nrec, 2, "create")
        if nseq < 0 or nrec < 0:
            raise LuaRuntimeError("bad argument to 'create' (size out of range)")
        # LuaPyre's split table grows lazily, so preallocation is only a semantic
        # hint and does not need to expose Python capacity management.
        return LuaTable()

    def insert(tab, *args):
        tab = need_table(tab, 1, "insert")
        length = tab.rawlen()
        if len(args) == 1:
            pos, value = length + 1, args[0]
        elif len(args) == 2:
            pos, value = need_integer(args[0], 2, "insert"), args[1]
        else:
            raise LuaRuntimeError("wrong number of arguments to 'insert'")
        if pos < 1 or pos > length + 1:
            raise LuaRuntimeError("bad argument #2 to 'insert' (position out of bounds)")
        for index in range(length, pos - 1, -1):
            tab.rawset(index + 1, tab.rawget(index))
        tab.rawset(pos, value)

    def move(a1, f, e, t, a2=None):
        a1 = need_table(a1, 1, "move")
        f = need_integer(f, 2, "move")
        e = need_integer(e, 3, "move")
        t = need_integer(t, 4, "move")
        a2 = a1 if a2 is None else need_table(a2, 5, "move")
        if f > e:
            return a2
        count = e - f + 1
        if count < 0 or t > (1 << 63) - count:
            raise LuaRuntimeError("too many elements to move")
        if a1 is a2 and t > f and t <= e:
            indexes = range(count - 1, -1, -1)
        else:
            indexes = range(count)
        for offset in indexes:
            a2.rawset(t + offset, a1.rawget(f + offset))
        return a2

    def pack(*values):
        result = LuaTable.from_sequence(values)
        result.rawset(b"n", len(values))
        return result

    def remove(tab, pos=None):
        tab = need_table(tab, 1, "remove")
        length = tab.rawlen()
        pos = length if pos is None else need_integer(pos, 2, "remove")
        if pos < 1 or pos > length:
            if (length == 0 and pos == 0) or pos == length + 1:
                return None
            raise LuaRuntimeError("bad argument #2 to 'remove' (position out of bounds)")
        removed = tab.rawget(pos)
        for index in range(pos, length):
            tab.rawset(index, tab.rawget(index + 1))
        tab.rawset(length, None)
        return removed

    def sort(tab, comp=None):
        tab = need_table(tab, 1, "sort")
        values = [tab.rawget(i) for i in range(1, tab.rawlen() + 1)]

        def less(a, b):
            if comp is None:
                return vm.less_than_sync(a, b)
            result = vm.call_sync(comp, (a, b))
            value = result[0] if result else None
            return value is not None and value is not False

        def compare(a, b):
            if less(a, b):
                return -1
            if less(b, a):
                return 1
            return 0

        values.sort(key=cmp_to_key(compare))
        for index, value in enumerate(values, 1):
            tab.rawset(index, value)

    def unpack(tab, i=1, j=None):
        tab = need_table(tab, 1, "unpack")
        i = need_integer(i, 2, "unpack")
        j = tab.rawlen() if j is None else need_integer(j, 3, "unpack")
        if j < i:
            return MultiValue(())
        count = j - i + 1
        if count > 1_000_000:
            raise LuaRuntimeError("too many results to unpack")
        return MultiValue(tuple(tab.rawget(index) for index in range(i, j + 1)))

    put(tablelib, "table.concat", concat)
    # Fields use their Lua-visible short names; HostFunction.name remains fully qualified.
    for short, fn in (
        ("concat", concat),
        ("create", create),
        ("insert", insert),
        ("move", move),
        ("pack", pack),
        ("remove", remove),
        ("sort", sort),
        ("unpack", unpack),
    ):
        tablelib.rawset(short.encode("ascii"), HostFunction(fn, f"table.{short}"))

    globals_table.rawset(b"table", tablelib)
    return tablelib


# Kept local to avoid exposing the registration helper as part of the Lua library.
from .vm import HostFunction
