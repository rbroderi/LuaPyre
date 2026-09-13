from __future__ import annotations

from functools import cmp_to_key

from .errors import LuaRuntimeError
from .table import LuaTable
from .values import MultiValue, i64
from .stdlib_support import (
    lua_c_function,
    need_bytes,
    need_integer,
    need_table,
    number_to_bytes,
)


def install_table_library(globals_table: LuaTable, vm) -> LuaTable:
    tablelib = LuaTable()

    def table_length(tab, function):
        tab = need_table(tab, 1, function)
        metamethod = vm._tm(tab, b"__len")
        if metamethod is None:
            return tab.rawlen()
        values = vm.call_sync(metamethod, (tab,))
        value = values[0] if values else None
        if type(value) is int or (type(value) is float and value.is_integer()):
            return int(value)
        raise LuaRuntimeError("object length is not an integer")

    def concat(tab, sep=b"", i=1, j=None):
        tab = need_table(tab, 1, "concat")
        sep = need_bytes(sep, 2, "concat")
        i = need_integer(i, 3, "concat")
        j = table_length(tab, "concat") if j is None else need_integer(j, 4, "concat")
        if i > j:
            return b""
        parts = []
        for index in range(i, j + 1):
            value = vm.index_sync(tab, index)
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
        limit = 1 << 31
        if nseq < 0 or nrec < 0 or nseq >= limit or nrec >= limit:
            raise LuaRuntimeError("bad argument to 'create' (size out of range)")
        if nseq + nrec >= 1 << 30:
            raise LuaRuntimeError("table overflow")
        result = LuaTable()
        result._reserved_bytes = (nseq + nrec) * 16
        vm.gc.adopt(result)
        return result

    def insert(tab, *args):
        tab = need_table(tab, 1, "insert")
        length = table_length(tab, "insert")
        if len(args) == 1:
            pos, value = i64(length + 1), args[0]
            default_position = True
        elif len(args) == 2:
            pos, value = need_integer(args[0], 2, "insert"), args[1]
            default_position = False
        else:
            raise LuaRuntimeError("wrong number of arguments to 'insert'")
        if not default_position and (pos < 1 or pos > length + 1):
            raise LuaRuntimeError("bad argument #2 to 'insert' (position out of bounds)")
        if not default_position:
            for index in range(length, pos - 1, -1):
                vm.set_index_sync(tab, index + 1, vm.index_sync(tab, index))
        vm.set_index_sync(tab, pos, value)

    def move(a1, f, e, t, a2=None):
        a1 = need_table(a1, 1, "move")
        f = need_integer(f, 2, "move")
        e = need_integer(e, 3, "move")
        t = need_integer(t, 4, "move")
        a2 = a1 if a2 is None else need_table(a2, 5, "move")
        if f > e:
            return a2
        count = e - f + 1
        if count < 0 or count > (1 << 63) - 1:
            raise LuaRuntimeError("too many elements to move")
        destination_end = t + count - 1
        if destination_end < -(1 << 63) or destination_end > (1 << 63) - 1:
            raise LuaRuntimeError("destination wrap around")
        if a1 is a2 and t > f and t <= e:
            indexes = range(count - 1, -1, -1)
        else:
            indexes = range(count)
        for offset in indexes:
            value = vm.index_sync(a1, f + offset)
            vm.set_index_sync(a2, t + offset, value)
        return a2

    def pack(*values):
        result = LuaTable.from_sequence(values)
        result.rawset(b"n", len(values))
        return result

    def remove(tab, pos=None):
        tab = need_table(tab, 1, "remove")
        length = table_length(tab, "remove")
        default_position = pos is None
        pos = length if default_position else need_integer(pos, 2, "remove")
        if default_position and length == 0:
            removed = vm.index_sync(tab, 0)
            if removed is not None:
                vm.set_index_sync(tab, 0, None)
            return removed
        if pos < 1 or pos > length:
            if (length == 0 and pos == 0) or pos == length + 1:
                return None
            raise LuaRuntimeError("bad argument #2 to 'remove' (position out of bounds)")
        removed = vm.index_sync(tab, pos)
        for index in range(pos, length):
            vm.set_index_sync(tab, index, vm.index_sync(tab, index + 1))
        vm.set_index_sync(tab, length, None)
        return removed

    def sort(tab, comp=None):
        tab = need_table(tab, 1, "table.sort")
        length = table_length(tab, "sort")
        if length > 1_000_000:
            raise LuaRuntimeError("array too big")
        values = [vm.index_sync(tab, i) for i in range(1, length + 1)]

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

        if comp is not None and values and less(values[0], values[0]):
            raise LuaRuntimeError("invalid order function for sorting")
        values.sort(key=cmp_to_key(compare))
        for index, value in enumerate(values, 1):
            vm.set_index_sync(tab, index, value)

    def unpack(tab, i=1, j=None):
        i = need_integer(i, 2, "unpack")
        if j is None:
            if isinstance(tab, LuaTable) and vm._tm(tab, b"__len") is None:
                j = tab.rawlen()
            else:
                metamethod = vm._tm(tab, b"__len")
                if metamethod is None:
                    raise LuaRuntimeError("bad argument #1 to 'unpack' (table expected)")
                values = vm.call_sync(metamethod, (tab,))
                j = need_integer(values[0] if values else None, 1, "unpack")
        else:
            j = need_integer(j, 3, "unpack")
        if j < i:
            return MultiValue(())
        count = j - i + 1
        if count > 1_000_000:
            raise LuaRuntimeError("too many results to unpack")
        return MultiValue(tuple(vm.index_sync(tab, index) for index in range(i, j + 1)))

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
        tablelib.rawset(short.encode("ascii"), lua_c_function(fn, f"table.{short}"))

    globals_table.rawset(b"table", tablelib)
    return tablelib
