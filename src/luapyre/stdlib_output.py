from __future__ import annotations

from .stdlib_support import tostring_value
from .table import LuaTable
from .vm import HostFunction


def install_output_library(globals_table: LuaTable, vm) -> None:
    """Install the safe console-output portion of the Lua base library."""

    def lua_print(*values):
        # Lua 5.5's print uses luaL_tolstring for each argument, so preserve
        # __tostring and Lua's own scalar formatting before handing the final
        # text to Python's stdout implementation.
        rendered = [tostring_value(vm, value).decode("utf-8", "replace") for value in values]
        print(*rendered, sep="\t")
        return None

    globals_table.rawset(b"print", HostFunction(lua_print, "print"))
