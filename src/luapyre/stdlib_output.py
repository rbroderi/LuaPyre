from __future__ import annotations

from .capabilities import RuntimeCapabilities
from .errors import LuaRuntimeError
from .stdlib_support import tostring_value
from .table import LuaTable
from .vm import HostFunction


def install_output_library(
    globals_table: LuaTable,
    vm,
    capabilities: RuntimeCapabilities,
) -> None:
    """Install safe host-output functions with replaceable byte sinks."""

    def lua_print(*values):
        rendered = [tostring_value(vm, value) for value in values]
        capabilities.output_sink(b"\t".join(rendered) + b"\n")
        return None

    globals_table.rawset(b"print", HostFunction(lua_print, "print"))

    warning_enabled = True

    def lua_warn(*messages):
        nonlocal warning_enabled
        if not messages:
            raise LuaRuntimeError("bad argument #1 to 'warn' (string expected)")
        for index, message in enumerate(messages, 1):
            if not isinstance(message, bytes):
                raise LuaRuntimeError(f"bad argument #{index} to 'warn' (string expected)")
        if len(messages) == 1 and messages[0].startswith(b"@"):
            if messages[0] == b"@off":
                warning_enabled = False
            elif messages[0] == b"@on":
                warning_enabled = True
            return None
        if warning_enabled:
            capabilities.warning_sink(b"".join(messages))
        return None

    # Replace the base implementation so output and warnings share the same
    # embedder-controlled capability boundary.
    globals_table.rawset(b"warn", HostFunction(lua_warn, "warn"))
