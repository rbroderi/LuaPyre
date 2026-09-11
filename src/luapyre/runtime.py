from __future__ import annotations

from .parser import Parser
from .compiler import Compiler
from .table import LuaTable
from .values import MultiValue, i64
from .vm import VM, HostFunction
from .stdlib import install_safe_stdlib


class LuaRuntime:
    """Sandbox-first LuaPyre runtime.

    The default environment contains only deterministic, in-memory helpers.
    Filesystem, network, process, import, eval, and Python introspection are
    absent unless the embedding application exposes an explicit capability.
    """

    def __init__(self, *, fuel=1_000_000, max_frames=1000, safe_stdlib=True):
        self.globals = LuaTable()
        if safe_stdlib:
            install_safe_stdlib(self.globals)
        self.vm = VM(self.globals, fuel=fuel, max_frames=max_frames)

    def _to_lua(self, value):
        if value is None or type(value) in (bool, float) or isinstance(value, bytes):
            return value
        if type(value) is int:
            return i64(value)
        if isinstance(value, str):
            return value.encode("utf-8")
        if isinstance(value, LuaTable):
            return value
        if isinstance(value, (list, tuple)):
            t = LuaTable()
            for i, item in enumerate(value, 1):
                t.rawset(i, self._to_lua(item))
            return t
        if isinstance(value, dict):
            t = LuaTable()
            for key, item in value.items():
                t.rawset(self._to_lua(key), self._to_lua(item))
            return t
        return value

    def expose(self, name: str, fn):
        if not callable(fn):
            raise TypeError("exposed host capability must be callable")

        def boundary(*args):
            return self._to_lua(fn(*args))

        self.globals.rawset(name.encode("utf-8"), HostFunction(boundary, name))
        return fn

    def set(self, name: str, value):
        self.globals.rawset(name.encode("utf-8"), self._to_lua(value))

    def get(self, name: str):
        return self.globals.rawget(name.encode("utf-8"))

    @staticmethod
    def multi_return(*values):
        return MultiValue(tuple(values))

    def compile(self, source: str):
        return Compiler().compile(Parser(source).parse())

    def execute(self, source: str, *, fuel=None):
        return self.vm.run(self.compile(source), fuel=fuel)

    def disassemble(self, source: str):
        return self.compile(source).disassemble()
