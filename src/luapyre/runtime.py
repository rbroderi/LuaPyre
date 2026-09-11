from __future__ import annotations

from .parser import Parser
from .compiler import Compiler
from .vm import VM, HostFunction


class LuaRuntime:
    """Sandbox-first LuaPyre runtime.

    No filesystem, network, or Python introspection capabilities are present
    unless explicitly exposed by the host application.
    """

    def __init__(self, *, fuel=1_000_000):
        self.globals = {}
        self.vm = VM(self.globals, fuel=fuel)

    def expose(self, name: str, fn):
        if not callable(fn):
            raise TypeError("exposed host capability must be callable")
        self.globals[name] = HostFunction(fn)
        return fn

    def compile(self, source: str):
        return Compiler().compile(Parser(source).parse())

    def execute(self, source: str, *, fuel=None):
        return self.vm.run(self.compile(source), fuel=fuel)

    def disassemble(self, source: str):
        return self.compile(source).disassemble()
