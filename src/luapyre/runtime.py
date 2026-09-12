from __future__ import annotations

from .binary_chunks import fresh_loaded_closure
from .bytecode import Closure
from .capabilities import RuntimeCapabilities
from .diagnostics import format_traceback
from .diagnostic_stdlib import install_diagnostic_stdlib
from .errors import LuaRuntimeError
from .gcvm import GarbageCollectedVM
from .optimizing_jitvm import OptimizingJITVM
from .parser import Parser
from .source_compiler import SourceCompiler
from .source_mode import FULLY_TYPED_MODE, detect_source_mode
from .stdlib import install_safe_stdlib
from .stdlib_output import install_output_library
from .stdlib_package import install_package_library
from .table import LuaTable
from .threadvm import LuaThread
from .typed_parser import TypedParser
from .values import MultiValue, i64
from .vm import HostFunction


class LuaRuntime:
    """Sandbox-first LuaPyre runtime.

    The default environment contains deterministic, in-memory helpers plus
    console output. Filesystem, network, process, native-library loading, and
    Python introspection remain absent unless the embedding application exposes
    an explicit capability.

    LuaPyre enables the guarded tiered JIT by default. Pass ``jit=False`` for
    the exact interpreter-only execution path, or lower ``jit_threshold`` when
    profiling short hot loops/functions. Source beginning with the first/second
    line cookie ``-- luapyre: typed`` opts into the fully typed compiler
    contract used as LuaPyre's primary optimization target.
    """

    def __init__(
        self,
        *,
        fuel=1_000_000,
        max_frames=1000,
        safe_stdlib=True,
        output=None,
        warning=None,
        file_loader=None,
        jit=True,
        jit_threshold=32,
    ):
        self.globals = LuaTable()
        if jit:
            self.vm = OptimizingJITVM(
                self.globals,
                fuel=fuel,
                max_frames=max_frames,
                jit_enabled=True,
                jit_threshold=jit_threshold,
            )
        else:
            self.vm = GarbageCollectedVM(self.globals, fuel=fuel, max_frames=max_frames)
        self.capabilities = RuntimeCapabilities()
        self.capabilities.set_output_sink(output)
        self.capabilities.set_warning_sink(warning)
        self.capabilities.set_file_loader(file_loader)
        self._package_state = None
        if safe_stdlib:
            install_safe_stdlib(self.globals, self.vm)
            install_output_library(self.globals, self.vm, self.capabilities)
            install_diagnostic_stdlib(self.globals, self.vm)
            self._package_state = install_package_library(
                self.globals, self.vm, self.capabilities
            )

    def _to_lua(self, value, *, _adopt=True):
        if value is None or type(value) in (bool, float) or isinstance(value, bytes):
            return value
        if type(value) is int:
            return i64(value)
        if isinstance(value, str):
            return value.encode("utf-8")
        if isinstance(value, (LuaTable, LuaThread, Closure, HostFunction)):
            if _adopt and not isinstance(value, HostFunction):
                self.vm.gc.adopt(value)
            return value
        if isinstance(value, (list, tuple)):
            t = LuaTable()
            for i, item in enumerate(value, 1):
                t.rawset(i, self._to_lua(item, _adopt=False))
            if _adopt:
                self.vm.gc.safepoint()
                self.vm.gc.adopt(t)
            return t
        if isinstance(value, dict):
            t = LuaTable()
            for key, item in value.items():
                t.rawset(
                    self._to_lua(key, _adopt=False),
                    self._to_lua(item, _adopt=False),
                )
            if _adopt:
                self.vm.gc.safepoint()
                self.vm.gc.adopt(t)
            return t

        # Opaque host userdata is represented by the Python object itself.
        # Do not replace this with a recyclable integer/ref-slot registry unless
        # every reuse and finalization path validates the referent identity (or
        # a generation token). A stale reusable wrapper can otherwise become
        # rebound to a different live host object; see Lupa GH-294 for the class
        # of bug this direct-reference invariant deliberately avoids.
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

    def set_output_sink(self, sink=None) -> None:
        """Replace the byte-oriented sink used by Lua ``print``."""
        self.capabilities.set_output_sink(sink)

    def set_warning_sink(self, sink=None) -> None:
        """Replace the byte-oriented sink used by Lua ``warn``."""
        self.capabilities.set_warning_sink(sink)

    def set_file_loader(self, loader=None) -> None:
        """Install or remove the host's read-only Lua file capability."""
        self.capabilities.set_file_loader(loader)
        if self._package_state is not None:
            self._package_state.refresh_file_capability()

    def preload(self, name: str | bytes, module) -> None:
        """Register an in-memory module for the standard preload searcher.

        ``module`` may be Lua source text/bytes, a Lua function, or a Python
        callable. Python callables receive the standard loader arguments
        ``(module_name, loader_data)`` and have their result converted through
        the ordinary host boundary.
        """
        if self._package_state is None:
            raise RuntimeError("preload requires safe_stdlib=True")
        key = name.encode("utf-8") if isinstance(name, str) else name
        if not isinstance(key, bytes):
            raise TypeError("module name must be str or bytes")

        if isinstance(module, str):
            proto = self.compile(module, chunkname=b"@" + key)
            loader = fresh_loaded_closure(proto, self.globals)
        elif isinstance(module, bytes):
            proto = self.compile(module.decode("utf-8"), chunkname=b"@" + key)
            loader = fresh_loaded_closure(proto, self.globals)
        elif isinstance(module, (Closure, HostFunction)):
            loader = module
        elif callable(module):
            def boundary(*args):
                return self._to_lua(module(*args))
            loader = HostFunction(boundary, f"package.preload[{key!r}]")
        else:
            raise TypeError("preloaded module must be Lua source or callable")
        self._package_state.preload.rawset(key, loader)

    @staticmethod
    def multi_return(*values):
        return MultiValue(tuple(values))

    def compile(self, source: str, *, chunkname: str | bytes = "=(luapyre)"):
        mode = detect_source_mode(source)
        fully_typed = mode == FULLY_TYPED_MODE
        parser = TypedParser(source) if fully_typed else Parser(source)
        return SourceCompiler(
            chunkname,
            fully_typed=fully_typed,
        ).compile(parser.parse())

    def execute(self, source: str, *, fuel=None, chunkname: str | bytes = "=(luapyre)"):
        return self.vm.run(self.compile(source, chunkname=chunkname), fuel=fuel)

    @property
    def jit_stats(self):
        """Return live tiered-JIT counters, or ``None`` for interpreter-only runtimes."""
        jit = getattr(self.vm, "jit", None)
        sync = getattr(self.vm, "sync_inline_cache_stats", None)
        if sync is not None:
            sync()
        return None if jit is None else jit.stats

    @property
    def jit_feedback(self):
        """Snapshot adaptive inline-cache and deoptimization feedback."""
        caches = getattr(self.vm, "inline_caches", None)
        return None if caches is None else caches.snapshot()

    @staticmethod
    def traceback(error: LuaRuntimeError, *, include_message: bool = True) -> bytes:
        """Format the structured Lua stack captured on a runtime exception."""
        if not isinstance(error, LuaRuntimeError):
            raise TypeError("traceback expects a LuaRuntimeError")
        return format_traceback(error, include_message=include_message)

    def collect(self):
        """Run one full Lua-level collection cycle."""
        return self.vm.gc.collect()

    def disassemble(self, source: str, *, chunkname: str | bytes = "=(luapyre)"):
        return self.compile(source, chunkname=chunkname).disassemble()
