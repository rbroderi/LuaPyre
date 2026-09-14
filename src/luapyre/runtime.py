from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields, is_dataclass
import inspect
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

from .binary_chunks import fresh_loaded_closure
from .bytecode import Closure, Ins, Op, Proto
from .capabilities import RuntimeCapabilities
from .diagnostics import format_traceback
from .diagnostic_stdlib import install_diagnostic_stdlib
from .stdlib_debug import install_debug_library
from .stdlib_io import install_io_library
from .stdlib_os import install_os_library
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
from .interop import LuaFunction, LuaInt
from .typed_parser import TypedParser
from .typesys import ANY
from .values import MultiValue, i64, truthy
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
        source_cache_size=128,
        debug_hooks=False,
    ):
        if type(source_cache_size) is not int or source_cache_size < 0:
            raise ValueError("source_cache_size must be a non-negative integer")
        if type(debug_hooks) is not bool:
            raise TypeError("debug_hooks must be a boolean")
        self.source_cache_size = source_cache_size
        self._source_cache = OrderedDict()
        self._source_cache_hits = 0
        self._source_cache_misses = 0
        self._python_call_cache = OrderedDict()
        self.globals = LuaTable()
        if jit:
            self.vm = OptimizingJITVM(
                self.globals,
                fuel=fuel,
                max_frames=max_frames,
                jit_enabled=True,
                jit_threshold=jit_threshold,
                debug_hooks_enabled=debug_hooks,
            )
        else:
            self.vm = GarbageCollectedVM(
                self.globals,
                fuel=fuel,
                max_frames=max_frames,
                debug_hooks_enabled=debug_hooks,
            )
        self.capabilities = RuntimeCapabilities()
        self.capabilities.set_output_sink(output)
        self.capabilities.set_warning_sink(warning)
        self.capabilities.set_file_loader(file_loader)
        self._package_state = None
        if safe_stdlib:
            install_safe_stdlib(self.globals, self.vm)
            install_output_library(self.globals, self.vm, self.capabilities)
            install_diagnostic_stdlib(self.globals, self.vm)
            install_io_library(self.globals, self.capabilities)
            install_os_library(self.globals, self.capabilities)
            install_debug_library(self.globals, self.vm, self.capabilities)
            self._package_state = install_package_library(
                self.globals, self.vm, self.capabilities
            )

    def _to_lua(self, value, *, _adopt=True, _seen=None):
        if value is None or type(value) in (bool, float) or isinstance(value, bytes):
            return value
        if type(value) is int:
            return value if -(1 << 63) <= value <= (1 << 63) - 1 else i64(value)
        if isinstance(value, str):
            return value.encode("utf-8")
        if isinstance(value, LuaFunction):
            if value._runtime is not self:
                raise ValueError("a Lua function cannot cross between runtimes")
            return value.raw
        if isinstance(value, (LuaTable, LuaThread, Closure, HostFunction)):
            if _adopt and not isinstance(value, HostFunction):
                self.vm.gc.adopt(value)
            return value
        if callable(value):
            return self._wrap_python_callable(value)

        container = (
            isinstance(value, (list, tuple, set, frozenset, dict))
            or is_dataclass(value) and not isinstance(value, type)
        )
        if container:
            seen = set() if _seen is None else _seen
            ident = id(value)
            if ident in seen:
                raise ValueError("cyclic Python values cannot be converted to Lua")
            seen.add(ident)
            try:
                t = LuaTable()
                if isinstance(value, (list, tuple)):
                    for i, item in enumerate(value, 1):
                        t.rawset(
                            i,
                            self._to_lua(item, _adopt=False, _seen=seen),
                        )
                elif isinstance(value, (set, frozenset)):
                    for item in value:
                        key = self._to_lua(item, _adopt=False, _seen=seen)
                        if type(key) not in (bool, int, float, bytes):
                            raise TypeError(
                                "set elements must be int, float, str, bytes, or bool"
                            )
                        t.rawset(key, True)
                elif isinstance(value, dict):
                    for key, item in value.items():
                        lua_key = self._to_lua(key, _adopt=False, _seen=seen)
                        if type(lua_key) not in (bool, int, float, bytes):
                            raise TypeError(
                                "dictionary keys must be int, float, str, bytes, or bool"
                            )
                        t.rawset(
                            lua_key,
                            self._to_lua(item, _adopt=False, _seen=seen),
                        )
                else:
                    for field in fields(value):
                        t.rawset(
                            field.name.encode("utf-8"),
                            self._to_lua(
                                getattr(value, field.name),
                                _adopt=False,
                                _seen=seen,
                            ),
                        )
                if _adopt:
                    self.vm.gc.safepoint()
                    self.vm.gc.adopt(t)
                return t
            finally:
                seen.remove(ident)

        # Opaque host userdata is represented by the Python object itself.
        # Do not replace this with a recyclable integer/ref-slot registry unless
        # every reuse and finalization path validates the referent identity (or
        # a generation token). A stale reusable wrapper can otherwise become
        # rebound to a different live host object; see Lupa GH-294 for the class
        # of bug this direct-reference invariant deliberately avoids.
        return value

    def _to_lua_result(self, value):
        if isinstance(value, MultiValue):
            return MultiValue(tuple(self._to_lua(item) for item in value.values))
        return self._to_lua(value)

    @staticmethod
    def _annotation_for_argument(signature, hints, index):
        if signature is None:
            return None
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if index < len(positional):
            parameter = positional[index]
            return hints.get(parameter.name, parameter.annotation)
        variadic = next(
            (
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind is inspect.Parameter.VAR_POSITIONAL
            ),
            None,
        )
        if variadic is None:
            return None
        return hints.get(variadic.name, variadic.annotation)

    def _wrap_python_callable(self, fn, name=None):
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            signature = None
        try:
            hints = get_type_hints(fn)
        except (NameError, TypeError):
            hints = {}
        return_hint = hints.get(
            "return",
            signature.return_annotation if signature is not None else None,
        )

        def boundary(*args):
            converted = []
            for index, value in enumerate(args):
                expected = self._annotation_for_argument(signature, hints, index)
                if expected is inspect.Parameter.empty:
                    expected = None
                converted.append(self._from_lua(value, expected))
            result = fn(*converted)
            if return_hint is LuaInt:
                # Validate before ordinary int conversion applies Lua's exact
                # wraparound and would conceal a broken fast-integer contract.
                result = self._from_lua(result, LuaInt)
            return self._to_lua_result(result)

        display_name = name or getattr(fn, "__name__", type(fn).__name__)
        return HostFunction(boundary, display_name)

    @staticmethod
    def _dense_table_values(value: LuaTable):
        if value.hash or any(item is None for item in value.array):
            raise TypeError("Lua table is not a dense sequence")
        return tuple(value.array)

    def _from_lua(self, value, expected=None, *, _seen=None):
        if expected in (inspect.Parameter.empty, Any, object):
            expected = None

        if expected is not None:
            # Common scalar annotations need no typing introspection. Keep the
            # exact representation checks before the generic container path.
            if expected is type(None):
                if value is not None:
                    raise TypeError("expected nil")
                return None
            if expected is bool:
                if type(value) is not bool:
                    raise TypeError("expected boolean")
                return value
            if expected is LuaInt:
                if type(value) is not int:
                    raise TypeError("expected integer")
                if not -(1 << 63) <= value <= (1 << 63) - 1:
                    raise OverflowError("LuaInt result is outside signed 64-bit range")
                return LuaInt(value)
            if expected is int:
                if type(value) is not int:
                    raise TypeError("expected integer")
                return value
            if expected is float:
                if type(value) is not float:
                    raise TypeError("expected float")
                return value
            if expected is bytes:
                if not isinstance(value, bytes):
                    raise TypeError("expected string")
                return value
            if expected is str:
                if not isinstance(value, bytes):
                    raise TypeError("expected string")
                return value.decode("utf-8")
            if expected is LuaFunction:
                if not isinstance(value, (Closure, HostFunction)):
                    raise TypeError("expected function")
                return LuaFunction(self, value, getattr(value, "name", "?"))

            origin = get_origin(expected)
            args = get_args(expected)
            if origin in (types.UnionType, Union):
                errors = []
                for option in args:
                    try:
                        return self._from_lua(value, option, _seen=_seen)
                    except (TypeError, UnicodeDecodeError) as error:
                        errors.append(str(error))
                raise TypeError(
                    f"Lua value does not match {expected!r}: " + "; ".join(errors)
                )

            if expected in (list, set, frozenset, dict, tuple):
                origin = expected
                args = ()

            if origin is tuple and isinstance(value, tuple):
                if len(args) == 2 and args[1] is Ellipsis:
                    return tuple(
                        self._from_lua(item, args[0], _seen=_seen)
                        for item in value
                    )
                if args and len(value) != len(args):
                    raise TypeError("Lua return has the wrong tuple length")
                return tuple(
                    self._from_lua(
                        item,
                        args[index] if args else None,
                        _seen=_seen,
                    )
                    for index, item in enumerate(value)
                )

            if origin in (list, set, frozenset, dict, tuple) or (
                isinstance(expected, type) and is_dataclass(expected)
            ):
                if not isinstance(value, LuaTable):
                    raise TypeError("expected table")
                seen = set() if _seen is None else _seen
                ident = id(value)
                if ident in seen:
                    raise ValueError("cyclic Lua tables cannot be converted to Python")
                seen.add(ident)
                try:
                    if origin is list:
                        item_type = args[0] if args else None
                        return [
                            self._from_lua(item, item_type, _seen=seen)
                            for item in self._dense_table_values(value)
                        ]
                    if origin in (set, frozenset):
                        item_type = args[0] if args else None
                        pairs = tuple(value.items())
                        if pairs and all(included is True for _, included in pairs):
                            # Python sets use the Lua membership-table shape.
                            # This also handles positive integer keys that the
                            # table stores in its dense array part.
                            items = tuple(key for key, _ in pairs)
                        elif not value.hash:
                            items = self._dense_table_values(value)
                        else:
                            items = tuple(
                                key for key, included in pairs if truthy(included)
                            )
                        result = {
                            self._from_lua(item, item_type, _seen=seen)
                            for item in items
                        }
                        return result if origin is set else frozenset(result)
                    if origin is tuple:
                        values = self._dense_table_values(value)
                        if len(args) == 2 and args[1] is Ellipsis:
                            return tuple(
                                self._from_lua(item, args[0], _seen=seen)
                                for item in values
                            )
                        if args and len(values) != len(args):
                            raise TypeError("Lua sequence has the wrong tuple length")
                        return tuple(
                            self._from_lua(
                                item,
                                args[index] if args else None,
                                _seen=seen,
                            )
                            for index, item in enumerate(values)
                        )
                    if origin is dict:
                        key_type = args[0] if args else None
                        item_type = (
                            args[1]
                            if len(args) > 1
                            else args[0] if len(args) == 1 else None
                        )
                        return {
                            self._from_lua(key, key_type, _seen=seen): self._from_lua(
                                item, item_type, _seen=seen
                            )
                            for key, item in value.items()
                        }

                    type_hints = get_type_hints(expected)
                    keyword = {}
                    for field in fields(expected):
                        lua_value = value.rawget(field.name.encode("utf-8"))
                        if lua_value is None and not value.rawhas(
                            field.name.encode("utf-8")
                        ):
                            raise TypeError(f"missing dataclass field {field.name!r}")
                        keyword[field.name] = self._from_lua(
                            lua_value,
                            type_hints.get(field.name, field.type),
                            _seen=seen,
                        )
                    return expected(**keyword)
                finally:
                    seen.remove(ident)
            raise TypeError(f"unsupported Python return type {expected!r}")

        if value is None or type(value) in (bool, int, float):
            return value
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value
        if isinstance(value, (Closure, HostFunction)):
            return LuaFunction(self, value, getattr(value, "name", "?"))
        if isinstance(value, tuple):
            return tuple(self._from_lua(item, _seen=_seen) for item in value)
        if isinstance(value, LuaTable):
            seen = set() if _seen is None else _seen
            ident = id(value)
            if ident in seen:
                raise ValueError("cyclic Lua tables cannot be converted to Python")
            seen.add(ident)
            try:
                if not value.hash and not any(item is None for item in value.array):
                    return [self._from_lua(item, _seen=seen) for item in value.array]
                return {
                    self._from_lua(key, _seen=seen): self._from_lua(item, _seen=seen)
                    for key, item in value.items()
                }
            finally:
                seen.remove(ident)
        return value

    def expose(self, name: str, fn):
        if not callable(fn):
            raise TypeError("exposed host capability must be callable")
        self.globals.rawset(
            name.encode("utf-8"),
            self._wrap_python_callable(fn, name),
        )
        return fn

    def set(self, name: str, value):
        self.globals.rawset(name.encode("utf-8"), self._to_lua(value))

    def get(self, name: str):
        return self.globals.rawget(name.encode("utf-8"))

    def get_python(self, name: str, *, return_type=None):
        """Read a global through the Python conversion boundary."""
        return self._from_lua(self.get(name), return_type)

    def function(self, name: str) -> LuaFunction:
        """Return a named Lua function as a callable Python object."""
        value = self.get(name)
        if not isinstance(value, (Closure, HostFunction)):
            raise TypeError(f"global {name!r} is not a Lua function")
        return LuaFunction(self, value, name)

    def call(self, function, *args, return_type=None, fuel=None):
        """Call a Lua function from Python with recursive value conversion."""
        if isinstance(function, str):
            function = self.get(function)
        elif isinstance(function, LuaFunction):
            if function._runtime is not self:
                raise ValueError("a Lua function cannot cross between runtimes")
            function = function.raw
        elif callable(function) and not isinstance(function, HostFunction):
            function = self._wrap_python_callable(function)

        if isinstance(function, Closure) and function.proto.jit_fully_typed:
            for arg, typ in zip(args, function.proto.param_types):
                if typ.name == "integer" and type(arg) is int:
                    if not -(1 << 63) <= arg <= (1 << 63) - 1:
                        raise OverflowError("LuaInt result is outside signed 64-bit range")
        lua_args = [self._to_lua(arg) for arg in args]
        direct_call = getattr(self.vm, "call_compiled_leaf", None)
        if direct_call is not None and isinstance(function, Closure):
            entered, result = direct_call(function, tuple(lua_args), fuel)
            if entered:
                return self._from_lua(result, return_type)
        cache_key = (id(function), len(lua_args))
        # Borrow an inactive trampoline. A recursive host callback or debug
        # hook must not overwrite constants the outer call is still loading.
        cached = self._python_call_cache.pop(cache_key, None)
        if cached is not None and cached[0] is function:
            proto = cached[1]
            proto.constants[1:] = lua_args
        else:
            constants = [function, *lua_args]
            code = [Ins(Op.LOADK, index, index) for index in range(len(constants))]
            # A stable non-tail CALL site lets repeated Python entry warm the
            # same feedback slot and enter whole-function typed compilation.
            code.extend(
                (
                    Ins(Op.CALL, 0, 0, 1, len(lua_args), -1),
                    Ins(Op.RETURNV, 0, 0, 0),
                    Ins(Op.HALT),
                )
            )
            proto = Proto(
                "=[python call]",
                code=code,
                constants=constants,
                register_count=max(1, len(constants)),
                param_types=[],
                return_types=[ANY],
                source=b"=[python call]",
                lineinfo=[1] * len(code),
            )
        try:
            result = self.vm.run(proto, fuel=fuel)
        finally:
            proto.constants[1:] = [None] * len(lua_args)
            self._python_call_cache[cache_key] = (function, proto)
            if len(self._python_call_cache) > max(16, self.source_cache_size):
                self._python_call_cache.popitem(last=False)
        return self._from_lua(result, return_type)

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

    def set_environment(self, values=None) -> None:
        """Replace the private environment exposed through ``os.getenv``."""
        self.capabilities.set_environment(values)

    def preload(self, name: str | bytes, module) -> None:
        """Register an in-memory module for the standard preload searcher.

        ``module`` may be Lua source text/bytes, a Lua function, or a Python
        callable. Python callables receive the standard loader arguments
        ``(module_name, loader_data)`` as raw Lua byte strings for compatibility;
        their result is converted through the ordinary host boundary.
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
            # Preserve the established low-level package-loader contract:
            # loader names and searcher data are Lua byte strings. General
            # Python callbacks installed with set()/expose() use the friendly
            # conversion boundary instead.
            def boundary(*args):
                return self._to_lua_result(module(*args))

            loader = HostFunction(
                boundary,
                f"package.preload[{key!r}]",
            )
        else:
            raise TypeError("preloaded module must be Lua source or callable")
        self._package_state.preload.rawset(key, loader)

    @staticmethod
    def multi_return(*values):
        return MultiValue(tuple(values))

    def compile(self, source: str, *, chunkname: str | bytes = "=(luapyre)"):
        key = (source, type(chunkname), chunkname)
        if self.source_cache_size:
            cached = self._source_cache.get(key)
            if cached is not None:
                self._source_cache.move_to_end(key)
                self._source_cache_hits += 1
                return cached
        self._source_cache_misses += 1
        mode = detect_source_mode(source)
        fully_typed = mode == FULLY_TYPED_MODE
        parser = TypedParser(source) if fully_typed else Parser(source)
        proto = SourceCompiler(
            chunkname,
            fully_typed=fully_typed,
        ).compile(parser.parse())
        if self.source_cache_size:
            self._source_cache[key] = proto
            self._source_cache.move_to_end(key)
            while len(self._source_cache) > self.source_cache_size:
                self._source_cache.popitem(last=False)
        return proto

    @property
    def source_cache_info(self):
        """Return bounded source-cache counters and current occupancy."""
        return {
            "hits": self._source_cache_hits,
            "misses": self._source_cache_misses,
            "size": len(self._source_cache),
            "maxsize": self.source_cache_size,
        }

    def clear_source_cache(self) -> None:
        self._source_cache.clear()
        self._source_cache_hits = 0
        self._source_cache_misses = 0

    def execute(self, source: str, *, fuel=None, chunkname: str | bytes = "=(luapyre)"):
        return self.vm.run(self.compile(source, chunkname=chunkname), fuel=fuel)

    def execute_python(
        self,
        source: str,
        *,
        return_type=None,
        fuel=None,
        chunkname: str | bytes = "=(luapyre)",
    ):
        """Execute Lua and recursively convert its result for Python callers."""
        return self._from_lua(
            self.execute(source, fuel=fuel, chunkname=chunkname),
            return_type,
        )

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
