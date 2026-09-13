from __future__ import annotations

from dataclasses import dataclass
import os

from .binary_chunks import NATIVE_MAGIC, PUC_MAGIC
from .bytecode import Closure
from .capabilities import RuntimeCapabilities
from .compiler import Compiler
from .errors import LuaRaisedError, LuaRuntimeError
from .parser import Parser
from .stdlib_support import need_bytes
from .table import LuaTable
from .values import MultiValue, truthy
from .vm import HostFunction


def _module_name(value, *, arg=1, function="require") -> bytes:
    return need_bytes(value, arg, function)


def _templates(path: bytes):
    for item in path.split(b";"):
        if item:
            yield item


def _module_path(name: bytes, template: bytes) -> bytes:
    return template.replace(b"?", name.replace(b".", b"/"))


@dataclass(slots=True)
class PackageState:
    globals_table: LuaTable
    vm: object
    capabilities: RuntimeCapabilities
    package: LuaTable
    loaded: LuaTable
    preload: LuaTable
    searchers: LuaTable
    loadfile_host: HostFunction
    dofile_host: object
    file_searcher_host: HostFunction

    def refresh_file_capability(self) -> None:
        # These functions can always access the runtime's private in-memory
        # filesystem. A host loader merely adds an explicit read capability.
        enabled = True
        self.globals_table.rawset(b"loadfile", self.loadfile_host if enabled else None)
        self.globals_table.rawset(b"dofile", self.dofile_host if enabled else None)
        self.searchers.rawset(2, self.file_searcher_host if enabled else None)


def install_package_library(
    globals_table: LuaTable,
    vm,
    capabilities: RuntimeCapabilities,
) -> PackageState:
    """Install a sandbox-safe package library.

    The default runtime can search only its private in-memory filesystem.
    An explicit host file-loader capability adds read access chosen by the
    embedder. Native/C loading is never exposed.
    """

    package = LuaTable()
    loaded = LuaTable()
    preload = LuaTable()
    searchers = LuaTable()
    package.rawset(b"loaded", loaded)
    package.rawset(b"preload", preload)
    package.rawset(b"searchers", searchers)
    package.rawset(b"path", b"?.lua;?/init.lua")
    package.rawset(b"cpath", b"")
    package.rawset(
        b"config",
        os.sep.encode("ascii", "replace") + b"\n;\n?\n!\n-\n",
    )

    def preload_searcher(name):
        name = _module_name(name, function="require")
        loader = preload.rawget(name)
        if loader is None:
            return f"no field package.preload['{name.decode('utf-8', 'replace')}']".encode()
        return MultiValue((loader, b":preload:"))

    searchers.rawset(1, HostFunction(preload_searcher, "package.preload searcher"))

    def _load_source(source: bytes, filename: bytes, mode=b"bt", env_args=()):
        if source.startswith(b"\xef\xbb\xbf"):
            source = source[3:]
        if source.startswith(b"#"):
            newline = source.find(b"\n")
            remainder = source[newline + 1:] if newline >= 0 else b""
            if remainder.startswith((NATIVE_MAGIC, PUC_MAGIC)):
                source = remainder
            else:
                source = b"--" + source[1:]
        load_host = globals_table.rawget(b"load")
        args = (source, b"@" + filename, mode, *env_args)
        return vm.call_sync(load_host, args)

    def loadfile(filename, *optional):
        filename = need_bytes(filename, 1, "loadfile")
        mode = optional[0] if optional else b"bt"
        env_args = optional[1:2]
        source, error = capabilities.read_file(filename)
        if source is None:
            return MultiValue((None, error))
        results = _load_source(source, filename, mode, env_args)
        return MultiValue(tuple(results))

    loadfile_host = HostFunction(loadfile, "loadfile")

    # The loaded chunk is invoked from Lua so dofile remains yieldable.
    factory = Compiler().compile(Parser("""
        local loadfile = ...
        return function (filename)
          local chunk, message = loadfile(filename)
          if chunk == nil then error(message, 0) end
          return chunk()
        end
    """).parse())
    dofile_host = vm.call_sync(
        Closure(factory, [], globals_table), (loadfile_host,), fuel=1_000
    )[0]

    def _find_file(name: bytes):
        path = package.rawget(b"path")
        if not isinstance(path, bytes):
            raise LuaRuntimeError("'package.path' must be a string")
        errors: list[bytes] = []
        for template in _templates(path):
            candidate = _module_path(name, template)
            source, error = capabilities.read_file(candidate)
            if source is not None:
                return candidate, source
            errors.append(b"no file '" + candidate + b"'")
        return None, b"\n\t".join(errors)

    def file_searcher(name):
        name = _module_name(name, function="require")
        candidate, source = _find_file(name)
        if candidate is None:
            return source
        results = _load_source(source, candidate)
        loader = results[0] if results else None
        if loader is None:
            error = results[1] if len(results) > 1 else b"error loading module"
            raise LuaRaisedError(error, located=True)
        return MultiValue((loader, candidate))

    file_searcher_host = HostFunction(file_searcher, "package Lua searcher")

    def native_searcher(name):
        # Report standard C-searcher candidates without loading native code.
        # This preserves require's diagnostics while keeping the sandbox from
        # importing shared libraries.
        name = _module_name(name, function="require")
        cpath = package.rawget(b"cpath")
        if not isinstance(cpath, bytes):
            raise LuaRuntimeError("'package.cpath' must be a string")
        module = name.replace(b".", os.sep.encode("ascii", "replace"))
        errors = [b"no file '" + template.replace(b"?", module) + b"'"
                  for template in _templates(cpath)]
        return b"\n\t".join(errors)

    searchers.rawset(3, HostFunction(native_searcher, "package C searcher"))

    def searchpath(name, path, sep=b".", rep=b"/"):
        name = need_bytes(name, 1, "searchpath")
        path = need_bytes(path, 2, "searchpath")
        sep = need_bytes(sep, 3, "searchpath")
        rep = need_bytes(rep, 4, "searchpath")
        module = name.replace(sep, rep)
        errors = []
        for template in _templates(path):
            candidate = template.replace(b"?", module)
            source, _error = capabilities.read_file(candidate)
            if source is not None:
                return candidate
            errors.append(b"\n\tno file '" + candidate + b"'")
        return MultiValue((None, b"".join(errors)))

    package.rawset(b"searchpath", HostFunction(searchpath, "package.searchpath"))

    def require(name):
        name = _module_name(name, function="require")
        current = loaded.rawget(name)
        if truthy(current):
            return current

        errors: list[bytes] = []
        index = 1
        loader = None
        loader_data = None
        while True:
            searcher = searchers.rawget(index)
            if searcher is None:
                suffix = b"\n\t".join(errors)
                message = b"module '" + name + b"' not found:"
                if suffix:
                    message += b"\n\t" + suffix
                raise LuaRuntimeError(message.decode("utf-8", "replace"))
            results = vm.call_sync(searcher, (name,))
            first = results[0] if results else None
            second = results[1] if len(results) > 1 else None
            if isinstance(first, (Closure, HostFunction)):
                loader, loader_data = first, second
                break
            if isinstance(first, bytes):
                errors.append(first)
            index += 1

        results = vm.call_sync(loader, (name, loader_data))
        result = results[0] if results else None
        if result is not None:
            loaded.rawset(name, result)
        module = loaded.rawget(name)
        if module is None:
            module = True
            loaded.rawset(name, module)
        return MultiValue((module, loader_data))

    globals_table.rawset(b"package", package)
    globals_table.rawset(b"require", HostFunction(require, "require"))

    # Mirror the standard registry's loaded table for libraries that are
    # already present in this sandboxed runtime.
    loaded.rawset(b"_G", globals_table)
    loaded.rawset(b"package", package)
    for name in (b"coroutine", b"debug", b"io", b"math", b"os", b"string", b"table", b"utf8"):
        value = globals_table.rawget(name)
        if value is not None:
            loaded.rawset(name, value)

    state = PackageState(
        globals_table,
        vm,
        capabilities,
        package,
        loaded,
        preload,
        searchers,
        loadfile_host,
        dofile_host,
        file_searcher_host,
    )
    state.refresh_file_capability()
    return state
