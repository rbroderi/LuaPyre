#!/usr/bin/env python3
"""Run pinned, unchanged tests from real Lua packages through LuaPyre."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile
import time

from luapyre import LuaRuntime, LuaRuntimeError, LuaTable, MultiValue
from luapyre.bytecode import Closure
from luapyre.vm import HostFunction


PENLIGHT_REPOSITORY = "https://github.com/lunarmodules/Penlight.git"
PENLIGHT_REVISION = "dfa483dddc0d751be1047667f4580dd90319169c"
LUA_CJSON_REPOSITORY = "https://github.com/mpx/lua-cjson.git"
LUA_CJSON_REVISION = "718f27293a981fb5e9e662e9aec0b7cf78317da6"
LUATEST_REPOSITORY = "https://github.com/mblayman/luatest.git"
LUATEST_REVISION = "d063c547b31d4df1dce1b0679ccf964d53ff5c1e"
LUACOV_REPOSITORY = "https://github.com/lunarmodules/luacov.git"
LUACOV_REVISION = "645a98468aad737de035a49a578c245bb3e555fb"
AWFY_REPOSITORY = "https://github.com/smarr/are-we-fast-yet.git"
AWFY_REVISION = "74306fec151070fd07157cefeacf19e7e0bcdc89"
PENLIGHT_PORTABLE_TESTS = (
    "test-__vector.lua",
    "test-class.lua",
    "test-class3.lua",
    "test-class4.lua",
    "test-compat.lua",
    "test-comprehension.lua",
    "test-config.lua",
    "test-data2.lua",
    "test-func.lua",
    "test-import_into.lua",
    "test-lexer.lua",
    "test-list.lua",
    "test-list2.lua",
    "test-map.lua",
    "test-orderedmap.lua",
    "test-seq.lua",
    "test-sip.lua",
    "test-stringio.lua",
    "test-tablex.lua",
    "test-tablex3.lua",
    "test-template.lua",
    "test-template2.lua",
    "test-url.lua",
)
AWFY_BENCHMARKS = (
    "bounce",
    "deltablue",
    "json",
    "list",
    "mandelbrot",
    "nbody",
    "permute",
    "queens",
    "sieve",
    "storage",
    "towers",
)


def _table(**fields) -> LuaTable:
    result = LuaTable()
    for name, value in fields.items():
        if callable(value):
            value = HostFunction(value, name)
        result.rawset(name.encode("ascii"), value)
    return result


def _install_test_harness_modules(runtime: LuaRuntime, output: list[bytes]) -> None:
    """Supply only the host objects Penlight's upstream test helper imports."""

    def write(*values):
        if values and isinstance(values[0], LuaTable):
            values = values[1:]
        output.append(
            b"".join(
                value if isinstance(value, bytes) else str(value).encode("utf-8")
                for value in values
            )
        )

    stdout = _table(write=write)
    stderr = _table(write=write)
    io = _table(
        stdout=stdout,
        stderr=stderr,
        write=write,
        type=lambda value: b"file" if value in (stdout, stderr) else None,
    )

    def debug_getupvalue(fn, index):
        if not isinstance(fn, Closure):
            return None
        descriptors = fn.proto.upvalues
        if 1 <= index <= len(descriptors):
            descriptor = descriptors[index - 1]
            return MultiValue((descriptor.name.encode("utf-8"), fn.upvalues[index - 1].value))
        if index == len(descriptors) + 1:
            return MultiValue((b"_ENV", fn.env))
        return None

    def debug_setupvalue(fn, index, value):
        if not isinstance(fn, Closure):
            return None
        descriptors = fn.proto.upvalues
        if 1 <= index <= len(descriptors):
            fn.upvalues[index - 1].value = value
            return descriptors[index - 1].name.encode("utf-8")
        if index == len(descriptors) + 1:
            fn.env = value
            return b"_ENV"
        return None

    def debug_getinfo(target, *_options):
        if isinstance(target, Closure):
            closure = target
            line = closure.proto.linedefined
        else:
            frames = runtime.vm._active_frames or ()
            level = int(target) if type(target) in (int, float) else 1
            frame = frames[-level] if 0 < level <= len(frames) else None
            closure = frame.closure if frame is not None else None
            line = frame.proto.line_for_pc(frame.pc) if frame is not None else 0
        source = closure.proto.source if closure is not None else b"?"
        if isinstance(source, str):
            source = source.encode("utf-8")
        return _table(short_src=source, currentline=line, func=closure)

    debug = _table(
        getinfo=debug_getinfo,
        getupvalue=debug_getupvalue,
        setupvalue=debug_setupvalue,
        gethook=lambda: MultiValue((None, b"", 0)),
        sethook=lambda *_args: None,
        traceback=lambda *_args: b"",
    )
    lfs = _table(
        attributes=lambda *_args: None,
        symlinkattributes=lambda *_args: None,
        currentdir=lambda: b".",
    )
    os_library = _table(
        clock=time.process_time,
        getenv=lambda *_args: None,
        tmpname=lambda: b"/tmp/luapyre-penlight",
        exit=lambda code=0, *_args: (_ for _ in ()).throw(
            LuaRuntimeError(f"upstream test requested os.exit({code})")
        ),
    )

    for name, value in ((b"io", io), (b"os", os_library), (b"debug", debug)):
        runtime.globals.rawset(name, value)
    loaded = runtime.get("package").rawget(b"loaded")
    loaded.rawset(b"debug", debug)
    loaded.rawset(b"lfs", lfs)


def _checkout(destination: Path, name: str, repository: str, revision: str) -> Path:
    checkout = destination / name
    if not checkout.exists():
        subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none", repository, checkout],
            check=True,
        )
    subprocess.run(
        ["git", "-C", checkout, "checkout", "--quiet", revision],
        check=True,
    )
    return checkout.resolve()


def run_penlight(root: Path) -> None:
    failures: list[str] = []
    for filename in PENLIGHT_PORTABLE_TESTS:
        test_file = root / "tests" / filename
        output: list[bytes] = []

        def load_file(name: str):
            candidate = (root / name).resolve()
            if root not in candidate.parents:
                return None
            try:
                return candidate.read_bytes()
            except OSError:
                return None

        runtime = LuaRuntime(
            file_loader=load_file,
            output=output.append,
            fuel=100_000_000,
            max_frames=5_000,
        )
        _install_test_harness_modules(runtime, output)
        runtime.get("package").rawset(
            b"path", b"lua/?.lua;lua/?/init.lua;tests/lua/?.lua"
        )
        arguments = LuaTable()
        arguments.rawset(0, ("tests/" + filename).encode("utf-8"))
        runtime.globals.rawset(b"arg", arguments)
        try:
            runtime.execute(
                test_file.read_text(encoding="utf-8"),
                chunkname="@tests/" + filename,
            )
        except Exception as error:
            failures.append(f"{filename}: {type(error).__name__}: {error}")
            print(f"FAIL  {filename}")
        else:
            print(f"PASS  {filename}")

    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"Penlight {PENLIGHT_REVISION}: "
        f"{len(PENLIGHT_PORTABLE_TESTS)}/{len(PENLIGHT_PORTABLE_TESTS)} portable tests passed"
    )


def audit_lua_cjson(root: Path) -> None:
    """Confirm the real upstream suite reaches the unsupported native module."""
    source = (root / "tests" / "test.lua").read_text(encoding="utf-8")
    runtime = LuaRuntime(fuel=100_000_000)
    try:
        runtime.execute(source, chunkname="@tests/test.lua")
    except LuaRuntimeError as error:
        if "module 'cjson' not found" not in str(error):
            raise
    else:
        raise RuntimeError("lua-cjson unexpectedly loaded without a native module bridge")
    print(
        f"UNSUPPORTED  lua-cjson {LUA_CJSON_REVISION}: "
        "upstream tests require its compiled Lua C module"
    )


def _read_only_loader(*roots: Path):
    roots = tuple(root.resolve() for root in roots)

    def load_file(name: str):
        for root in roots:
            candidate = (root / name).resolve()
            if candidate != root and root not in candidate.parents:
                continue
            try:
                return candidate.read_bytes()
            except OSError:
                pass
        return None

    return load_file


def run_luatest(root: Path) -> None:
    """Run unchanged core tests with adapters for LuaRocks dependencies."""
    runtime = LuaRuntime(
        file_loader=_read_only_loader(root), fuel=100_000_000, max_frames=10_000
    )
    runtime.get("package").rawset(
        b"path", b"./lua/?.lua;./lua/?/init.lua;./tests/?.lua;./?.lua"
    )
    runtime.preload(
        "pl.tablex",
        """return {sort=function(t)
 local keys={}
 for k in pairs(t) do keys[#keys+1]=k end
 table.sort(keys)
 local i=0
 return function()
  i=i+1
  local k=keys[i]
  if k ~= nil then return k,t[k] end
 end
end}""",
    )
    for name in (
        "argparse",
        "inspect",
        "luatest.collection",
        "luatest.configuration",
        "luatest.coverage",
    ):
        runtime.preload(name, lambda *_args: LuaTable())
    runtime.preload(
        "luatest.reporter",
        """local Reporter={}
setmetatable(Reporter,{__call=function()
 return {
  start_module=function() end,
  start_test=function() end,
  finish_test=function() end,
  finish_module=function() end,
  finish_execution=function() end
 }
end})
return Reporter""",
    )

    assertion = LuaTable()

    def is_equal(actual, expected):
        if actual != expected:
            raise LuaRuntimeError(f"expected {actual!r} == {expected!r}")

    def is_true(value):
        if not value:
            raise LuaRuntimeError(
                "Expected objects to be the same.\n"
                "Passed in:\n(boolean) false\n"
                "Expected:\n(boolean) true"
            )

    assertion.rawset(b"is_equal", HostFunction(is_equal, "assert.is_equal", max_args=2))
    assertion.rawset(b"is_true", HostFunction(is_true, "assert.is_true", max_args=1))
    runtime.preload("luassert", lambda *_args: assertion)
    runtime.preload(
        "luassert.stub",
        """return function(target, key)
 local calls={}
 local spy={_calls=calls}
 setmetatable(spy,{__call=function(_, ...) calls[#calls+1]={...} end})
 target[key]=spy
 return spy
end""",
    )

    def assert_stub(spy):
        result = LuaTable()

        def was_called_with(*expected):
            calls = spy.rawget(b"_calls")
            for _, arguments in calls.items():
                actual = [value for _, value in sorted(arguments.items())]
                while len(actual) < len(expected):
                    actual.append(None)
                if len(actual) == len(expected) and all(
                    left == right for left, right in zip(actual, expected)
                ):
                    return None
            raise LuaRuntimeError("expected call not found")

        result.rawset(
            b"was_called_with", HostFunction(was_called_with, "was_called_with")
        )
        return result

    assertion.rawset(
        b"stub", HostFunction(assert_stub, "assert.stub", max_args=1)
    )

    passed = 0
    for filename in ("tests/test_main.lua", "tests/test_executor.lua"):
        tests = runtime.execute(
            (root / filename).read_text(encoding="utf-8"), chunkname="@" + filename
        )
        for test_name, test in tests.items():
            runtime.vm.call_sync(test, ())
            passed += 1
            print(f"PASS  {filename}::{test_name.decode('utf-8')}")
    if passed != 5:
        raise RuntimeError(f"expected 5 luatest core tests, ran {passed}")
    print(f"luatest {LUATEST_REVISION}: {passed}/5 core module tests passed")


def run_luacov(root: Path) -> None:
    """Run the unchanged pure-Lua scanner spec and audit collector support."""
    runtime = LuaRuntime(
        file_loader=_read_only_loader(root), fuel=100_000_000, max_frames=10_000
    )
    runtime.get("package").rawset(b"path", b"src/?.lua;src/?/init.lua")
    passed: list[bytes] = []
    failures: list[tuple[bytes, Exception]] = []

    def call_zero(fn):
        return runtime.vm.call_sync(fn, ())

    def describe(_name, fn):
        return call_zero(fn)

    def it(name, fn):
        try:
            call_zero(fn)
        except Exception as error:
            failures.append((name, error))
        else:
            passed.append(name)

    assertion = LuaTable()

    def is_table(value):
        if not isinstance(value, LuaTable):
            raise LuaRuntimeError("expected a table")

    def is_equal(expected, actual):
        if expected != actual:
            raise LuaRuntimeError(f"{expected!r} != {actual!r}")

    assertion.rawset(b"is_table", HostFunction(is_table, "assert.is_table", max_args=1))
    assertion.rawset(b"is_equal", HostFunction(is_equal, "assert.is_equal", max_args=2))
    runtime.globals.rawset(b"assert", assertion)
    runtime.globals.rawset(b"describe", HostFunction(describe, "describe", max_args=2))
    runtime.globals.rawset(b"it", HostFunction(it, "it", max_args=2))
    runtime.execute(
        (root / "spec" / "linescanner_spec.lua").read_text(encoding="utf-8"),
        chunkname="@spec/linescanner_spec.lua",
    )
    if failures:
        raise RuntimeError(
            "\n".join(
                f"{name.decode('utf-8')}: {type(error).__name__}: {error}"
                for name, error in failures
            )
        )
    if len(passed) != 24:
        raise RuntimeError(f"expected 24 LuaCov scanner specs, ran {len(passed)}")
    print(f"LuaCov {LUACOV_REVISION}: {len(passed)}/24 line-scanner specs passed")

    audit = LuaRuntime(file_loader=_read_only_loader(root), fuel=10_000_000)
    audit.get("package").rawset(b"path", b"src/?.lua;src/?/init.lua")
    try:
        audit.execute('require "luacov"')
    except LuaRuntimeError as error:
        if "module 'debug' not found" not in str(error):
            raise
    else:
        raise RuntimeError("LuaCov collector unexpectedly loaded without debug hooks")
    print("PARTIAL  LuaCov collection requires debug.sethook and report-file I/O")


def run_awfy(root: Path) -> None:
    """Run upstream's native Lua harness and built-in result checks."""
    lua_root = (root / "benchmarks" / "Lua").resolve()
    harness = (lua_root / "harness.lua").read_text(encoding="utf-8")
    for benchmark in AWFY_BENCHMARKS:
        output: list[bytes] = []
        runtime = LuaRuntime(
            file_loader=_read_only_loader(lua_root),
            output=output.append,
            fuel=500_000_000,
            max_frames=10_000,
        )
        runtime.get("package").rawset(b"path", b"?.lua")
        runtime.globals.rawset(b"os", _table(clock=time.process_time))
        arguments = LuaTable()
        arguments.rawset(0, b"harness.lua")
        arguments.rawset(1, benchmark.encode("ascii"))
        arguments.rawset(2, 1)
        arguments.rawset(3, 1)
        runtime.globals.rawset(b"arg", arguments)
        runtime.execute(harness, chunkname="@harness.lua")
        print(f"PASS  are-we-fast-yet::{benchmark}")
    print(
        f"Are We Fast Yet {AWFY_REVISION}: "
        f"{len(AWFY_BENCHMARKS)}/{len(AWFY_BENCHMARKS)} benchmarks passed"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        help="existing checkout at the pinned Penlight revision",
    )
    parser.add_argument(
        "--cjson-source",
        type=Path,
        help="existing checkout at the pinned lua-cjson revision",
    )
    parser.add_argument("--luatest-source", type=Path)
    parser.add_argument("--luacov-source", type=Path)
    parser.add_argument("--awfy-source", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    supplied = (
        args.source,
        args.cjson_source,
        args.luatest_source,
        args.luacov_source,
        args.awfy_source,
    )
    if any(source is not None for source in supplied) and not all(
        source is not None for source in supplied
    ):
        parser.error("all five --*-source checkouts must be supplied together")
    if all(source is not None for source in supplied):
        run_penlight(args.source.resolve())
        audit_lua_cjson(args.cjson_source.resolve())
        run_luatest(args.luatest_source.resolve())
        run_luacov(args.luacov_source.resolve())
        run_awfy(args.awfy_source.resolve())
        return
    elif args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        penlight = _checkout(
            args.cache_dir, "Penlight", PENLIGHT_REPOSITORY, PENLIGHT_REVISION
        )
        cjson = _checkout(
            args.cache_dir, "lua-cjson", LUA_CJSON_REPOSITORY, LUA_CJSON_REVISION
        )
        luatest = _checkout(
            args.cache_dir, "luatest", LUATEST_REPOSITORY, LUATEST_REVISION
        )
        luacov = _checkout(
            args.cache_dir, "luacov", LUACOV_REPOSITORY, LUACOV_REVISION
        )
        awfy = _checkout(args.cache_dir, "are-we-fast-yet", AWFY_REPOSITORY, AWFY_REVISION)
    else:
        with tempfile.TemporaryDirectory(prefix="luapyre-upstream-") as directory:
            temporary = Path(directory)
            run_penlight(
                _checkout(
                    temporary, "Penlight", PENLIGHT_REPOSITORY, PENLIGHT_REVISION
                )
            )
            audit_lua_cjson(
                _checkout(
                    temporary,
                    "lua-cjson",
                    LUA_CJSON_REPOSITORY,
                    LUA_CJSON_REVISION,
                )
            )
            run_luatest(
                _checkout(temporary, "luatest", LUATEST_REPOSITORY, LUATEST_REVISION)
            )
            run_luacov(
                _checkout(temporary, "luacov", LUACOV_REPOSITORY, LUACOV_REVISION)
            )
            run_awfy(
                _checkout(temporary, "are-we-fast-yet", AWFY_REPOSITORY, AWFY_REVISION)
            )
            return
    run_penlight(penlight)
    audit_lua_cjson(cjson)
    run_luatest(luatest)
    run_luacov(luacov)
    run_awfy(awfy)


if __name__ == "__main__":
    main()
