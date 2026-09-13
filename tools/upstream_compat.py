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
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    if (args.source is None) != (args.cjson_source is None):
        parser.error("--source and --cjson-source must be supplied together")
    if args.source is not None and args.cjson_source is not None:
        run_penlight(args.source.resolve())
        audit_lua_cjson(args.cjson_source.resolve())
        return
    elif args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        penlight = _checkout(
            args.cache_dir, "Penlight", PENLIGHT_REPOSITORY, PENLIGHT_REVISION
        )
        cjson = _checkout(
            args.cache_dir, "lua-cjson", LUA_CJSON_REPOSITORY, LUA_CJSON_REVISION
        )
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
            return
    run_penlight(penlight)
    audit_lua_cjson(cjson)


if __name__ == "__main__":
    main()
