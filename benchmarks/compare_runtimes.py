from __future__ import annotations

import argparse
import gc
import importlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable

from luapyre import LuaRuntime

from vm_programs import WORKLOADS


@dataclass(frozen=True, slots=True)
class Backend:
    name: str
    detail: str
    prepare: Callable[[str, object], Callable[[], object]]


@dataclass(frozen=True, slots=True)
class Timing:
    median_ms: float
    best_ms: float


def _luapyre_backend(*, jit: bool) -> Backend:
    label = "LuaPyre JIT" if jit else "LuaPyre interp"

    def prepare(source: str, expected: object):
        runtime = LuaRuntime(fuel=20_000_000, jit=jit)
        proto = runtime.compile(source)

        def run():
            return runtime.vm.run(proto, fuel=20_000_000)

        return run

    return Backend(label, "native Python runtime", prepare)


def _lupa_backend(label: str, module_name: str) -> Backend | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None

    probe = module.LuaRuntime(unpack_returned_tuples=True)
    implementation = getattr(probe, "lua_implementation", module_name)
    version = getattr(module, "LUA_VERSION", "?")
    detail = f"{implementation} / {version} via {module_name}"

    def prepare(source: str, expected: object):
        lua = module.LuaRuntime(unpack_returned_tuples=True)
        fn = lua.eval("function()\n" + source + "\nend")
        return fn

    return Backend(label, detail, prepare)


def discover_backends(require_all: bool = False) -> list[Backend]:
    backends = [_luapyre_backend(jit=True), _luapyre_backend(jit=False)]

    lua55 = _lupa_backend("Lua 5.5", "lupa.lua55")
    luajit = _lupa_backend("LuaJIT 2.1", "lupa.luajit21")
    if luajit is None:
        luajit = _lupa_backend("LuaJIT 2.0", "lupa.luajit20")

    missing = []
    if lua55 is None:
        missing.append("lupa.lua55")
    else:
        backends.append(lua55)
    if luajit is None:
        missing.append("lupa.luajit21/lupa.luajit20")
    else:
        backends.append(luajit)

    if require_all and missing:
        raise RuntimeError("missing benchmark backends: " + ", ".join(missing))
    if missing:
        print("Skipping unavailable backend(s): " + ", ".join(missing), file=sys.stderr)
    return backends


def measure(
    run: Callable[[], object],
    expected: object,
    *,
    repeats: int,
    warmups: int,
) -> Timing:
    for _ in range(warmups):
        result = run()
        if result != expected:
            raise AssertionError(f"expected {expected!r}, got {result!r}")

    samples = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = run()
            elapsed = time.perf_counter_ns() - start
            if result != expected:
                raise AssertionError(f"expected {expected!r}, got {result!r}")
            samples.append(elapsed / 1_000_000)
    finally:
        if was_enabled:
            gc.enable()

    return Timing(statistics.median(samples), min(samples))


def _ratio(value: float, reference: float | None) -> str:
    if reference is None or reference <= 0.0:
        return "   n/a"
    return f"{value / reference:6.2f}x"


def run_suite(*, repeats: int, warmups: int, require_all: bool) -> None:
    backends = discover_backends(require_all=require_all)
    print(sys.version.replace("\n", " "))
    for backend in backends:
        print(f"{backend.name:16s} {backend.detail}")
    print()

    results: dict[str, dict[str, Timing]] = {}
    for workload, (source, expected) in WORKLOADS.items():
        row = {}
        for backend in backends:
            run = backend.prepare(source, expected)
            row[backend.name] = measure(
                run,
                expected,
                repeats=repeats,
                warmups=warmups,
            )
        results[workload] = row

    names = [backend.name for backend in backends]
    header = "workload     " + "  ".join(f"{name:>16s}" for name in names)
    print(header)
    print("-" * len(header))
    for workload, row in results.items():
        values = "  ".join(
            f"{row[name].median_ms:13.3f} ms" for name in names
        )
        print(f"{workload:12s} {values}")

    print("\nLuaPyre JIT relative performance (lower ratios are better):")
    print("workload       vs interp   vs Lua 5.5   vs LuaJIT")
    for workload, row in results.items():
        jit_ms = row["LuaPyre JIT"].median_ms
        interp = row.get("LuaPyre interp")
        lua55 = row.get("Lua 5.5")
        luajit = row.get("LuaJIT 2.1") or row.get("LuaJIT 2.0")
        print(
            f"{workload:12s} "
            f"{_ratio(jit_ms, interp.median_ms if interp else None):>11s} "
            f"{_ratio(jit_ms, lua55.median_ms if lua55 else None):>12s} "
            f"{_ratio(jit_ms, luajit.median_ms if luajit else None):>11s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LuaPyre JIT/interpreter with Lupa Lua 5.5 and LuaJIT."
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="fail instead of skipping when Lua 5.5 or LuaJIT Lupa modules are unavailable",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be >= 1 and warmups must be >= 0")
    run_suite(
        repeats=args.repeats,
        warmups=args.warmups,
        require_all=args.require_all,
    )


if __name__ == "__main__":
    main()
