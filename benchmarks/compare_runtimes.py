from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import platform
from pathlib import Path
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable

from luapyre import LuaRuntime

from vm_programs import WORKLOADS, Workload


@dataclass(frozen=True, slots=True)
class Backend:
    name: str
    detail: str
    dialect: str
    prepare: Callable[[str], Callable[[], object]]


@dataclass(frozen=True, slots=True)
class Timing:
    cold_ms: float
    warmup_median_ms: float
    median_ms: float
    best_ms: float


def _luapyre_backend(*, jit: bool) -> Backend:
    label = "LuaPyre JIT" if jit else "LuaPyre interp"

    def prepare(source: str):
        runtime = LuaRuntime(fuel=20_000_000, jit=jit)
        proto = runtime.compile(source)

        def run():
            return runtime.vm.run(proto, fuel=20_000_000)

        return run

    return Backend(label, "native Python runtime", "luapyre", prepare)


def _lupa_backend(label: str, module_name: str) -> Backend | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None

    probe = module.LuaRuntime(unpack_returned_tuples=True)
    implementation = getattr(probe, "lua_implementation", module_name)
    version = getattr(module, "LUA_VERSION", "?")
    detail = f"{implementation} / {version} via {module_name}"

    def prepare(source: str):
        lua = module.LuaRuntime(unpack_returned_tuples=True)
        fn = lua.eval("function()\n" + source + "\nend")
        return fn

    return Backend(label, detail, "lua", prepare)


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
    workload: Workload,
    *,
    repeats: int,
    warmups: int,
) -> Timing:
    start = time.perf_counter_ns()
    result = run()
    cold_ms = (time.perf_counter_ns() - start) / 1_000_000
    if not workload.validate(result):
        raise AssertionError(f"expected {workload.expected!r}, got {result!r}")

    warmup_samples = []
    for _ in range(warmups):
        start = time.perf_counter_ns()
        result = run()
        warmup_samples.append((time.perf_counter_ns() - start) / 1_000_000)
        if not workload.validate(result):
            raise AssertionError(f"expected {workload.expected!r}, got {result!r}")

    samples = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = run()
            elapsed = time.perf_counter_ns() - start
            if not workload.validate(result):
                raise AssertionError(f"expected {workload.expected!r}, got {result!r}")
            samples.append(elapsed / 1_000_000)
    finally:
        if was_enabled:
            gc.enable()

    return Timing(
        cold_ms,
        statistics.median(warmup_samples) if warmup_samples else cold_ms,
        statistics.median(samples),
        min(samples),
    )


def _ratio(value: float, reference: float | None) -> str:
    if reference is None or reference <= 0.0:
        return "   n/a"
    return f"{value / reference:6.2f}x"


def _selected_workloads(groups: set[str] | None) -> dict[str, Workload]:
    if not groups:
        return WORKLOADS
    return {
        name: workload
        for name, workload in WORKLOADS.items()
        if workload.group in groups
    }


def run_suite(
    *,
    repeats: int,
    warmups: int,
    require_all: bool,
    groups: set[str] | None = None,
) -> tuple[list[Backend], dict[str, dict[str, Timing]]]:
    backends = discover_backends(require_all=require_all)
    workloads = _selected_workloads(groups)
    if not workloads:
        raise RuntimeError("no workloads matched the requested group(s)")

    print(sys.version.replace("\n", " "))
    for backend in backends:
        print(f"{backend.name:16s} {backend.detail}")
    print()

    results: dict[str, dict[str, Timing]] = {}
    for name, workload in workloads.items():
        row = {}
        for backend in backends:
            source = workload.source_for(backend.dialect)
            run = backend.prepare(source)
            row[backend.name] = measure(
                run,
                workload,
                repeats=repeats,
                warmups=warmups,
            )
        results[name] = row

    names = [backend.name for backend in backends]
    width = max(16, max(len(name) for name in workloads))
    header = f"{'workload':{width}s} " + "  ".join(
        f"{name:>16s}" for name in names
    )
    print(header)
    print("-" * len(header))
    for name, row in results.items():
        values = "  ".join(
            f"{row[backend].median_ms:13.3f} ms" for backend in names
        )
        print(f"{name:{width}s} {values}")

    print("\nLuaPyre JIT relative performance (lower ratios are better):")
    print(f"{'workload':{width}s}   vs interp   vs Lua 5.5   vs LuaJIT")
    for name, row in results.items():
        jit_ms = row["LuaPyre JIT"].median_ms
        interp = row.get("LuaPyre interp")
        lua55 = row.get("Lua 5.5")
        luajit = row.get("LuaJIT 2.1") or row.get("LuaJIT 2.0")
        print(
            f"{name:{width}s} "
            f"{_ratio(jit_ms, interp.median_ms if interp else None):>11s} "
            f"{_ratio(jit_ms, lua55.median_ms if lua55 else None):>12s} "
            f"{_ratio(jit_ms, luajit.median_ms if luajit else None):>11s}"
        )

    return backends, results


def write_json_report(
    path: Path,
    *,
    backends: list[Backend],
    results: dict[str, dict[str, Timing]],
    repeats: int,
    warmups: int,
) -> None:
    report = {
        "schema_version": 3,
        "commit": os.environ.get("GITHUB_SHA"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "repeats": repeats,
        "warmups": warmups,
        "backends": [
            {
                "name": backend.name,
                "detail": backend.detail,
                "dialect": backend.dialect,
            }
            for backend in backends
        ],
        "workloads": {
            name: {
                "group": WORKLOADS[name].group,
                "typed_luapyre_source": WORKLOADS[name].luapyre_source is not None,
                "timings": {
                    backend: {
                        "cold_ms": timing.cold_ms,
                        "warmup_median_ms": timing.warmup_median_ms,
                        "median_ms": timing.median_ms,
                        "best_ms": timing.best_ms,
                    }
                    for backend, timing in row.items()
                },
            }
            for name, row in results.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LuaPyre JIT/interpreter with Lupa Lua 5.5 and LuaJIT."
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--group",
        action="append",
        choices=sorted({workload.group for workload in WORKLOADS.values()}),
        help="run only this workload group; may be supplied more than once",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="fail instead of skipping when Lua 5.5 or LuaJIT Lupa modules are unavailable",
    )
    parser.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="also write a machine-readable JSON report",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be >= 1 and warmups must be >= 0")
    backends, results = run_suite(
        repeats=args.repeats,
        warmups=args.warmups,
        require_all=args.require_all,
        groups=set(args.group) if args.group else None,
    )
    if args.json is not None:
        write_json_report(
            args.json,
            backends=backends,
            results=results,
            repeats=args.repeats,
            warmups=args.warmups,
        )


if __name__ == "__main__":
    main()
