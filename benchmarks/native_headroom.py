"""Compare LuaPyre with native Lua 5.5 and reduced-contract Python.

The native column runs the same untyped Lua source through Lupa's pinned
``lupa.lua55`` backend. The Python functions preserve each algorithm but omit
Lua runtime semantics, so they are engineering lower bounds rather than
equivalent implementations or mathematical limits.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from lupa.lua55 import LuaRuntime as NativeLuaRuntime

from python_headroom import REFERENCES, prepare as prepare_luapyre
from vm_programs import WORKLOADS


ORDERS = (
    ("luapyre", "native_lua55", "python_lower_bound"),
    ("native_lua55", "python_lower_bound", "luapyre"),
    ("python_lower_bound", "luapyre", "native_lua55"),
)


def prepare_native(name):
    runtime = NativeLuaRuntime(unpack_returned_tuples=True)
    version = runtime.eval("_VERSION")
    if version != "Lua 5.5":
        raise RuntimeError(f"expected Lua 5.5, got {version!r}")
    if name == "python_calls_1000":
        function = runtime.execute("return function(x) return x + 1 end")

        def run():
            for value in range(1000):
                assert function(value) == value + 1
            return 1000

        validate = lambda result: result == 1000
    else:
        workload = WORKLOADS[name]
        function = runtime.execute(
            "return function()\n" + workload.source_for("lua") + "\nend"
        )
        run = function
        validate = workload.validate
    return runtime, run, validate


def measure(run, validate, warmups, repeats):
    for _ in range(warmups):
        assert validate(run())
    samples = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = run()
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
            assert validate(result), result
    finally:
        if was_enabled:
            gc.enable()
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def run_worker(names, order, warmups, repeats):
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {affinity})
    results = {}
    for name in names:
        luapyre_runtime, luapyre_run, validate = prepare_luapyre(name)
        native_runtime, native_run, native_validate = prepare_native(name)
        runners = {
            "luapyre": luapyre_run,
            "native_lua55": native_run,
            "python_lower_bound": REFERENCES[name],
        }
        row = {
            label: measure(runners[label], validate, warmups, repeats)
            for label in order
        }
        # Retain both runtimes until all samples and the warm-counter probe end.
        assert native_runtime.eval("_VERSION") == "Lua 5.5"
        assert native_validate(native_run())
        before = asdict(luapyre_runtime.jit_stats)
        assert validate(luapyre_run())
        after = asdict(luapyre_runtime.jit_stats)
        row["luapyre_warm_counters"] = {
            key: after[key] - value
            for key, value in before.items()
            if type(value) is int and after[key] != value
        }
        results[name] = row
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "affinity_cpu": affinity,
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
        "order": list(order),
        "workloads": results,
    }


def aggregate(revision, runs, warmups, repeats):
    results = {}
    for name in runs[0]["workloads"]:
        row = {}
        for label in ORDERS[0]:
            medians = [run["workloads"][name][label]["median_ms"] for run in runs]
            row[label] = {
                "median_ms": statistics.median(medians),
                "process_medians_ms": medians,
            }
        row["luapyre_over_native"] = (
            row["luapyre"]["median_ms"] / row["native_lua55"]["median_ms"]
        )
        row["luapyre_over_python_lower_bound"] = (
            row["luapyre"]["median_ms"] / row["python_lower_bound"]["median_ms"]
        )
        row["native_over_python_lower_bound"] = (
            row["native_lua55"]["median_ms"]
            / row["python_lower_bound"]["median_ms"]
        )
        results[name] = row
    return {
        "schema_version": 1,
        "revision": revision,
        "python": runs[0]["python"],
        "native_runtime": "Lua 5.5 via lupa.lua55",
        "platform": runs[0]["platform"],
        "warmups": warmups,
        "repeats": repeats,
        "processes": len(runs),
        "method": (
            f"{len(runs)} isolated processes; rotated implementation order; fixed "
            "CPU affinity and PYTHONHASHSEED=0; Python GC disabled during timing; "
            "Lua GC active; every result checked; median of process medians"
        ),
        "interpretation": (
            "Native Lua uses the same untyped Lua algorithm. Python lower bounds "
            "use the same algorithm but omit Lua runtime semantics and are not "
            "attainable-speed guarantees or mathematical lower bounds."
        ),
        "runs": runs,
        "workloads": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--warmups", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--case", action="append", choices=REFERENCES)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--order", choices=range(len(ORDERS)), type=int, help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 1 or args.processes < 1:
        parser.error("warmups, repeats, and processes must be positive")
    names = args.case or list(REFERENCES)
    if args.worker:
        if args.order is None:
            parser.error("workers require --order")
        json.dump(
            run_worker(names, ORDERS[args.order], args.warmups, args.repeats),
            sys.stdout,
        )
        return
    if args.json is None:
        parser.error("parent runs require --json")

    runs = []
    for process_index in range(args.processes):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--revision",
            args.revision,
            "--worker",
            "--order",
            str(process_index % len(ORDERS)),
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
        ]
        for name in args.case or ():
            command.extend(("--case", name))
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, env=environment
        )
        runs.append(json.loads(completed.stdout))
        print(
            f"{runs[-1]['python']} process {process_index + 1}/{args.processes}: "
            f"{'/'.join(runs[-1]['order'])}",
            flush=True,
        )
    report = aggregate(args.revision, runs, args.warmups, args.repeats)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    for name, row in report["workloads"].items():
        print(
            f"{name}: {row['luapyre_over_native']:.2f}x native Lua 5.5; "
            f"{row['luapyre_over_python_lower_bound']:.2f}x Python lower bound"
        )


if __name__ == "__main__":
    main()
