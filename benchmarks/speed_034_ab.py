"""Measure the accepted 0.34 targets and controls in one isolated process."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform

from luapyre import LuaRuntime
from speed_030_ab import median_ms
from vm_programs import WORKLOADS


CASES = (
    "table_mix",
    "sieve",
    "string_build",
    "binary_trees",
    "fib_recursive",
    "spectral_norm",
    "python_calls_1000",
    "typed_arith",
    "typed_branch",
)


def prepare(name: str):
    runtime = LuaRuntime(fuel=20_000_000)
    if name == "python_calls_1000":
        function = runtime.execute_python(
            "-- luapyre: typed\n"
            "return function(x: integer): integer return x + 1 end"
        )

        def run():
            for value in range(1000):
                assert function(value) == value + 1

        return run

    workload = WORKLOADS[name]
    proto = runtime.compile(workload.source_for("luapyre"))

    def run():
        result = runtime.vm.run(proto, fuel=20_000_000)
        assert workload.validate(result), (name, result)

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--case", action="append", choices=CASES)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 1:
        parser.error("warmups and repeats must be positive")

    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {affinity})

    results = {}
    for name in args.case or CASES:
        results[name] = median_ms(prepare(name), args.repeats, args.warmups)
        print(f"{args.label}\t{name}\t{results[name]:.6f}", flush=True)

    args.json.write_text(json.dumps({
        "schema_version": 1,
        "label": args.label,
        "revision": args.revision,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "affinity_cpu": affinity,
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "method": "median elapsed milliseconds; Python GC disabled only during timed samples",
        "validation": "Expected workload result checked on every execution",
        "median_ms": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
