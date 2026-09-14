"""Measure the 0.32 compiled-call pipeline against an earlier checkout."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from luapyre import LuaRuntime
from speed_030_ab import median_ms
from vm_programs import WORKLOADS


NAMES = ("fib_recursive", "spectral_norm", "binary_trees", "table_mix")


def workload_runner(name: str):
    workload = WORKLOADS[name]
    runtime = LuaRuntime(fuel=20_000_000)
    proto = runtime.compile(workload.source_for("luapyre"))
    assert workload.validate(runtime.vm.run(proto, fuel=20_000_000))

    def run():
        result = runtime.vm.run(proto, fuel=20_000_000)
        assert workload.validate(result)

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative and repeats positive")
    results = {
        name: median_ms(workload_runner(name), args.repeats, args.warmups)
        for name in NAMES
    }
    for name, value in results.items():
        print(f"{args.label}\t{name}\t{value:.6f}")
    if args.json:
        args.json.write_text(json.dumps({
            "label": args.label,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "warmups": args.warmups,
            "repeats": args.repeats,
            "median_ms": results,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
