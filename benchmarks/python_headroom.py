"""Compare LuaPyre with reduced-contract Python versions of the same algorithms.

These Python functions omit Lua fuel, debug state, dynamic guards, metatables,
and Lua GC. They are engineering references, not an attainable speed guarantee
or a mathematical lower bound. No closed forms, memoization, or vectorized
libraries replace the measured algorithms.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

from luapyre import LuaRuntime
from vm_programs import WORKLOADS


def typed_arith():
    total = 0
    for i in range(1, 30001):
        total = total + i
    return total


def typed_branch():
    total = 0
    for i in range(1, 30001):
        if i % 2 == 0:
            total = total + i
        else:
            total = total - 1
    return total


def fib_recursive():
    def fib(n):
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)

    return fib(20)


def binary_trees():
    def make(depth):
        if depth == 0:
            return {"left": None, "right": None}
        return {"left": make(depth - 1), "right": make(depth - 1)}

    def check(node):
        if node["left"] is None:
            return 1
        return 1 + check(node["left"]) + check(node["right"])

    total = 0
    for _ in range(30):
        total = total + check(make(8))
    return total


def sieve():
    n = 5000
    composite = {}
    count = 0
    for p in range(2, n + 1):
        if not composite.get(p):
            count = count + 1
            multiple = p * p
            while multiple <= n:
                composite[multiple] = True
                multiple = multiple + p
    return count


def table_mix():
    n = 6000
    values = [None]
    for i in range(1, n + 1):
        values.append((i * 17) % 1009)
    total = 0
    for round_ in range(1, 5):
        for i in range(1, n + 1):
            value = values[i]
            value = (value * 33 + i + round_) % 10007
            values[i] = value
            total = (total + value) % 1000000007
    return total


def string_build():
    value = b""
    for i in range(1, 3001):
        if i % 2 == 0:
            value = value + b"ab"
        else:
            value = value + b"xyz"
    return len(value)


def spectral_norm():
    def weight(i, j):
        return 1.0 / (((i + j) * (i + j + 1) / 2.0) + i + 1.0)

    def multiply_av(x, n):
        out = [None]
        for i in range(1, n + 1):
            total = 0.0
            for j in range(1, n + 1):
                total = total + weight(i - 1, j - 1) * x[j]
            out.append(total)
        return out

    def multiply_atv(x, n):
        out = [None]
        for i in range(1, n + 1):
            total = 0.0
            for j in range(1, n + 1):
                total = total + weight(j - 1, i - 1) * x[j]
            out.append(total)
        return out

    def multiply_at_av(x, n):
        return multiply_atv(multiply_av(x, n), n)

    n = 30
    u = [None]
    for _ in range(n):
        u.append(1.0)
    v = [None]
    for _ in range(5):
        v = multiply_at_av(u, n)
        u = multiply_at_av(v, n)
    vbv = 0.0
    vv = 0.0
    for i in range(1, n + 1):
        vbv = vbv + u[i] * v[i]
        vv = vv + v[i] * v[i]
    return math.sqrt(vbv / vv)


def python_calls_1000():
    def bump(value):
        return value + 1

    for value in range(1000):
        assert bump(value) == value + 1
    return 1000


REFERENCES = {
    fn.__name__: fn for fn in (
        typed_arith, typed_branch, fib_recursive, binary_trees, sieve,
        table_mix, string_build, spectral_norm, python_calls_1000,
    )
}


def prepare(name):
    runtime = LuaRuntime(fuel=20_000_000)
    if name == "python_calls_1000":
        function = runtime.execute_python(
            "-- luapyre: typed\n"
            "return function(x: integer): integer return x + 1 end"
        )

        def run():
            for value in range(1000):
                assert function(value) == value + 1
            return 1000

        validate = lambda result: result == 1000
    else:
        workload = WORKLOADS[name]
        proto = runtime.compile(workload.source_for("luapyre"))

        def run():
            return runtime.vm.run(proto, fuel=20_000_000)

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--python-first", action="store_true")
    parser.add_argument("--case", action="append", choices=REFERENCES)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 1:
        parser.error("warmups and repeats must be positive")
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {affinity})
    results = {}
    for name in args.case or REFERENCES:
        runtime, run, validate = prepare(name)
        variants = [("luapyre", run), ("python_reference", REFERENCES[name])]
        if args.python_first:
            variants.reverse()
        row = {
            label: measure(fn, validate, args.warmups, args.repeats)
            for label, fn in variants
        }
        before = asdict(runtime.jit_stats)
        assert validate(run())
        after = asdict(runtime.jit_stats)
        row["luapyre_warm_counters"] = {
            key: value - before[key] for key, value in after.items()
            if type(value) is int and value != before[key]
        }
        row["ratio"] = row["luapyre"]["median_ms"] / row["python_reference"]["median_ms"]
        results[name] = row
        print(f"{platform.python_version()} {name}: {row['ratio']:.2f}x Python reference", flush=True)
    args.json.write_text(json.dumps({
        "schema_version": 1, "revision": args.revision,
        "python": platform.python_version(), "platform": platform.platform(),
        "affinity_cpu": affinity, "hash_seed": os.environ.get("PYTHONHASHSEED"),
        "warmups": args.warmups, "repeats": args.repeats,
        "python_first": args.python_first,
        "method": "Separate unprofiled samples; Python GC disabled during timing; every result checked",
        "interpretation": "Reduced-contract same-algorithm references; omit Lua runtime semantics; ratios are not guaranteed available speedups",
        "workloads": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
