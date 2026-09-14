"""Measure the 0.37 table and scalar-boundary candidates.

The 0.36 cases remain available unchanged. New cases add table traversal and
deletion scaling; every timed result is checked against a same-algorithm
Python reference. Run this file against both baseline and candidate sources.
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
import time

from luapyre import LuaRuntime
from speed_036_ab import CASES as CASES_036, Case, FUEL, call_batch, validate


_DENSE = tuple(range(1, 2049))
_HASH = {f"key-{index}": index for index in range(1, 2049)}


def next_dense():
    total = 0
    for _ in range(4):
        for value in _DENSE:
            total += value
    return total


def next_hash():
    total = 0
    for _ in range(4):
        for value in _HASH.values():
            total += value
    return total


def delete_dense():
    values = {index: index for index in range(1, 2049)}
    total = 0
    for index in range(1, 2049):
        total += values[index]
        del values[index]
    return total


def delete_hash():
    values = {f"key-{index}": index for index in range(1, 2049)}
    total = 0
    for index in range(1, 2049):
        key = f"key-{index}"
        total += values[key]
        del values[key]
    return total


NEW_CASES = {
    "next_dense": Case("""
local values: table = {}
for i = 1, 2048 do values[i] = i end
return function(): integer
    local total: integer = 0
    for round = 1, 4 do
        for _, value in pairs(values) do total = total + value end
    end
    return total
end
""", next_dense, 8392704, "Repeated full traversal of an unchanged dense table", ((),)),
    "next_hash": Case("""
local values: table = {}
for i = 1, 2048 do values["key-" .. i] = i end
return function(): integer
    local total: integer = 0
    for round = 1, 4 do
        for _, value in pairs(values) do total = total + value end
    end
    return total
end
""", next_hash, 8392704, "Repeated full traversal of an unchanged hash table", ((),)),
    "delete_dense": Case("""
return function(): integer
    local values: table = {}
    for i = 1, 2048 do values[i] = i end
    local total: integer = 0
    for i = 1, 2048 do total = total + values[i]; values[i] = nil end
    return total
end
""", delete_dense, 2098176, "Forward deletion sweep over a dense table", ((),)),
    "delete_hash": Case("""
return function(): integer
    local values: table = {}
    for i = 1, 2048 do values["key-" .. i] = i end
    local total: integer = 0
    for i = 1, 2048 do
        local key = "key-" .. i
        total = total + values[key]
        values[key] = nil
    end
    return total
end
""", delete_hash, 2098176, "Forward deletion sweep over a hash table", ((),)),
}

CASES = {**CASES_036, **NEW_CASES}
UNTYPED_CASES = frozenset(("next_dense", "next_hash"))


def prepare(name, *, jit=True, threshold=32):
    case = CASES[name]
    runtime = LuaRuntime(jit=jit, jit_threshold=threshold, fuel=FUEL)
    source = case.source if name in UNTYPED_CASES else "-- luapyre: typed\n" + case.source
    if case.arguments is None:
        proto = runtime.compile(source)

        def run():
            return runtime.vm.run(proto, fuel=FUEL)
        reference = case.reference
    else:
        function = runtime.execute_python(source)
        expected = tuple(case.reference(*args) for args in case.arguments)
        run = call_batch(function, case.arguments, expected)
        reference = call_batch(case.reference, case.arguments, expected)
    return runtime, run, reference


def counter_delta(before, runtime):
    return {
        key: value - before[key]
        for key, value in asdict(runtime.jit_stats).items()
        if type(value) is int and value != before[key]
    }


def timed_samples(run, case, count, *, disable_gc):
    was_enabled = gc.isenabled()
    if disable_gc:
        gc.disable()
    samples = []
    try:
        for _ in range(count):
            started = time.perf_counter_ns()
            result = run()
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
            assert validate(case, result), (result, case.expected)
    finally:
        if was_enabled:
            gc.enable()
        else:
            gc.disable()
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def measure(name, warmups, repeats, *, python_first=False):
    case = CASES[name]
    started = time.perf_counter_ns()
    runtime, run, reference = prepare(name)
    setup_ms = (time.perf_counter_ns() - started) / 1_000_000
    before = asdict(runtime.jit_stats)
    cold = timed_samples(run, case, 1, disable_gc=False)
    cold_counters = counter_delta(before, runtime)
    before = asdict(runtime.jit_stats)
    warm = timed_samples(run, case, warmups, disable_gc=False)
    warm_counters = counter_delta(before, runtime)
    timed_samples(reference, case, warmups, disable_gc=False)
    variants = [("luapyre", run), ("python_reference", reference)]
    if python_first:
        variants.reverse()
    before = asdict(runtime.jit_stats)
    steady = {
        label: timed_samples(fn, case, repeats, disable_gc=True)
        for label, fn in variants
    }
    steady_counters = counter_delta(before, runtime)
    return {
        "purpose": case.purpose,
        "setup_ms": setup_ms,
        "cold": cold,
        "cold_counters": cold_counters,
        "warmup": warm,
        "warmup_counters": warm_counters,
        "steady": steady,
        "steady_counters": steady_counters,
        "compilation_during_steady": {
            key: value for key, value in steady_counters.items() if "compile" in key
        },
        "ratio": steady["luapyre"]["median_ms"] / steady["python_reference"]["median_ms"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--python-first", action="store_true")
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
        results[name] = measure(
            name, args.warmups, args.repeats, python_first=args.python_first
        )
        print(f"{name}: {results[name]['ratio']:.2f}x Python reference", flush=True)
    args.json.write_text(json.dumps({
        "schema_version": 1,
        "revision": args.revision,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "affinity_cpu": affinity,
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
        "jit_threshold": 32,
        "fuel": FUEL,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "python_first": args.python_first,
        "method": "Setup, first execution, warmup, and steady phases separated; GC disabled only during steady timing; every result checked",
        "interpretation": "Reduced-contract Python references, not promised speedups; boundary cases use the same checked driver on both sides.",
        "workloads": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
