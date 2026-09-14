"""Focused 0.36 admission probes, with cold/warm timing and checked references.

No runtime patches or timing assertions: run this same harness against each
source checkout. Keep speed_035_ab.py as the unchanged release control corpus.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from collections.abc import Callable

from luapyre import LuaRuntime


FUEL = 20_000_000


@dataclass(frozen=True)
class Case:
    source: str
    reference: Callable
    expected: int | float
    purpose: str
    arguments: tuple[tuple, ...] | None = None


def recursive_linear():
    def visit(n):
        if n == 0:
            return 0
        return n + visit(n - 1)
    return visit(80)


def recursive_balanced():
    def visit(n):
        if n == 0:
            return 1
        return 1 + visit(n - 1) + visit(n - 1)
    return visit(9)


def record_continuations():
    def build(n):
        if n == 0:
            return {"count": 1}
        return {"left": build(n - 1), "right": build(n - 1), "count": n}

    def check(t):
        if t.get("left") is None:
            return t["count"]
        return t["count"] + check(t["left"]) + check(t["right"])

    total = 0
    for _ in range(8):
        total = total + check(build(6))
    return total


def dense_read():
    values = [None]
    for i in range(1, 257):
        values.append(i)
    total = 0
    for _ in range(12):
        for i in range(1, 257):
            total = total + values[i]
    return total


def dense_alias_write():
    values = [None]
    for i in range(1, 257):
        values.append(i)
    alias = values
    total = 0
    for _ in range(12):
        for i in range(1, 257):
            value = values[i] + 1
            alias[i] = value
            total = total + values[i]
    return total


def sparse_nested():
    values = {}
    for p in range(2, 41):
        k = p * p
        while k <= 800:
            values[k] = True
            k = k + p
    total = 0
    for k in range(1, 801):
        if values.get(k):
            total = total + 1
    return total


def weight(i, j):
    return 1.0 / ((i + j) * (i + j + 1) / 2 + i + 1)


def numeric_helper():
    total = 0.0
    for i in range(1, 31):
        for j in range(1, 31):
            total = total + weight(i, j)
    return total


def numeric_inline():
    total = 0.0
    for i in range(1, 31):
        for j in range(1, 31):
            total = total + 1.0 / ((i + j) * (i + j + 1) / 2 + i + 1)
    return total


def leaf_local(value):
    adjusted = value + 1
    return adjusted * 2


def leaf_float(value):
    return value + 0.5


def leaf_binary(left, right):
    return left + right


CASES = {
    "recursive_linear": Case("""
local function visit(n: integer): integer
    if n == 0 then return 0 end
    return n + visit(n - 1)
end
return visit(80)
""", recursive_linear, 3240, "Non-tail scalar recursion; depth rather than branching"),
    "recursive_balanced": Case("""
local function visit(n: integer): integer
    if n == 0 then return 1 end
    return 1 + visit(n - 1) + visit(n - 1)
end
return visit(9)
""", recursive_balanced, 1023, "Symmetric recursion; different base-case distribution than Fibonacci"),
    "record_continuations": Case("""
local function build(n: integer): table
    if n == 0 then return {count=1} end
    return {left=build(n-1), right=build(n-1), count=n}
end
local function check(t: table): integer
    if t.left == nil then return t.count end
    return t.count + check(t.left) + check(t.right)
end
local total: integer = 0
for round = 1, 8 do total = total + check(build(6)) end
return total
""", record_continuations, 1472, "Constant record keys crossing recursive call continuations"),
    "dense_read": Case("""
local values: table = {}
for i = 1, 256 do values[i] = i end
local total: integer = 0
for round = 1, 12 do
    for i = 1, 256 do
        local value: integer = values[i]
        total = total + value
    end
end
return total
""", dense_read, 394752, "Read-only dense region; representation proof starting point"),
    "dense_alias_write": Case("""
local values: table = {}
for i = 1, 256 do values[i] = i end
local alias: table = values
local total: integer = 0
for round = 1, 12 do
    for i = 1, 256 do
        local value: integer = values[i] + 1
        alias[i] = value
        total = total + values[i]
    end
end
return total
""", dense_alias_write, 414720, "Aliased writes must invalidate values but need not invalidate storage"),
    "sparse_nested": Case("""
local values: table = {}
for p = 2, 40 do
    local k: integer = p * p
    while k <= 800 do
        values[k] = true
        k = k + p
    end
end
local total: integer = 0
for k = 1, 800 do
    if values[k] then total = total + 1 end
end
return total
""", sparse_nested, 660, "Nested marking regions; no dense-array admission assumption"),
    "numeric_helper": Case("""
local function weight(i: integer, j: integer): float
    return 1.0 / ((i+j)*(i+j+1)/2+i+1)
end
local total: float = 0.0
for i = 1, 30 do
    for j = 1, 30 do total = total + weight(i, j) end
end
return total
""", numeric_helper, 4.119707013245727, "Nested inlined helper; argument/range facts across calls"),
    "numeric_inline": Case("""
local total: float = 0.0
for i = 1, 30 do
    for j = 1, 30 do
        total = total + 1.0 / ((i+j)*(i+j+1)/2+i+1)
    end
end
return total
""", numeric_inline, 4.119707013245727, "Same numerical body without the helper-call boundary"),
    "python_leaf_local": Case("""
return function(value: integer): integer
    local adjusted: integer = value + 1
    return adjusted * 2
end
""", leaf_local, 1001000, "Integer leaf with LOCAL; leftover cells and initializations",
        tuple((i,) for i in range(1000))),
    "python_leaf_float": Case("""
return function(value: float): float return value + 0.5 end
""", leaf_float, 500000.0, "Strict float unary entry control",
        tuple((float(i),) for i in range(1000))),
    "python_leaf_binary": Case("""
return function(left: integer, right: integer): integer return left + right end
""", leaf_binary, 1000000, "Two-integer entry control",
        tuple((i, i + 1) for i in range(1000))),
}


def validate(case, result):
    if type(case.expected) is float:
        return type(result) is float and math.isclose(
            result, case.expected, rel_tol=1e-12, abs_tol=1e-12
        )
    return type(result) is int and result == case.expected


def call_batch(function, arguments, expected):
    def run():
        total = 0
        for args, wanted in zip(arguments, expected):
            value = function(*args)
            assert type(value) is type(wanted) and value == wanted
            total = total + value
        return total
    return run


def prepare(name, *, jit=True, threshold=32):
    case = CASES[name]
    runtime = LuaRuntime(jit=jit, jit_threshold=threshold, fuel=FUEL)
    source = "-- luapyre: typed\n" + case.source
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
        "purpose": case.purpose, "setup_ms": setup_ms,
        "cold": cold, "cold_counters": cold_counters,
        "warmup": warm, "warmup_counters": warm_counters,
        "steady": steady, "steady_counters": steady_counters,
        "compilation_during_steady": {
            key: value for key, value in steady_counters.items()
            if "compile" in key
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
        results[name] = measure(name, args.warmups, args.repeats, python_first=args.python_first)
        print(f"{name}: {results[name]['ratio']:.2f}x Python reference", flush=True)
    args.json.write_text(json.dumps({
        "schema_version": 1, "revision": args.revision,
        "python": platform.python_version(), "platform": platform.platform(),
        "affinity_cpu": affinity, "hash_seed": os.environ.get("PYTHONHASHSEED"),
        "jit_threshold": 32, "fuel": FUEL,
        "warmups": args.warmups, "repeats": args.repeats,
        "python_first": args.python_first,
        "method": "Setup, first execution, warmup, and steady phases separated; GC disabled only during steady timing; every result checked",
        "interpretation": "Reduced-contract Python references, not promised speedups; bound-call cases include the same per-call checking driver on both sides. Setup includes runtime/source preparation, not isolated JIT compilation latency.",
        "workloads": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
