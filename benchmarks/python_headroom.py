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


def n_body():
    pi = 3.141592653589793
    solar_mass = 4.0 * pi * pi
    days_per_year = 365.24
    bodies = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, solar_mass],
        [4.841431442464721, -1.1603200440274284, -0.10362204447112311,
         0.001660076642744037 * days_per_year,
         0.007699011184197404 * days_per_year,
         -0.0000690460016972063 * days_per_year,
         0.0009547919384243266 * solar_mass],
        [8.34336671824458, 4.124798564124305, -0.4035234171143214,
         -0.002767425107268624 * days_per_year,
         0.004998528012349172 * days_per_year,
         0.00002304172975737639 * days_per_year,
         0.0002858859806661308 * solar_mass],
        [12.894369562139131, -15.111151401698631, -0.22330757889265573,
         0.002964601375647616 * days_per_year,
         0.0023784717395948095 * days_per_year,
         -0.000029658956854023756 * days_per_year,
         0.00004366244043351563 * solar_mass],
        [15.379697114850917, -25.919314609987964, 0.17925877295037118,
         0.0026806777249038932 * days_per_year,
         0.001628241700382423 * days_per_year,
         -0.00009515922545197159 * days_per_year,
         0.000051513890204661145 * solar_mass],
    ]
    px = py = pz = 0.0
    for body in bodies:
        px += body[3] * body[6]
        py += body[4] * body[6]
        pz += body[5] * body[6]
    bodies[0][3] = -px / solar_mass
    bodies[0][4] = -py / solar_mass
    bodies[0][5] = -pz / solar_mass
    for _ in range(1000):
        for i in range(len(bodies) - 1):
            bi = bodies[i]
            for j in range(i + 1, len(bodies)):
                bj = bodies[j]
                dx, dy, dz = bi[0] - bj[0], bi[1] - bj[1], bi[2] - bj[2]
                distance2 = dx * dx + dy * dy + dz * dz
                magnitude = 0.01 / (distance2 * math.sqrt(distance2))
                bi[3] -= dx * bj[6] * magnitude
                bi[4] -= dy * bj[6] * magnitude
                bi[5] -= dz * bj[6] * magnitude
                bj[3] += dx * bi[6] * magnitude
                bj[4] += dy * bi[6] * magnitude
                bj[5] += dz * bi[6] * magnitude
        for body in bodies:
            body[0] += 0.01 * body[3]
            body[1] += 0.01 * body[4]
            body[2] += 0.01 * body[5]
    energy = 0.0
    for i, bi in enumerate(bodies):
        energy += 0.5 * bi[6] * (bi[3] * bi[3] + bi[4] * bi[4] + bi[5] * bi[5])
        for bj in bodies[i + 1:]:
            dx, dy, dz = bi[0] - bj[0], bi[1] - bj[1], bi[2] - bj[2]
            energy -= bi[6] * bj[6] / math.sqrt(dx * dx + dy * dy + dz * dz)
    return energy


def mandelbrot():
    size = 40
    checksum = 0
    for y in range(size):
        ci = 2.0 * y / size - 1.0
        byte = bits = 0
        for x in range(size):
            cr = 2.0 * x / size - 1.5
            zr = zi = tr = ti = 0.0
            iterations = 0
            while iterations < 50 and tr + ti <= 4.0:
                zi = 2.0 * zr * zi + ci
                zr = tr - ti + cr
                tr, ti = zr * zr, zi * zi
                iterations += 1
            byte = byte * 2 + int(tr + ti <= 4.0)
            bits += 1
            if bits == 8:
                checksum = (checksum * 131 + byte) % 2147483647
                byte = bits = 0
    return checksum


def fannkuch_redux():
    n = 7
    perm1 = list(range(n))
    count = [0] * (n + 1)
    max_flips, checksum, sign, r = 0, 0, 1, n
    while True:
        while r != 1:
            count[r - 1] = r
            r -= 1
        perm = perm1.copy()
        flips = 0
        k = perm[0]
        while k != 0:
            perm[:k + 1] = reversed(perm[:k + 1])
            flips += 1
            k = perm[0]
        checksum += sign * flips
        max_flips = max(max_flips, flips)
        while True:
            if r == n:
                return checksum * 100 + max_flips
            first = perm1[0]
            perm1[:r] = perm1[1:r + 1]
            perm1[r] = first
            count[r] -= 1
            if count[r] > 0:
                sign = -sign
                break
            r += 1


def fasta():
    alu = b"GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGACCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAATACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCAGCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGGAGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCCAGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA"
    iub = [(b"a", 97, .27), (b"c", 99, .12), (b"g", 103, .12),
           (b"t", 116, .27), (b"B", 66, .02), (b"D", 68, .02),
           (b"H", 72, .02), (b"K", 75, .02), (b"M", 77, .02),
           (b"N", 78, .02), (b"R", 82, .02), (b"S", 83, .02),
           (b"V", 86, .02), (b"W", 87, .02), (b"Y", 89, .02)]
    homo = [(b"a", 97, .3029549426680), (b"c", 99, .1979883004921),
            (b"g", 103, .1975473066391), (b"t", 116, .3015094502008)]

    def cumulative(dist):
        total = 0.0
        out = []
        for char, code, probability in dist:
            total += probability
            out.append((char, code, total))
        return out

    iub, homo = cumulative(iub), cumulative(homo)
    seed = 42

    def pick(dist):
        nonlocal seed
        seed = (seed * 3877 + 29573) % 139968
        value = seed / 139968
        for item in dist:
            if value < item[2]:
                return item
        return dist[-1]

    chunks = []
    checksum = 0
    for i in range(1000):
        char = alu[i % len(alu):i % len(alu) + 1]
        chunks.append(char)
        checksum += char[0]
    for dist, count_ in ((iub, 1500), (homo, 2500)):
        for _ in range(count_):
            char, code, _ = pick(dist)
            chunks.append(char)
            checksum += code
    output = b"".join(chunks)
    return len(output) * 1000000 + checksum


def k_nucleotide():
    motif = b"GGTATTTTAATTTATAGTGGTAAGATATTAAGATAATATTTGGTGGTAGTTTTAATGTGTAA"
    sequence = motif * 20
    checksum = 0
    for k in (1, 2, 3, 4, 6, 12, 18):
        counts = {}
        for i in range(len(sequence) - k + 1):
            key = sequence[i:i + k]
            counts[key] = counts.get(key, 0) + 1
        checksum = (checksum * 131 + counts.get(sequence[:k], 0)) % 2147483647
        checksum = (checksum * 131 + counts.get(sequence[6:6 + k], 0)) % 2147483647
    return checksum


def reverse_complement():
    sequence = b"ACGTUMRWSYKVHDBN" * 250
    complement = dict(zip(b"ACGTUMRWSYKVHDBN", b"TGCAAKYWSRMBDHVN"))
    output = bytearray()
    for i in range(len(sequence) - 1, -1, -1):
        output.append(complement[sequence[i]])
    reversed_ = bytes(output)
    checksum = 0
    for i, byte in enumerate(reversed_, 1):
        checksum = (checksum + i * byte) % 2147483647
    return checksum


def python_calls_1000():
    def bump(value):
        return value + 1

    for value in range(1000):
        assert bump(value) == value + 1
    return 1000


REFERENCES = {
    fn.__name__: fn for fn in (
        typed_arith, typed_branch, fib_recursive, binary_trees, sieve,
        table_mix, string_build, spectral_norm, n_body, mandelbrot,
        fannkuch_redux, fasta, k_nucleotide, reverse_complement,
        python_calls_1000,
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
        row["fast_path_admissions"] = dict(
            runtime.jit_stats.fast_path_admissions
        )
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
