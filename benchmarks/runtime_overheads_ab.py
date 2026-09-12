from __future__ import annotations

import argparse
import gc
import statistics
import time

from luapyre import LuaRuntime


TRACE_SOURCE = """-- luapyre: typed
local i: integer = 0
local total: integer = 0
while i < 30000 do
    total = total + i
    i = i + 1
end
return total
"""
FIB_SOURCE = """-- luapyre: typed
local function fib(n: integer): integer
    if n < 2 then
        return n
    end
    return fib(n - 1) + fib(n - 2)
end
return fib(18)
"""


def median_ms(fn, *, repeats: int, warmups: int) -> float:
    for _ in range(warmups):
        fn()
    samples: list[float] = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            fn()
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
    finally:
        if was_enabled:
            gc.enable()
    return statistics.median(samples)


def trace_runner():
    runtime = LuaRuntime(fuel=10_000_000)
    proto = runtime.compile(TRACE_SOURCE)
    assert runtime.vm.run(proto) == 449985000
    return lambda: runtime.vm.run(proto)


def recursive_runner():
    runtime = LuaRuntime(jit_threshold=1, fuel=10_000_000)
    proto = runtime.compile(FIB_SOURCE)
    assert runtime.vm.run(proto) == 2584
    return lambda: runtime.vm.run(proto)


def source_runner():
    runtime = LuaRuntime()

    def run():
        for _ in range(500):
            assert runtime.execute("return 42") == 42

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    workloads = (
        ("trace_codegen", trace_runner()),
        ("recursive_frames", recursive_runner()),
        ("source_cache", source_runner()),
    )
    for name, runner in workloads:
        elapsed = median_ms(runner, repeats=args.repeats, warmups=args.warmups)
        print(f"{args.label}\t{name}\t{elapsed:.6f}")


if __name__ == "__main__":
    main()
