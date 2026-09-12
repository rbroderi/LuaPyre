from __future__ import annotations

import gc
import statistics
import sys
import time

from luapyre import LuaRuntime


WORKLOADS = {
    "arithmetic": (
        """
        local s = 0
        for i = 1, 30000 do
            s = s + i
        end
        return s
        """,
        450015000,
    ),
    "tables": (
        """
        local t = {}
        for i = 1, 8000 do
            t[i] = i * 3
        end
        local s = 0
        for i = 1, 8000 do
            s = s + t[i]
        end
        return s
        """,
        96012000,
    ),
    "calls": (
        """
        local function bump(x)
            return x + 1
        end
        local s = 0
        for i = 1, 8000 do
            s = bump(s)
        end
        return s
        """,
        8000,
    ),
    "branches": (
        """
        local s = 0
        for i = 1, 30000 do
            if i % 2 == 0 then
                s = s + i
            else
                s = s - 1
            end
        end
        return s
        """,
        225000000,
    ),
    "coroutines": (
        """
        local function worker(n)
            local s = 0
            for i = 1, n do
                s = s + i
                if i % 10 == 0 then
                    coroutine.yield(s)
                end
            end
            return s
        end

        local co = coroutine.create(worker)
        local ok, value = coroutine.resume(co, 2000)
        while coroutine.status(co) ~= "dead" do
            ok, value = coroutine.resume(co)
        end
        return value
        """,
        2001000,
    ),
}


def bench(source: str, expected: object, repeats: int = 7) -> tuple[float, float]:
    runtime = LuaRuntime(fuel=5_000_000)
    proto = runtime.compile(source)

    for _ in range(2):
        result = runtime.vm.run(proto, fuel=5_000_000)
        if result != expected:
            raise AssertionError(f"expected {expected!r}, got {result!r}")

    samples = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = runtime.vm.run(proto, fuel=5_000_000)
            elapsed = time.perf_counter_ns() - start
            if result != expected:
                raise AssertionError(f"expected {expected!r}, got {result!r}")
            samples.append(elapsed / 1_000_000)
    finally:
        if was_enabled:
            gc.enable()

    return statistics.median(samples), min(samples)


def main() -> None:
    print(sys.version)
    for name, (source, expected) in WORKLOADS.items():
        median_ms, best_ms = bench(source, expected)
        print(f"{name:12s} median={median_ms:9.3f} ms  best={best_ms:9.3f} ms")


if __name__ == "__main__":
    main()
