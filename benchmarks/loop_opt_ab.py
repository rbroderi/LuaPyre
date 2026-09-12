from __future__ import annotations

import argparse
import gc
import statistics
import time

from luapyre import LuaRuntime


WORKLOADS = {
    "licm_invariant": (
        """-- luapyre: typed
local function work(n: integer, a: integer, b: integer): integer
    local i: integer = 0
    local total: integer = 0
    while i < n do
        local bias: integer = a + b
        total = total + bias + i
        i = i + 1
    end
    return total
end
local total: integer = 0
for k = 1, 4000 do
    total = total + work(80, 17, 9)
end
return total
""",
        20960000,
    ),
    "induction_while": (
        """-- luapyre: typed
local function sum_to(n: integer): integer
    local i: integer = 0
    local total: integer = 0
    while i < n do
        total = total + i
        i = i + 1
    end
    return total
end
local total: integer = 0
for k = 1, 5000 do
    total = total + sum_to(100)
end
return total
""",
        24750000,
    ),
}


def measure(source: str, expected: object, *, repeats: int, warmups: int) -> float:
    runtime = LuaRuntime(fuel=80_000_000, jit_threshold=1)
    proto = runtime.compile(source)
    for _ in range(warmups):
        assert runtime.vm.run(proto, fuel=80_000_000) == expected

    samples: list[float] = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = runtime.vm.run(proto, fuel=80_000_000)
            elapsed = time.perf_counter_ns() - start
            assert result == expected
            samples.append(elapsed / 1_000_000)
    finally:
        if was_enabled:
            gc.enable()
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=4)
    args = parser.parse_args()
    for name, (source, expected) in WORKLOADS.items():
        median = measure(source, expected, repeats=args.repeats, warmups=args.warmups)
        print(f"{args.label}\t{name}\t{median:.6f}")


if __name__ == "__main__":
    main()
