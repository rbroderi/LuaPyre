from __future__ import annotations

import argparse
import gc
import statistics
import time

from luapyre import LuaRuntime


WORKLOADS = {
    "pure_leaf_cse": (
        """-- luapyre: typed
local function calc(a: integer, b: integer): integer
    local first = a + b
    local same = a + b
    local dead = a * b
    return same
end
local s = 0
for i = 1, 12000 do
    s = s + calc(i, 3)
end
return s
""",
        72042000,
    ),
    "nested_static_call": (
        """-- luapyre: typed
local function outer(a: integer, b: integer): integer
    local function add(x: integer, y: integer): integer
        local first = x + y
        local same = x + y
        local dead = x * y
        return same
    end
    local value = add(a, b)
    return value + 1
end
local s = 0
for i = 1, 8000 do
    s = s + outer(i, 3)
end
return s
""",
        32036000,
    ),
    "constant_fold_leaf": (
        """-- luapyre: typed
local function folded(x: integer): integer
    local a = 9223372036854775807
    local b = 1
    local wrapped = a + b
    local zero = wrapped - wrapped
    return x + zero + 1
end
local s = 0
for i = 1, 12000 do
    s = s + folded(i)
end
return s
""",
        72018000,
    ),
}


def measure(source: str, expected: object, *, repeats: int, warmups: int) -> float:
    runtime = LuaRuntime(fuel=40_000_000, jit_threshold=1)
    proto = runtime.compile(source)
    for _ in range(warmups):
        assert runtime.vm.run(proto, fuel=40_000_000) == expected

    samples: list[float] = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = runtime.vm.run(proto, fuel=40_000_000)
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
    parser.add_argument("--repeats", type=int, default=13)
    parser.add_argument("--warmups", type=int, default=4)
    args = parser.parse_args()
    for name, (source, expected) in WORKLOADS.items():
        median = measure(source, expected, repeats=args.repeats, warmups=args.warmups)
        print(f"{args.label}\t{name}\t{median:.6f}")


if __name__ == "__main__":
    main()
