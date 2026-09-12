from __future__ import annotations

import argparse
import gc
import statistics
import time

from luapyre import LuaRuntime


WORKLOADS = {
    "dense_register_loop": (
        "local total = 0; for i = 1, 50000 do total = total + i end; return total",
        1_250_025_000,
    ),
    "typed_leaf_safe_add": (
        """
local function identity(x: integer): integer return x + 0 end
local total = 0
for i = 1, 30000 do total = total + identity(i) end
return total
""",
        450_015_000,
    ),
    "typed_literal_range": (
        """-- luapyre: typed
local total = 0
for i = 1, 30000 do
    local shifted = i + 2
    total = total + shifted
end
return total
""",
        450_075_000,
    ),
}


def measure(source: str, expected: object, repeats: int, warmups: int) -> float:
    runtime = LuaRuntime(fuel=50_000_000, jit_threshold=1)
    proto = runtime.compile(source)
    for _ in range(warmups):
        assert runtime.vm.run(proto, fuel=50_000_000) == expected
    samples: list[float] = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            assert runtime.vm.run(proto, fuel=50_000_000) == expected
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
    finally:
        if enabled:
            gc.enable()
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Same-runner A/B corpus for CPython specialization-friendly codegen"
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=13)
    parser.add_argument("--warmups", type=int, default=4)
    args = parser.parse_args()
    for name, (source, expected) in WORKLOADS.items():
        median = measure(source, expected, args.repeats, args.warmups)
        print(f"{args.label}\t{name}\t{median:.6f}")


if __name__ == "__main__":
    main()
