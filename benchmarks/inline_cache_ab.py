from __future__ import annotations

import argparse
import gc
import statistics
import time

from luapyre import LuaRuntime


WORKLOADS = {
    "monomorphic_call": (
        """
local function bump(x) return x + 1 end
local total = 0
for i = 1, 30000 do total = bump(total) end
return total
""",
        30000,
    ),
    "monomorphic_table_read": (
        """
local t = {value = 7}
local total = 0
for i = 1, 30000 do total = total + t.value end
return total
""",
        210000,
    ),
    "versioned_table_read": (
        """
local t = {value = 1}
local total = 0
for i = 1, 30000 do
    total = total + t.value
    if i == 15000 then t.value = 3 end
end
return total
""",
        60000,
    ),
}


def measure(source: str, expected: object, *, repeats: int, warmups: int) -> float:
    runtime = LuaRuntime(fuel=20_000_000)
    proto = runtime.compile(source)
    for _ in range(warmups):
        assert runtime.vm.run(proto, fuel=20_000_000) == expected

    samples: list[float] = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = runtime.vm.run(proto, fuel=20_000_000)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
            assert result == expected
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
