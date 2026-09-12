from __future__ import annotations

import argparse
import gc
import statistics
import time

from luapyre import LuaRuntime


WORKLOADS = {
    "virtual_branch_frame": (
        """-- luapyre: typed
local function outer(x: integer): integer
    local function choose(y: integer): integer
        if y < 0 then return y - 1 end
        return y + 1
    end
    local result: integer = choose(x)
    return result
end
local total: integer = 0
for i = 1, 8000 do total = total + outer(i - 4000) end
return total
""",
        4002,
    ),
    "virtual_open_results": (
        """-- luapyre: typed
local function outer(x: integer): integer, integer
    local function pair(y: integer): integer, integer
        if y < 0 then return y - 1, y end
        return y + 1, y
    end
    return pair(x)
end
local total: integer = 0
for i = 1, 8000 do
    local a: integer, b: integer = outer(i)
    total = total + a + b
end
return total
""",
        64016000,
    ),
}


def measure(source: str, expected: object, repeats: int, warmups: int) -> tuple[float, object]:
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
    return statistics.median(samples), runtime.jit_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=13)
    parser.add_argument("--warmups", type=int, default=4)
    args = parser.parse_args()
    for name, (source, expected) in WORKLOADS.items():
        median, stats = measure(source, expected, args.repeats, args.warmups)
        elisions = getattr(stats, "virtual_frame_elisions", 0)
        multivalues = getattr(stats, "virtual_multivalue_elisions", 0)
        print(f"{args.label}\t{name}\t{median:.6f}\tframes={elisions}\tmultivalues={multivalues}")


if __name__ == "__main__":
    main()
