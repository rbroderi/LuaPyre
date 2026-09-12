from __future__ import annotations

import argparse
import gc
import statistics
import time

from luapyre import LuaRuntime


SOURCE = """-- luapyre: typed
local i: integer = 0
local total: integer = 0
while i < 30000 do
    total = total + i
    i = i + 1
end
return total
"""
EXPECTED = 449985000


def measure(*, repeats: int, warmups: int) -> float:
    samples: list[float] = []
    for _ in range(warmups):
        runtime = LuaRuntime(fuel=10_000_000)
        assert runtime.execute(SOURCE) == EXPECTED
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            runtime = LuaRuntime(fuel=10_000_000)
            proto = runtime.compile(SOURCE)
            start = time.perf_counter_ns()
            result = runtime.vm.run(proto)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
            assert result == EXPECTED
    finally:
        if was_enabled:
            gc.enable()
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=13)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    print(f"{args.label}\ttrace_osr_while\t{measure(repeats=args.repeats, warmups=args.warmups):.6f}")


if __name__ == "__main__":
    main()
