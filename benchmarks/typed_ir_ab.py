from __future__ import annotations

import argparse
import gc
import statistics
import time

from luapyre import LuaRuntime


WORKLOADS = {
    "typed_global_read": """-- luapyre: typed
global g: integer
g = 7
local s = 0
for i = 1, 30000 do
    local v: integer = g
    s = s + v
end
return s
""",
    "typed_const_field": """-- luapyre: typed
local t = {value = 7}
local s = 0
for i = 1, 30000 do
    local v: integer = t.value
    s = s + v
end
return s
""",
}


def measure(source: str, *, repeats: int, warmups: int) -> float:
    runtime = LuaRuntime(fuel=20_000_000, jit_threshold=1)
    proto = runtime.compile(source)
    for _ in range(warmups):
        assert runtime.vm.run(proto, fuel=20_000_000) == 210000

    samples: list[float] = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = runtime.vm.run(proto, fuel=20_000_000)
            elapsed = time.perf_counter_ns() - start
            assert result == 210000
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
    for name, source in WORKLOADS.items():
        median = measure(source, repeats=args.repeats, warmups=args.warmups)
        print(f"{args.label}\t{name}\t{median:.6f}")


if __name__ == "__main__":
    main()
