from __future__ import annotations

import argparse
import statistics
import time

from luapyre import LuaRuntime


BINARY_TREES = '''
local function tree(depth)
    if depth == 0 then return {item = 1} end
    return {left = tree(depth - 1), right = tree(depth - 1)}
end
local count = 0
for i = 1, 8 do
    local root = tree(9)
    if root.left then count = count + 1 end
end
return count
'''

WEAK_CHURN = '''
local weak = setmetatable({}, {__mode = "v"})
do local value = {}; weak[1] = value end
for i = 1, 4000 do local garbage = {i} end
return weak[1] == nil
'''


def measure(source: str, *, running: bool, repeats: int) -> tuple[float, object, object]:
    samples = []
    result = None
    stats = None
    for _ in range(repeats):
        runtime = LuaRuntime(fuel=20_000_000, jit_threshold=2)
        runtime.vm.gc.running = running
        proto = runtime.compile(source)
        start = time.perf_counter_ns()
        result = runtime.vm.run(proto, fuel=20_000_000)
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
        stats = runtime.vm.gc.stats
    return statistics.median(samples), result, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    for name, source in (("binary_trees", BINARY_TREES), ("weak_churn", WEAK_CHURN)):
        for mode, running in (("stopped", False), ("generational", True)):
            median, result, stats = measure(source, running=running, repeats=args.repeats)
            print(
                f"{name}\t{mode}\t{median:.6f} ms\tresult={result!r}"
                f"\tminor={stats.minor_cycles}\tmajor={stats.major_cycles}"
                f"\tpython_young={stats.python_young_steps}"
            )


if __name__ == "__main__":
    main()
