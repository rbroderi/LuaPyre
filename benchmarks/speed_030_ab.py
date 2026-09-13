from __future__ import annotations

import argparse
import gc
import statistics
import time

from luapyre import LuaRuntime


SOURCES = {
    "typed_guard": """-- luapyre: typed
local function bump(x: integer): integer return x + 1 end
local total: integer = 0
for i = 1, 8000 do total = bump(total) end
return total
""",
    "diamond_cfg": """-- luapyre: typed
local total: integer = 0
for i = 1, 30000 do
    if i % 2 == 0 then total = total + i else total = total - 1 end
end
return total
""",
    "constant_key": """-- luapyre: typed
local point = {value = 7}
local total: integer = 0
for i = 1, 30000 do
    local value: integer = point.value
    total = total + value
end
return total
""",
    "recursive_frames": """-- luapyre: typed
local function fib(n: integer): integer
    if n < 2 then return n end
    return fib(n - 1) + fib(n - 2)
end
return fib(18)
""",
}
EXPECTED = {
    "typed_guard": 8000,
    "diamond_cfg": 225000000,
    "constant_key": 210000,
    "recursive_frames": 2584,
}


def median_ms(fn, repeats: int, warmups: int) -> float:
    for _ in range(warmups):
        fn()
    samples = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            fn()
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
    finally:
        if enabled:
            gc.enable()
    return statistics.median(samples)


def lua_runner(name: str):
    runtime = LuaRuntime(fuel=20_000_000)
    proto = runtime.compile(SOURCES[name])
    assert runtime.vm.run(proto, fuel=20_000_000) == EXPECTED[name]
    return lambda: runtime.vm.run(proto, fuel=20_000_000)


def python_entry_runner():
    runtime = LuaRuntime()
    function = runtime.execute_python(
        "-- luapyre: typed\nreturn function(x: integer): integer return x + 1 end"
    )

    def run():
        for value in range(1000):
            assert function(value, return_type=int) == value + 1

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    runners = [(name, lua_runner(name)) for name in SOURCES]
    runners.append(("python_entry_1000", python_entry_runner()))
    for name, runner in runners:
        print(
            f"{args.label}\t{name}\t"
            f"{median_ms(runner, args.repeats, args.warmups):.6f}"
        )


if __name__ == "__main__":
    main()
