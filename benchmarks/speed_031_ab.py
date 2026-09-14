"""Run this same file against each checkout with PYTHONPATH=<checkout>/src."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from luapyre import LuaRuntime
from speed_030_ab import SOURCES, lua_runner, median_ms, python_entry_runner


def scalar_entry_runner(return_type):
    runtime = LuaRuntime()
    function = runtime.execute_python(
        "-- luapyre: typed\nreturn function(x: integer): integer return x + 1 end"
    )

    def run():
        for value in range(1000):
            assert function(value, return_type=return_type) == value + 1

    return run


def array_field_runner():
    runtime = LuaRuntime()
    proto = runtime.compile("""-- luapyre: typed
local point = {7}
local total: integer = 0
for i = 1, 30000 do
    local value: integer = point[1]
    total = total + value
end
return total
""")

    def run():
        assert runtime.vm.run(proto) == 210000

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be positive and warmups nonnegative")
    runners = [(name, lua_runner(name)) for name in SOURCES]
    runners.extend([
        ("constant_array_key", array_field_runner()),
        ("python_entry_1000", python_entry_runner()),
        ("python_entry_inferred_1000", scalar_entry_runner(None)),
        ("python_entry_union_1000", scalar_entry_runner(int | str)),
    ])
    results = {}
    for name, runner in runners:
        results[name] = median_ms(runner, args.repeats, args.warmups)
        print(f"{args.label}\t{name}\t{results[name]:.6f}")
    if args.json:
        args.json.write_text(json.dumps({
            "label": args.label,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "warmups": args.warmups,
            "repeats": args.repeats,
            "median_ms": results,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
