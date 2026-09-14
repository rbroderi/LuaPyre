"""Profile warmed LuaPyre execution; use unprofiled A/B runs to measure speed."""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import asdict
import json
from pathlib import Path
import platform
import pstats

from luapyre import LuaRuntime
from vm_programs import WORKLOADS


DEFAULT_WORKLOADS = (
    "typed_arith", "typed_calls", "typed_branch", "typed_global_read",
    "typed_const_field", "fib_recursive", "sieve", "binary_trees",
    "table_mix", "string_build", "spectral_norm", "coroutines",
    "python_calls_1000",
)


def prepare(name):
    runtime = LuaRuntime(fuel=20_000_000)
    if name == "python_calls_1000":
        function = runtime.execute_python(
            "-- luapyre: typed\nreturn function(x: integer): integer return x + 1 end"
        )

        def run():
            for value in range(1000):
                assert function(value) == value + 1
    else:
        workload = WORKLOADS[name]
        proto = runtime.compile(workload.source_for("luapyre"))

        def run():
            result = runtime.vm.run(proto, fuel=20_000_000)
            assert workload.validate(result), (name, result)

    return runtime, run


def profile(name, warmups, executions=1):
    runtime, run = prepare(name)
    for _ in range(warmups):
        run()
    before = asdict(runtime.jit_stats)
    profiler = cProfile.Profile()
    for _ in range(executions):
        profiler.runcall(run)
    after = asdict(runtime.jit_stats)
    stats = pstats.Stats(profiler)
    records = [
        dict(file=Path(file).name, line=line, function=function,
             calls=calls, primitive_calls=primitive,
             self_seconds=self_time, cumulative_seconds=cumulative)
        for (file, line, function), (primitive, calls, self_time, cumulative, _)
        in stats.stats.items()
    ]
    return dict(
        name=name,
        profiled_executions=executions,
        total_profiled_seconds=stats.total_tt,
        counters={key: value - before[key] for key, value in after.items()
                  if isinstance(value, int) and value != before[key]},
        by_self=sorted(records, key=lambda row: row["self_seconds"], reverse=True)[:10],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Source revision being profiled")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--executions", type=int, default=1)
    parser.add_argument("--workload", action="append", choices=DEFAULT_WORKLOADS)
    args = parser.parse_args()
    if args.warmups < 1:
        parser.error("warmups must be positive")
    if args.executions < 1:
        parser.error("executions must be positive")
    results = [profile(name, args.warmups, args.executions) for name in args.workload or DEFAULT_WORKLOADS]
    args.json.write_text(json.dumps(dict(
        schema_version=1,
        revision=args.revision,
        python=platform.python_version(),
        platform=platform.platform(),
        warmups=args.warmups,
        profiled_executions=args.executions,
        method=f"{args.warmups} warmups, then {args.executions} cProfile execution(s); GC remains enabled",
        interpretation="Call counts and attribution are diagnostic; profiled timings are not speed measurements",
        workloads=results,
    ), indent=2) + "\n")
    for result in results:
        print(result["name"])
        for row in result["by_self"][:5]:
            share = 100 * row["self_seconds"] / result["total_profiled_seconds"]
            print(f"  {row['function']}: {row['calls']} calls, {share:.1f}% self time")


if __name__ == "__main__":
    main()
