"""Exploratory 0.33 experiments; patches apply only to this benchmark process.

These are workload probes, not accepted runtime optimizations. Run each variant
in a separate process with PYTHONPATH selecting the source under investigation.
The spectral probe deliberately selects named benchmark callees: its result
measures a tier-selection opportunity, not a general implementation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import time

from luapyre import LuaRuntime
from speed_030_ab import median_ms
from vm_programs import WORKLOADS


CASES = {
    "spectral_tiers": ("spectral_norm", "prefer_structured"),
    "table_guards": ("table_mix", "literal_guards"),
    "table_allocation": ("binary_trees", "fresh_table_adoption"),
    "frame_rebind": ("fib_recursive", "one_argument_frame"),
}


def install_probe(runtime, variant):
    if variant == "baseline":
        return
    if variant == "prefer_structured":
        jit = runtime.vm.jit
        original = jit.get_compiled_function

        def prefer_loops(closure):
            if closure.proto.name in ("multiplyAv", "multiplyAtv"):
                # Cache refusal to avoid recompilation while keeping existing
                # interpreter backedge compilation and all its side exits.
                jit._function_cache[id(closure.proto)] = (closure.proto, None)
                return None
            return original(closure)

        jit.get_compiled_function = prefer_loops
    elif variant == "literal_guards":
        from luapyre.ast_jit import AstPythonJIT
        from luapyre.bytecode import Op

        original = AstPythonJIT._emit_instruction

        def emit(self, frame, item, **kwargs):
            lines = original(self, frame, item, **kwargs)
            if lines is not None and item.ins.op is Op.GUARD:
                ins = item.ins
                before = f"_type_matches(consts[{ins.b}],"
                after = f"_type_matches({frame.proto.constants[ins.b]!r},"
                # Feed immutable type names to the existing shared AST guard
                # inliner. This preserves the check, its fuel, and its side exit.
                return [line.replace(before, after) for line in lines]
            return lines

        AstPythonJIT._emit_instruction = emit
    elif variant == "fresh_table_adoption":
        from luapyre.gcvm import GarbageCollectedVM
        from luapyre.table import LuaTable

        def fresh_table(vm, frames=()):
            roots = frames or vm._active_frames or ()
            collector = vm.gc
            collector.safepoint(roots)
            table = LuaTable()
            # This freshly constructed empty table has no outgoing edges.
            table._gc_owner = collector
            collector.stats.allocations += 1
            collector.account_bytes(collector._object_size(table))
            return table

        GarbageCollectedVM._new_table = fresh_table
    elif variant == "one_argument_frame":
        from luapyre.errors import LuaRuntimeError
        from luapyre.optimizing_jitvm import OptimizingJITVM
        from luapyre.values import static_value_type, type_matches

        original = OptimizingJITVM._acquire_compiled_frame

        def acquire(vm, compiled, closure, args, dest, want, *, validate_args=True):
            proto = closure.proto
            if not compiled.frame_pool or proto.param_count != 1 or proto.env_reg >= 0 or not args:
                return original(vm, compiled, closure, args, dest, want, validate_args=validate_args)
            frame = compiled.frame_pool.pop()
            value = args[0]
            if validate_args:
                expected = proto.param_types[0].name
                if not type_matches(expected, value):
                    raise LuaRuntimeError(
                        f"argument 1: expected {expected}, got {static_value_type(value).name}"
                    )
            frame.regs[0] = value
            frame.closure = closure
            frame.pc = 0
            frame.return_reg = dest
            frame.return_want = want
            if vm.debug_hooks_enabled:
                frame.hook_call_values = tuple(args[:1])
            return frame

        OptimizingJITVM._acquire_compiled_frame = acquire
    else:
        raise ValueError(variant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--variant", choices=("baseline", "probe"), required=True)
    parser.add_argument("--warmups", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 1:
        parser.error("warmups and repeats must be positive")
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {affinity})
    name, variant = CASES[args.case]
    if args.variant == "baseline":
        variant = "baseline"
    workload = WORKLOADS[name]
    runtime = LuaRuntime(fuel=20_000_000)
    install_probe(runtime, variant)
    proto = runtime.compile(workload.source_for("luapyre"))

    def run():
        assert workload.validate(runtime.vm.run(proto, fuel=20_000_000))

    start = time.perf_counter_ns()
    run()
    cold_ms = (time.perf_counter_ns() - start) / 1_000_000
    elapsed = median_ms(run, args.repeats, args.warmups)
    before = asdict(runtime.jit_stats)
    run()
    after = asdict(runtime.jit_stats)
    result = dict(
        case=args.case, workload=name, variant=variant,
        python=platform.python_version(), affinity_cpu=affinity,
        hash_seed=os.environ.get("PYTHONHASHSEED"),
        warmups=args.warmups, repeats=args.repeats, cold_run_ms=cold_ms,
        steady_median_ms=elapsed,
        counter_delta={k: v - before[k] for k, v in after.items()
                       if type(v) is int and v != before[k]},
        validation="Expected benchmark result checked on every execution; no conformance claim for the probes",
    )
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['python']} {args.case} {variant}: {elapsed:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
