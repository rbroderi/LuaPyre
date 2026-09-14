"""Audit generated Python and warmed CPython specialization without speed claims.

Captures the final AST/source passed to compile and profiles checked warm runs
to select generated functions. Generic and adaptive opcode counts describe the
same code object; counts are static, not dynamic instruction frequencies.
"""
from __future__ import annotations

import argparse
import ast
import builtins
from collections import Counter
import cProfile
import dis
import json
from pathlib import Path
import platform
from types import CodeType
from unittest.mock import patch

from python_headroom import REFERENCES, prepare


def inspect_case(name, warmups, executions, top, include_source, *, prepare_case=prepare):
    sources = {}
    original_compile = builtins.compile

    def capture(source, filename, mode, *args, **kwargs):
        result = original_compile(source, filename, mode, *args, **kwargs)
        if str(filename).startswith("<luapyre-") and isinstance(result, CodeType):
            tree = source if isinstance(source, ast.AST) else ast.parse(source)
            functions = {
                node.name: node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }

            def visit(code):
                if code.co_name in functions:
                    sources[id(code)] = functions[code.co_name]
                for value in code.co_consts:
                    if isinstance(value, CodeType):
                        visit(value)

            visit(result)
        return result

    with patch("builtins.compile", capture):
        runtime, run, validate = prepare_case(name)
        for _ in range(warmups):
            assert validate(run())
        profiler = cProfile.Profile()
        for _ in range(executions):
            assert validate(profiler.runcall(run))

    generated = [
        entry for entry in profiler.getstats()
        if isinstance(entry.code, CodeType)
        and entry.code.co_filename.startswith("<luapyre-")
    ]
    rows = []
    for entry in sorted(generated, key=lambda item: item.inlinetime, reverse=True)[:top]:
        code = entry.code
        node = sources.get(id(code))
        stores = Counter(
            item.id for item in ast.walk(node)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
        ) if node is not None else Counter()
        row = {
            "file": code.co_filename,
            "function": code.co_name,
            "profiled_calls": entry.callcount,
            "profiled_executions": executions,
            "bytecode_bytes": len(code.co_code),
            "locals": code.co_nlocals,
            "names": list(code.co_names),
            "generic_opcodes": dict(sorted(Counter(
                item.opname for item in dis.get_instructions(code, adaptive=False)
            ).items())),
            "adaptive_opcodes": dict(sorted(Counter(
                item.opname for item in dis.get_instructions(code, adaptive=True)
            ).items())),
            "ast_stores": dict(sorted(stores.items())),
        }
        if include_source and node is not None:
            row["generated_python"] = ast.unparse(node)
        rows.append(row)

    pools = [
        compiled.frame_pool
        for _proto, compiled in runtime.vm.jit._function_cache.values()
        if compiled is not None
    ]
    return {
        "name": name,
        "hot_generated_functions": rows,
        "cached_function_entries": len(runtime.vm.jit._function_cache),
        "cached_call_entries": len(runtime.vm.jit._call_entry_cache),
        "retained_pooled_frames": sum(len(pool) for pool in pools),
        "retained_register_slots": sum(
            len(frame.regs) for pool in pools for frame in pool
        ),
        "retention_note": "Slot counts only; not transitive heap bytes or a leak test",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=7)
    parser.add_argument("--executions", type=int, default=3)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument(
        "--suite", choices=("headroom", "036", "037"), default="headroom"
    )
    parser.add_argument("--case", action="append")
    parser.add_argument("--include-source", action="store_true")
    args = parser.parse_args()
    if min(args.warmups, args.executions, args.top) < 1:
        parser.error("warmups, executions, and top must be positive")
    cases = REFERENCES
    prepare_case = prepare
    if args.suite == "036":
        from speed_036_ab import CASES, prepare as prepare_probe, validate

        cases = CASES

        def prepare_case(name):
            runtime, run, _reference = prepare_probe(name)
            return runtime, run, lambda value: validate(CASES[name], value)
    elif args.suite == "037":
        from speed_037_ab import CASES, prepare as prepare_probe, validate

        cases = CASES

        def prepare_case(name):
            runtime, run, _reference = prepare_probe(name)
            return runtime, run, lambda value: validate(CASES[name], value)

    if any(name not in cases for name in args.case or ()):
        parser.error(f"case must be one of: {', '.join(cases)}")
    rows = [
        inspect_case(
            name, args.warmups, args.executions, args.top,
            args.include_source, prepare_case=prepare_case,
        )
        for name in args.case or cases
    ]
    args.json.write_text(json.dumps({
        "schema_version": 1,
        "revision": args.revision,
        "suite": args.suite,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "warmups": args.warmups,
        "profiled_executions": args.executions,
        "interpretation": "Diagnostic static code counts and pool slot counts; no elapsed-time or transitive-memory claims",
        "workloads": rows,
    }, indent=2) + "\n")
    for row in rows:
        names = ", ".join(
            item["function"] for item in row["hot_generated_functions"]
        )
        print(f"{row['name']}: {names}")


if __name__ == "__main__":
    main()
