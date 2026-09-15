"""Correctness gates for the portable language-benchmark corpus."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

from luapyre import LuaRuntime


BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from python_headroom import REFERENCES  # noqa: E402
from vm_programs import LANGUAGE_BENCHMARKS, WORKLOADS  # noqa: E402


ADDED_CASES = (
    "n_body",
    "mandelbrot",
    "fannkuch_redux",
    "fasta",
    "k_nucleotide",
    "reverse_complement",
)


def test_language_benchmark_inventory_is_explicit_and_portable():
    assert LANGUAGE_BENCHMARKS == (
        "n_body",
        "mandelbrot",
        "spectral_norm",
        "fannkuch_redux",
        "binary_trees",
        "fasta",
        "k_nucleotide",
        "reverse_complement",
    )
    assert "pidigits" not in WORKLOADS
    for name in LANGUAGE_BENCHMARKS:
        workload = WORKLOADS[name]
        assert workload.group == "language"
        assert "io." not in workload.source


@pytest.mark.parametrize("name", LANGUAGE_BENCHMARKS)
def test_python_lower_bound_matches_checked_result(name):
    assert WORKLOADS[name].validate(REFERENCES[name]())


@pytest.mark.parametrize("name", ADDED_CASES)
def test_added_lua_source_matches_checked_result_in_luapyre(name):
    workload = WORKLOADS[name]
    runtime = LuaRuntime(fuel=20_000_000)
    result = runtime.execute(workload.source_for("luapyre"))
    assert workload.validate(result)


@pytest.mark.parametrize("name", ADDED_CASES)
def test_added_lua_source_matches_checked_result_in_native_lua55(name):
    lua55 = pytest.importorskip("lupa.lua55")
    runtime = lua55.LuaRuntime(unpack_returned_tuples=True)
    assert runtime.eval("_VERSION") == "Lua 5.5"
    workload = WORKLOADS[name]
    result = runtime.execute("return function()\n" + workload.source + "\nend")()
    assert workload.validate(result)
