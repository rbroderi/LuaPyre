"""Correctness gates for future optimizations; deliberately no speed thresholds."""
from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import sys

import pytest

from luapyre import LuaRuntime, LuaRuntimeError


_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "speed_036_ab.py"
_SPEC = importlib.util.spec_from_file_location("luapyre_speed_036_probes", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
probes = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probes
_SPEC.loader.exec_module(probes)


@pytest.mark.parametrize("name", probes.CASES)
def test_probe_reference_cold_and_warmed_jit_match_interpreter(name):
    case = probes.CASES[name]
    _, cold, reference = probes.prepare(name, jit=False)
    oracle = cold()
    assert probes.validate(case, oracle)
    assert probes.validate(case, reference())
    _, run, _ = probes.prepare(name, threshold=1)
    for _ in range(5):
        actual = run()
        assert probes.validate(case, actual)
        assert actual == oracle


def test_measurement_reports_phases_samples_and_tier_changes():
    result = probes.measure("python_leaf_local", 2, 3)
    assert len(result["cold"]["samples_ms"]) == 1
    assert len(result["warmup"]["samples_ms"]) == 2
    for variant in result["steady"].values():
        assert len(variant["samples_ms"]) == 3
    assert "compilation_during_steady" in result
    assert result["steady_counters"]["python_direct_entries"] == 3000


def test_measurement_rejects_boolean_integer_confusion():
    case = probes.Case("return 1", lambda: 1, 1, "test")
    assert probes.validate(case, 1)
    assert not probes.validate(case, True)
    assert not probes.validate(case, 1.0)


@pytest.mark.parametrize("enabled", [False, True])
def test_measurement_restores_gc_even_when_validation_fails(enabled):
    initial = gc.isenabled()
    (gc.enable if enabled else gc.disable)()
    try:
        case = probes.CASES["recursive_linear"]
        with pytest.raises(AssertionError):
            probes.timed_samples(lambda: -1, case, 1, disable_gc=True)
        assert gc.isenabled() is enabled
    finally:
        (gc.enable if initial else gc.disable)()


SMALL_REGIONS = (
    """-- luapyre: typed
local function visit(n: integer): integer
    if n == 0 then return 1 end
    return 1 + visit(n-1) + visit(n-1)
end
return visit(2)
""",
    """-- luapyre: typed
local values = {1, 2, 3}
local alias = values
local total: integer = 0
for i = 1, 3 do
    alias[i] = values[i] + 1
    total = total + values[i]
end
return total
""",
    """-- luapyre: typed
local values: table = {}
for p = 2, 3 do
    local k: integer = p*p
    while k <= 8 do values[k] = true; k = k+p end
end
return values[4], values[6], values[8]
""",
)


def outcome(run):
    try:
        return ("result", run())
    except LuaRuntimeError as error:
        return (type(error).__name__, str(error))


@pytest.mark.parametrize("source", SMALL_REGIONS)
def test_future_regions_preserve_every_fuel_budget_through_completion(source):
    hot = LuaRuntime(jit_threshold=1)
    hot_proto = hot.compile(source)
    oracle = LuaRuntime(jit=False)
    cold_proto = oracle.compile(source)
    for _ in range(5):
        hot.vm.run(hot_proto)
    # Find the true interpreter completion threshold; test *all* smaller
    # budgets, the threshold, and two larger budgets (not a guessed cutoff).
    first_complete = next(
        fuel for fuel in range(1000)
        if outcome(lambda: oracle.vm.run(cold_proto, fuel=fuel))[0] == "result"
    )
    for fuel in range(first_complete + 3):
        assert outcome(lambda: hot.vm.run(hot_proto, fuel=fuel)) == outcome(
            lambda: oracle.vm.run(cold_proto, fuel=fuel)
        ), fuel


def test_future_dense_proof_keeps_alias_deletion_and_metatable_fallback():
    source = """-- luapyre: typed
global setmetatable: function
local values = {1, 2, 3, 4}
local alias = values
local total: integer = 0
for i = 1, 4 do
    if i == 3 then
        alias[3] = nil
        setmetatable(values, {__index=function(t: table, k: integer): integer return 20 end})
    end
    local value: integer = values[i]
    total = total + value
end
return total
"""
    for jit in (False, True):
        runtime = LuaRuntime(jit=jit, jit_threshold=1)
        for _ in range(5):
            assert runtime.execute(source) == 27


def test_future_recursive_entry_keeps_lowered_stack_limit():
    runtime = LuaRuntime(jit_threshold=1)
    function = runtime.execute_python("""-- luapyre: typed
local function visit(n: integer): integer
    if n == 0 then return 0 end
    return n + visit(n-1)
end
return visit
""")
    for n in (4, 8, 16, 4, 8, 16):
        assert function(n) == n * (n + 1) // 2
    runtime.vm.max_frames = 2
    with pytest.raises(LuaRuntimeError, match="stack overflow"):
        function(16)


def test_future_recursive_entry_rechecks_a_mutated_capture():
    source = """-- luapyre: typed
local function visit(n: integer): integer
    if n == 0 then return 0 end
    return 1 + visit(n-1)
end
local first: integer = 0
for i = 1, 40 do first = visit(8) end
local saved = visit
visit = function(n: integer): integer return 100+n end
return first, saved(2)
"""
    for jit in (False, True):
        runtime = LuaRuntime(jit=jit, jit_threshold=1)
        for _ in range(3):
            assert runtime.execute(source) == (8, 102)


@pytest.mark.parametrize("name,valid,bad", [
    ("python_leaf_local", (2,), (True,)),
    ("python_leaf_float", (2.0,), (2,)),
    ("python_leaf_binary", (2, 3), (2, False)),
])
def test_future_boundary_specializations_keep_argument_validation(name, valid, bad):
    case = probes.CASES[name]
    runtime = LuaRuntime(jit_threshold=1)
    function = runtime.execute_python("-- luapyre: typed\n" + case.source)
    for _ in range(4):
        assert function(*valid) == case.reference(*valid)
    with pytest.raises(LuaRuntimeError, match="expected"):
        function(*bad)
    with pytest.raises(LuaRuntimeError, match="expected"):
        function()
