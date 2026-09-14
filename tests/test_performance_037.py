"""Correctness coverage for the accepted 0.37 performance paths."""
from __future__ import annotations

import importlib.util
import dis
from pathlib import Path
import sys

import pytest

from luapyre import LuaRuntime, LuaRuntimeError
from luapyre.table import LuaTable, _hash_key


_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "speed_037_ab.py"
sys.path.insert(0, str(_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("luapyre_speed_037", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
probes = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probes
_SPEC.loader.exec_module(probes)


@pytest.mark.parametrize("name", probes.NEW_CASES)
def test_new_probe_reference_interpreter_and_warmed_jit(name):
    case = probes.CASES[name]
    _, interpreted, reference = probes.prepare(name, jit=False)
    expected = interpreted()
    assert probes.validate(case, expected)
    assert probes.validate(case, reference())
    _, warmed, _ = probes.prepare(name, threshold=1)
    for _ in range(4):
        assert warmed() == expected


def test_next_index_supports_independent_traversals_and_live_values():
    table = LuaTable.from_sequence((10, 20, 30))
    first = table.next_item()
    assert first == (1, 10)
    assert table.next_item() == first
    table.rawset(2, 99)
    assert table.next_item(1) == (2, 99)
    assert table.next_item(2) == (3, 30)


def test_next_index_preserves_deleted_key_chains_and_reinsertion():
    table = LuaTable()
    for key in (b"a", b"b", b"c", b"d"):
        table.rawset(key, key)
    table.rawset(b"b", None)
    table.rawset(b"c", None)
    assert table.next_item(b"b") == (b"d", b"d")
    assert table.next_item(b"c") == (b"d", b"d")
    table.rawset(b"b", 42)
    assert dict(table.items())[b"b"] == 42
    assert table.next_item(b"b") is None


def test_next_index_rejects_unknown_and_normalizes_numeric_keys():
    table = LuaTable.from_sequence((10, 20))
    assert table.next_item(1.0) == (2, 20)
    with pytest.raises(LuaRuntimeError, match="invalid key"):
        table.next_item(b"unknown")


def test_prehashed_deletion_uses_indexed_successor():
    table = LuaTable()
    for key in (b"a", b"b", b"c"):
        table.rawset_prehashed(key, _hash_key(key), key)
    table.rawset_prehashed(b"a", _hash_key(b"a"), None)
    assert table.successor_after_deleted(b"a") == (True, b"b")


@pytest.mark.parametrize("name,args", [
    ("python_leaf_local", (3,)),
    ("python_leaf_float", (3.0,)),
    ("python_leaf_binary", (3, 4)),
])
def test_scalar_boundary_shapes_use_direct_entry(name, args):
    case = probes.CASES[name]
    runtime = LuaRuntime(jit_threshold=1)
    function = runtime.execute_python("-- luapyre: typed\n" + case.source)
    before = runtime.jit_stats.python_direct_entries
    for _ in range(4):
        assert function(*args) == case.reference(*args)
    assert runtime.jit_stats.python_direct_entries > before


def test_float_and_binary_scalar_entries_keep_validation_and_live_jit_toggle():
    float_fn = LuaRuntime(jit_threshold=1).execute_python("""-- luapyre: typed
return function(value: float): float return value + 0.5 end
""")
    assert float_fn(2.0) == 2.5
    with pytest.raises(LuaRuntimeError, match="expected float"):
        float_fn(2)

    runtime = LuaRuntime(jit_threshold=1)
    binary = runtime.execute_python("""-- luapyre: typed
return function(left: integer, right: integer): integer return left + right end
""")
    assert binary(2, 3) == 5
    runtime.vm.jit.enabled = False
    assert binary(4, 5) == 9
    runtime.vm.jit.enabled = True
    with pytest.raises(LuaRuntimeError, match="expected integer"):
        binary(2, False)


@pytest.mark.parametrize("name", [
    "python_leaf_local", "python_leaf_float", "python_leaf_binary",
])
def test_scalar_entries_have_no_argument_or_result_containers(name):
    case = probes.CASES[name]
    runtime = LuaRuntime(jit_threshold=1)
    function = runtime.execute_python("-- luapyre: typed\n" + case.source)
    compiled = runtime.vm.jit.get_leaf(function.raw.proto)
    assert compiled is not None and compiled.scalar_runner is not None
    opnames = {
        instruction.opname
        for instruction in dis.get_instructions(compiled.scalar_runner)
    }
    assert not {"BUILD_LIST", "BUILD_MAP", "BUILD_TUPLE"} & opnames
    assert "cells" not in compiled.scalar_runner.__code__.co_varnames


def test_warmed_prehashed_deletion_uses_successor_preserving_helper(monkeypatch):
    original = LuaTable.rawset_prehashed
    deletions = 0

    def observed(table, key, token, value):
        nonlocal deletions
        if value is None:
            deletions += 1
        return original(table, key, token, value)

    monkeypatch.setattr(LuaTable, "rawset_prehashed", observed)
    runtime = LuaRuntime(jit_threshold=1)
    remove = runtime.execute_python("""-- luapyre: typed
return function(t: table): table
    t.a = nil
    return t
end
""")
    for _ in range(40):
        assert remove({"a": 1, "b": 2}) == {"b": 2}
    assert deletions > 0
