"""Generality, semantics, and policy coverage for the corrected 0.40 tranche."""
from __future__ import annotations

import inspect

import pytest

from luapyre import LuaQuotaError, LuaRuntime, LuaRuntimeError, LuaTable
from luapyre.function_jit import (
    TypedFunctionJITMixin,
    _FUNC_RETURN,
    _fresh_nil_record_prefix,
    _pure_return_prefix,
    _pure_table_nil_return_prefix,
)


TREE_FUNCTIONS = """-- luapyre: typed
local function make(depth: integer): table
    if depth == 0 then return {left=nil, right=nil} end
    return {left=make(depth-1), right=make(depth-1)}
end
local function check(node: table): integer
    if node.left == nil then return 1 end
    return 1 + check(node.left) + check(node.right)
end
return make, check
"""

MATRIX_FUNCTION = """-- luapyre: typed
local function weight(i: integer, j: integer): float
    return 1.0 / (((i+j)*(i+j+1)/2.0)+i+1.0)
end
local function transform(x: table, n: integer): table
    local out: table = {}
    for i = 1, n do
        local total: float = 0.0
        for j = 1, n do
            total = total + weight(i-1, j-1) * x[j]
        end
        out[i] = total
    end
    return out
end
return transform
"""


def _outcome(runtime, proto, fuel):
    try:
        return "return", runtime.vm.run(proto, fuel=fuel)
    except LuaQuotaError:
        return "quota", None


def test_whole_algorithm_replacement_compilers_are_permanently_absent():
    source = inspect.getsource(TypedFunctionJITMixin)
    forbidden = (
        "expected_ops",
        "_compile_dense_matrix_product",
        "_compile_fresh_binary_record_builder",
        "_compile_binary_record_counter",
        "luapyre-dense-matrix-function",
        "luapyre-fresh-binary-record-function",
        "luapyre-binary-record-counter-function",
    )
    assert all(token not in source for token in forbidden)


def test_tree_and_matrix_use_generic_typed_cfg_compiler():
    runtime = LuaRuntime(jit_threshold=1)
    make, check = runtime.execute(TREE_FUNCTIONS)
    transform = runtime.execute_python(MATRIX_FUNCTION)
    for closure in (make, check, transform.raw):
        compiled = runtime.vm.jit.get_compiled_function(closure)
        assert compiled is not None
        assert compiled.runner.__code__.co_filename == "<luapyre-ast-function>"
    assert transform([1.0, 1.0], 2) == pytest.approx(
        [1.5, 0.5333333333333333]
    )


@pytest.mark.parametrize(
    "body,reversed_operands",
    [
        ("if n == 0 then return 1 end return n", False),
        ("local z: integer=0; if n == z then local r: integer=1; return r end return n", False),
        ("if 0 == n then return 1 end return n", True),
    ],
)
def test_scalar_base_analysis_survives_semantic_mutations(body, reversed_operands):
    runtime = LuaRuntime(jit_threshold=1)
    closure = runtime.execute(
        "-- luapyre: typed\nlocal function f(n: integer): integer "
        + body
        + " end return f"
    )
    prefix = _pure_return_prefix(closure.proto)
    assert prefix is not None and prefix.reversed_operands is reversed_operands


@pytest.mark.parametrize(
    "function,argument,expected",
    [
        ("local function f(n: integer): integer if n <= 0 then return 0 end return n+f(n-1) end", 8, 36),
        ("local function f(n: integer): integer if n < 2 then return n end return f(n-1)+f(n-2) end", 8, 21),
        ("local function f(n: integer): integer if 0 >= n then return 1 end return f(n-1)+f(n-1)+f(n-1) end", 4, 81),
    ],
)
def test_generic_recursive_scalar_entry_admits_three_call_graphs(function, argument, expected):
    runtime = LuaRuntime(jit_threshold=1)
    closure = runtime.execute("-- luapyre: typed\n" + function + " return f")
    compiled = runtime.vm.jit.get_compiled_function(closure)
    entry = runtime.vm.jit.get_call_entry(compiled, arg_count=1, trusted_args=True)
    assert entry.scalar_runner is not None
    assert runtime.vm.call_sync(closure, (argument,)) == (expected,)
    assert runtime.jit_stats.fast_path_admissions["scalar_call_entry"] >= 1
    assert runtime.jit_stats.fast_path_admissions["pure_scalar_base"] >= 1


@pytest.mark.parametrize("fields", ["a=nil", "a=nil,b=nil,c=nil", "z=nil,a=nil,q=nil,b=nil"])
def test_fresh_record_layout_is_independent_of_field_count_and_order(fields):
    runtime = LuaRuntime(jit_threshold=1)
    closure = runtime.execute(
        "-- luapyre: typed\nlocal function f(n: integer): table "
        f"if 0 == n then return {{{fields}}} end "
        "return {left=f(n-1),right=f(n-1)} end return f"
    )
    prefix = _fresh_nil_record_prefix(closure.proto)
    assert prefix is not None
    assert prefix.version == fields.count("=")
    compiled = runtime.vm.jit.get_compiled_function(closure)
    assert runtime.jit_stats.fast_path_admissions["fresh_record_layout"] >= 1
    entry = runtime.vm.jit.get_call_entry(compiled, arg_count=1, trusted_args=True)
    status, first = entry.scalar_runner(runtime.vm, [], closure, 0, 0, 1, 100, [0])
    status2, second = entry.scalar_runner(runtime.vm, [], closure, 0, 0, 1, 100, [0])
    assert status == status2 == _FUNC_RETURN
    assert isinstance(first, LuaTable) and first is not second and first.hash == {}


@pytest.mark.parametrize(
    "body",
    [
        "if node.left == nil then return 1 end return 2",
        "local absent=nil; if absent == node.left then local one: integer=1; return one end return 2",
    ],
)
def test_table_base_analysis_survives_temporaries_and_reversed_equality(body):
    runtime = LuaRuntime(jit_threshold=1)
    closure = runtime.execute(
        "-- luapyre: typed\nlocal function f(node: table): integer "
        + body
        + " end return f"
    )
    assert _pure_table_nil_return_prefix(closure.proto) is not None


@pytest.mark.parametrize(
    "program,expected",
    [
        ("local function f(x: integer): integer return (x+2)*(x-1) end", 3040),
        ("local function f(x: integer): integer local y: integer=x*x; return y+3*x+1 end", 3520),
        ("local function f(x: integer): integer local a: integer=x+1; local b: integer=x-1; return a*b+x end", 3060),
    ],
)
def test_pure_scalar_expression_dags_admit_three_different_shapes(program, expected):
    source = "-- luapyre: typed\n" + program + (
        " local total: integer=0 for i=1,20 do total=total+f(i) end return total"
    )
    runtime = LuaRuntime(jit_threshold=1)
    assert runtime.execute(source) == expected
    assert runtime.execute(source) == expected
    assert runtime.jit_stats.fast_path_admissions["pure_scalar_expression_dag"] >= 1


def test_equivalent_expression_tree_and_temporary_keep_dag_admission():
    bodies = (
        "return (x+1)+(x*2)",
        "local doubled: integer=x*2; local incremented: integer=x+1; return doubled+incremented",
    )
    results = []
    for body in bodies:
        runtime = LuaRuntime(jit_threshold=1)
        source = (
            "-- luapyre: typed\nlocal function f(x: integer): integer "
            + body
            + " end local total: integer=0 for i=1,20 do total=total+f(i) end return total"
        )
        results.append(runtime.execute(source))
        runtime.execute(source)
        assert runtime.jit_stats.fast_path_admissions["pure_scalar_expression_dag"] >= 1
    assert results[0] == results[1]


@pytest.mark.parametrize(
    "program,expected",
    [
        ("local s: integer=0 for i=1,5 do for j=1,4 do s=s+i*j end end return s", 150),
        ("local s: integer=1 for i=1,4 do local row: integer=0 for j=1,3 do row=row+i+j end s=s+row end return s", 55),
        ("local s: float=0.0 for i=1,4 do for j=1,i do s=s+1.0/(i+j) end end return s", 2.3345238095238092),
    ],
)
def test_nested_reduction_ir_admits_three_different_shapes(program, expected):
    runtime = LuaRuntime(jit_threshold=1)
    source = "-- luapyre: typed\n" + program
    assert runtime.execute(source) == pytest.approx(expected)
    assert runtime.execute(source) == pytest.approx(expected)
    assert runtime.jit_stats.fast_path_admissions["nested_numeric_region"] >= 1


def test_generic_paths_preserve_every_fuel_boundary():
    source = "-- luapyre: typed\n" + TREE_FUNCTIONS.split("\n", 1)[1] + "\nreturn check(make(2))"
    source = source.replace("return make, check\n", "")
    hot = LuaRuntime(jit_threshold=1)
    cold = LuaRuntime(jit=False)
    hot_proto, cold_proto = hot.compile(source), cold.compile(source)
    for _ in range(3):
        assert hot.vm.run(hot_proto, fuel=10_000) == 7
    first_complete = next(f for f in range(1_000) if _outcome(cold, cold_proto, f)[0] == "return")
    for fuel in range(first_complete + 3):
        assert _outcome(hot, hot_proto, fuel) == _outcome(cold, cold_proto, fuel)


def test_generic_matrix_path_preserves_invalid_argument_errors():
    runtime = LuaRuntime(jit_threshold=1)
    function = runtime.execute_python(MATRIX_FUNCTION)
    with pytest.raises(LuaRuntimeError):
        function([1.0, b"not-a-number"], 2)


def test_empty_table_size_cache_keeps_exact_accounting():
    runtime = LuaRuntime()
    before = runtime.vm.gc.stats.allocated_bytes
    table = runtime.vm._new_table()
    assert runtime.vm.gc.stats.allocated_bytes - before == runtime.vm.gc._object_size(table)
