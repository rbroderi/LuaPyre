from __future__ import annotations

import ast

import pytest

from luapyre import LuaQuotaError, LuaRuntime, LuaRuntimeError, LuaTable
from luapyre.ast_backend import (
    JumpListLayout,
    inline_local_jump_list,
    optimize_semantic_helpers,
)
from luapyre.bytecode import Op


def test_local_jump_list_is_ast_inlined_before_compile():
    tree = ast.parse(
        """
def run(x):
    state = 0
    def jump0():
        x = x + 1
        return 1
    def jump1():
        x = x * 2
        return -1
    jumps = (jump0, jump1)
    while state >= 0:
        state = jumps[state]()
    return x
"""
    )
    tree = inline_local_jump_list(
        tree,
        JumpListLayout("jumps", "state", ("jump0", "jump1")),
    )
    text = ast.unparse(tree)
    assert "jumps[state]()" not in text
    assert "def jump0" not in text
    assert "match state" in text
    ns = {}
    exec(compile(tree, "<jump-list-test>", "exec"), ns)
    assert ns["run"](3) == 8


def test_semantic_ast_fast_paths_preserve_dense_table_behavior():
    tree = ast.parse(
        """
def run(t, key, value):
    t.rawset(key, value)
    result = t.rawget(key)
    return result, t.rawlen(), t.version
"""
    )
    tree = optimize_semantic_helpers(tree)
    text = ast.unparse(tree)
    assert "t.array[key - 1]" in text
    assert "t.version += 1" in text
    assert "len(t.array)" in text

    ns = {"_LuaTable": LuaTable, "_NUM_TYPES": (int, float)}
    exec(compile(tree, "<semantic-table-fastpath-test>", "exec"), ns)
    run = ns["run"]

    table = LuaTable()
    assert run(table, 1, 42) == (42, 1, 1)
    assert run(table, 1, 43) == (43, 1, 2)

    # A gap takes the exact rawset/rawget fallback and therefore lives in the
    # hash part until the dense array catches up.
    assert run(table, 3, 99) == (99, 1, 3)
    assert table.rawget(3) == 99

    # Nil deletion also remains on rawset so trailing-array shrink semantics are
    # not approximated by the fast path.
    assert run(table, 2, 7) == (7, 3, 4)
    assert run(table, 3, None) == (None, 2, 5)


def test_semantic_ast_fast_paths_preserve_lua_equality_and_type_guards():
    tree = ast.parse(
        """
def equal(a, b):
    return _lua_equal(a, b)

def integer_value(value):
    return _type_matches("integer", value)
"""
    )
    tree = optimize_semantic_helpers(tree)
    text = ast.unparse(tree)
    assert "_lua_equal" not in text
    assert "_type_matches" not in text

    ns = {"_LuaTable": LuaTable, "_NUM_TYPES": (int, float)}
    exec(compile(tree, "<semantic-helper-fastpath-test>", "exec"), ns)
    equal = ns["equal"]
    integer_value = ns["integer_value"]

    assert equal(1, 1.0) is True
    assert equal(True, 1) is False
    assert equal(b"x", b"x") is True
    assert equal(None, None) is True
    assert equal(LuaTable(), LuaTable()) is False
    same = LuaTable()
    assert equal(same, same) is True
    assert integer_value(4) is True
    assert integer_value(4.0) is False
    assert integer_value(True) is False


def test_fully_typed_nested_while_stays_in_super_region():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    source = """-- luapyre: typed
local total = 0
for i = 1, 100 do
    local j = 1
    while j <= 20 do
        total = total + i + j
        j = j + 1
    end
end
return total
"""
    proto = runtime.compile(source)
    assert Op.JFORLOOP in [ins.op for ins in proto.code]
    assert runtime.vm.run(proto) == 122000
    assert runtime.jit_stats.loop_compiles >= 1


def test_fully_typed_small_call_is_inlined_inside_hot_loop():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    source = """-- luapyre: typed
local function bump(x: integer): integer
    return x + 1
end
local total = 0
for i = 1, 1000 do
    total = bump(total)
end
return total
"""
    proto = runtime.compile(source)
    assert Op.JFORLOOP in [ins.op for ins in proto.code]
    assert runtime.vm.run(proto) == 1000
    assert runtime.jit_stats.loop_compiles >= 1


def test_fully_typed_recursive_function_uses_direct_ast_calls():
    runtime = LuaRuntime(jit_threshold=1, fuel=3_000_000)
    source = """-- luapyre: typed
local function fib(n: integer): integer
    if n < 2 then
        return n
    end
    return fib(n - 1) + fib(n - 2)
end
return fib(12)
"""
    assert runtime.execute(source) == 144
    assert runtime.vm.jit.function_compiles >= 1
    assert runtime.vm.jit.function_executions >= 1
    assert runtime.jit_stats.compiled_frame_allocations <= 12


def test_compiled_recursive_frame_pool_preserves_stack_limit():
    runtime = LuaRuntime(jit_threshold=1, max_frames=8, fuel=3_000_000)
    source = """-- luapyre: typed
local function fib(n: integer): integer
    if n < 2 then
        return n
    end
    return fib(n - 1) + fib(n - 2)
end
return fib(20)
"""
    with pytest.raises(LuaRuntimeError, match="stack overflow"):
        runtime.execute(source)


def test_fully_typed_nested_numeric_loops_compile():
    runtime = LuaRuntime(jit_threshold=1, fuel=4_000_000)
    source = """-- luapyre: typed
local total = 0
for i = 1, 100 do
    for j = 1, 50 do
        total = total + i * j
    end
end
return total
"""
    assert runtime.execute(source) == 6438750
    assert runtime.jit_stats.loop_compiles >= 1


def test_fully_typed_dense_tables_compile_and_preserve_results():
    runtime = LuaRuntime(jit_threshold=1, fuel=4_000_000)
    source = """-- luapyre: typed
local t = {}
for i = 1, 100 do
    t[i] = i * 2
end
local total = 0
for round = 1, 4 do
    for i = 1, 100 do
        local value: integer = t[i]
        value = value + 1
        t[i] = value
        total = total + value
    end
end
return total
"""
    assert runtime.execute(source) == 41400
    assert runtime.jit_stats.loop_compiles >= 1


def test_ast_super_regions_preserve_quota_fallback():
    source = """-- luapyre: typed
local total = 0
for i = 1, 1000 do
    local j = 1
    while j <= 10 do
        total = total + i + j
        j = j + 1
    end
end
return total
"""
    with pytest.raises(LuaQuotaError):
        LuaRuntime(jit=False).execute(source, fuel=50)
    with pytest.raises(LuaQuotaError):
        LuaRuntime(jit=True, jit_threshold=1).execute(source, fuel=50)
