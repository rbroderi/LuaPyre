from __future__ import annotations

import ast

import pytest

from luapyre import LuaQuotaError, LuaRuntime
from luapyre.ast_backend import JumpListLayout, inline_local_jump_list
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
