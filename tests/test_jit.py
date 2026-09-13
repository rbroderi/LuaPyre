from __future__ import annotations

import pytest

from luapyre import LuaQuotaError, LuaRuntime
from luapyre.bytecode import Op


def test_hot_numeric_loop_compiles_and_runs():
    runtime = LuaRuntime(jit_threshold=2, fuel=1_000_000)
    source = """
        local s = 0
        for i = 1, 1000 do
            s = s + i
        end
        return s
    """
    proto = runtime.compile(source)
    assert runtime.vm.run(proto) == 500500
    stats = runtime.jit_stats
    assert stats is not None
    assert stats.loop_compiles >= 1
    assert stats.loop_executions >= 1
    assert stats.loop_iterations >= 1


def test_hot_raw_table_loops_compile_without_changing_semantics():
    runtime = LuaRuntime(jit_threshold=2, fuel=1_000_000)
    source = """
        local t = {}
        for i = 1, 300 do
            t[i] = i * 3
        end
        local s = 0
        for i = 1, 300 do
            s = s + t[i]
        end
        return s
    """
    assert runtime.execute(source) == 135450
    stats = runtime.jit_stats
    assert stats is not None
    assert stats.loop_compiles >= 2


def test_typed_leaf_uses_trusted_specialized_opcode_and_jits():
    runtime = LuaRuntime(jit_threshold=2, fuel=1_000_000)
    source = """
        local function add(a: integer, b: integer): integer
            return a + b
        end
        local s = 0
        for i = 1, 100 do
            s = add(s, i)
        end
        return s
    """
    proto = runtime.compile(source)
    child = proto.children[0]
    assert child.jit_trust_types is True
    assert any(ins.op is Op.ADD_I for ins in child.code)
    assert runtime.vm.run(proto) == 5050
    stats = runtime.jit_stats
    assert stats is not None
    assert stats.leaf_compiles >= 1
    assert stats.leaf_executions >= 1
    assert stats.leaf_frame_elisions >= 1


def test_hot_branch_loop_compiles_as_one_region():
    runtime = LuaRuntime(jit_threshold=2, fuel=2_000_000)
    source = """
        local s = 0
        for i = 1, 3000 do
            if i % 2 == 0 then
                s = s + i
            else
                s = s - 1
            end
        end
        return s
    """
    proto = runtime.compile(source)
    assert any(ins.op is Op.JFORLOOP for ins in proto.code)
    assert runtime.vm.run(proto) == 2250000
    stats = runtime.jit_stats
    assert stats is not None
    assert stats.loop_compiles >= 1
    assert stats.loop_executions >= 1


def test_fully_typed_branch_loop_specializes_integer_arithmetic():
    runtime = LuaRuntime(jit_threshold=2, fuel=2_000_000)
    source = """-- luapyre: typed
local s = 0
for i = 1, 3000 do
    if i % 2 == 0 then
        s = s + i
    else
        s = s - 1
    end
end
return s
"""
    proto = runtime.compile(source)
    ops = [ins.op for ins in proto.code]
    assert proto.jit_fully_typed is True
    assert Op.ADD_I in ops
    assert Op.SUB_I in ops
    assert Op.JFORLOOP in ops
    assert runtime.vm.run(proto) == 2250000


def test_interpreter_only_mode_remains_available():
    runtime = LuaRuntime(jit=False)
    assert runtime.execute("return 20 + 22") == 42
    assert runtime.jit_stats is None


def test_jit_preserves_instruction_fuel_quota():
    source = """
        local s = 0
        for i = 1, 1000 do
            s = s + i
        end
        return s
    """
    with pytest.raises(LuaQuotaError):
        LuaRuntime(jit=False).execute(source, fuel=25)
    with pytest.raises(LuaQuotaError):
        LuaRuntime(jit=True, jit_threshold=1).execute(source, fuel=25)


def test_branch_region_preserves_instruction_fuel_quota():
    source = """
        local s = 0
        for i = 1, 1000 do
            if i % 2 == 0 then
                s = s + i
            else
                s = s - 1
            end
        end
        return s
    """
    with pytest.raises(LuaQuotaError):
        LuaRuntime(jit=False).execute(source, fuel=40)
    with pytest.raises(LuaQuotaError):
        LuaRuntime(jit=True, jit_threshold=1).execute(source, fuel=40)


def test_structurally_supported_loops_are_quickened():
    runtime = LuaRuntime(jit=False)
    eligible = runtime.compile(
        "local s = 0; for i = 1, 20 do s = s + i end; return s"
    )
    assert any(ins.op is Op.JFORLOOP for ins in eligible.code)
    assert runtime.vm.run(eligible) == 210

    branchy = runtime.compile(
        "local s = 0; for i = 1, 20 do "
        "if i % 2 == 0 then s = s + i else s = s - 1 end "
        "end; return s"
    )
    assert any(ins.op is Op.JFORLOOP for ins in branchy.code)
    assert runtime.vm.run(branchy) == 100

    unsupported_call = runtime.compile(
        "local function bump(x) return x + 1 end; "
        "local s = 0; for i = 1, 20 do s = bump(s) end; return s"
    )
    assert any(ins.op is Op.FORLOOP for ins in unsupported_call.code)
    assert not any(ins.op is Op.JFORLOOP for ins in unsupported_call.code)
    assert runtime.vm.run(unsupported_call) == 20
