from __future__ import annotations

import dis
import sys

import pytest

from luapyre import LuaRuntime, LuaTable, MultiValue
from luapyre.bytecode import Cell, Closure, Ins, Op, Proto
from luapyre.jit import DEOPT, CompiledLoop
from luapyre.range_analysis import INT_MAX, INT_MIN, analyze_integer_ranges
from luapyre.threadvm import LuaThread
from luapyre.vm import Frame, HostFunction


def _typed_child(expression: str):
    runtime = LuaRuntime(jit_threshold=1)
    proto = runtime.compile(
        "-- luapyre: typed\n"
        "local function f(x: integer): integer\n"
        f"  return {expression}\n"
        "end\n"
        "return f(1)\n"
    ).children[0]
    return runtime, proto


def _runner_frame(proto: Proto, argument: object) -> Frame:
    return Frame(Closure(proto), [argument] + [None] * (proto.register_count - 1))


def test_integer_ranges_elide_only_proven_safe_wraps():
    _, safe = _typed_child("x + 0")
    safe_pc = next(pc for pc, ins in enumerate(safe.code) if ins.op is Op.ADD_I)
    assert analyze_integer_ranges(safe).overflow_free(safe_pc)

    _, unsafe = _typed_child("x + 1")
    unsafe_pc = next(pc for pc, ins in enumerate(unsafe.code) if ins.op is Op.ADD_I)
    assert not analyze_integer_ranges(unsafe).overflow_free(unsafe_pc)


def test_literal_loop_range_is_conservative_about_loop_carried_values():
    proto = LuaRuntime().compile(
        "-- luapyre: typed\n"
        "local total = 0\n"
        "for i = 1, 10 do\n"
        "  local next_i = i + 2\n"
        "  total = total + i\n"
        "end\n"
        "return total\n"
    )
    adds = [pc for pc, ins in enumerate(proto.code) if ins.op is Op.ADD_I]
    plan = analyze_integer_ranges(proto)
    assert plan.overflow_free(adds[0])
    assert not plan.overflow_free(adds[1])


@pytest.mark.skipif(sys.implementation.name != "cpython", reason="CPython bytecode audit")
def test_warmed_typed_leaf_specializes_and_keeps_registers_in_fast_locals():
    runtime, proto = _typed_child("x + 0")
    compiled = runtime.vm.jit._compile_leaf(proto)
    assert compiled is not None
    frame = _runner_frame(proto, 41)
    for _ in range(2_000):
        assert compiled.runner(runtime.vm, frame) == (41,)

    instructions = tuple(dis.get_instructions(compiled.runner, adaptive=True))
    opnames = {ins.opname for ins in instructions}
    assert "BINARY_OP_ADD_INT" in opnames
    assert not any(name.startswith("STORE_SUBSCR") for name in opnames)
    # Constant-index list reads remain only in the one-time register prologue.
    assert sum(name.startswith("BINARY_SUBSCR") for name in opnames) <= 3
    assert not any(ins.argrepr == "&" for ins in instructions)


def test_unproven_integer_overflow_still_wraps_at_signed_64_bits():
    runtime, proto = _typed_child("x + 1")
    compiled = runtime.vm.jit._compile_leaf(proto)
    assert compiled is not None
    assert compiled.runner(runtime.vm, _runner_frame(proto, INT_MAX)) == (INT_MIN,)
    assert any(ins.argrepr == "&" for ins in dis.get_instructions(compiled.runner))


def test_range_proven_typed_loop_omits_hot_int_float_dispatch():
    runtime = LuaRuntime(jit_threshold=1)
    proto = runtime.compile(
        "-- luapyre: typed\n"
        "local total = 0\n"
        "for i = 1, 100 do\n"
        "  local shifted = i + 2\n"
        "  total = total + shifted\n"
        "end\n"
        "return total\n"
    )
    assert runtime.vm.run(proto) == 5_250
    compiled = next(
        value
        for value in runtime.vm._jit_loop_states.values()
        if isinstance(value, CompiledLoop)
    )
    assert compiled.runner.__name__ == "_jit_structured_loop"
    assert "type" not in compiled.runner.__code__.co_names


def test_promoted_leaf_spills_before_sentinel_deopt():
    proto = Proto(
        "manual",
        code=[
            Ins(Op.LOADK, 0, 0),
            Ins(Op.ADD_I, 2, 0, 1),
            Ins(Op.RETURN, 2, 1),
        ],
        constants=[7],
        register_count=3,
    )
    runtime = LuaRuntime(jit_threshold=1)
    compiled = runtime.vm.jit._compile_leaf(proto)
    assert compiled is not None
    frame = _runner_frame(proto, 0)
    frame.regs[1] = b"not an integer"
    assert compiled.runner(runtime.vm, frame) is DEOPT
    assert frame.regs[0] == 7


def test_hot_vm_objects_are_slotted_and_have_no_instance_dict():
    proto = Proto("slots")
    instances = (
        LuaTable(),
        Closure(proto),
        Cell(),
        Frame(Closure(proto), []),
        HostFunction(lambda: None),
        MultiValue(()),
        LuaThread(None),
        proto,
    )
    assert all(not hasattr(value, "__dict__") for value in instances)
