from __future__ import annotations

from luapyre import LuaQuotaError, LuaRuntime
from luapyre.trace_jit import TraceJIT


_TRACE_SOURCE = """-- luapyre: typed
local i: integer = 0
local total: integer = 0
while i < 80 do
    total = total + i
    i = i + 1
end
return total
"""


def test_hot_cfg_edge_compiles_trace_and_enters_with_live_frame_state():
    runtime = LuaRuntime(jit_threshold=3, fuel=2_000_000)
    assert runtime.execute(_TRACE_SOURCE) == 3160
    stats = runtime.jit_stats
    assert stats.trace_compiles >= 1
    assert stats.trace_executions >= 1
    assert stats.osr_entries >= 1
    assert stats.trace_side_exits >= 1
    compiled = [
        trace
        for _proto, trace in runtime.vm.trace_jit.cache.values()
        if trace is not None
    ]
    assert compiled
    assert compiled[0].runner.__code__.co_filename == "<luapyre-cfg-trace>"
    assert compiled[0].plan.loop


def test_trace_side_exit_records_exact_exit_pc():
    runtime = LuaRuntime(jit_threshold=2, fuel=2_000_000)
    proto = runtime.compile(_TRACE_SOURCE)
    assert runtime.vm.run(proto) == 3160
    recorded = {
        pc: count
        for (proto_id, pc), (owner, count) in runtime.vm.trace_jit.side_exits.items()
        if proto_id == id(proto) and owner is proto
    }
    assert recorded
    # The loop trace exits to the first post-loop instruction.
    assert recorded.get(14, 0) >= 1


def test_plain_lua_never_enters_typed_cfg_osr():
    runtime = LuaRuntime(jit_threshold=2, fuel=2_000_000)
    source = _TRACE_SOURCE.replace("-- luapyre: typed\n", "")
    assert runtime.execute(source) == 3160
    assert runtime.jit_stats.trace_compiles == 0
    assert runtime.jit_stats.osr_entries == 0


def test_trace_can_return_directly_from_osr_frame():
    runtime = LuaRuntime(jit_threshold=2, fuel=2_000_000)
    proto = runtime.compile(_TRACE_SOURCE)
    for _ in range(4):
        assert runtime.vm.run(proto) == 3160
    assert runtime.jit_stats.trace_executions >= 1
    assert runtime.jit_stats.osr_entries >= 1
    assert runtime.jit_stats.trace_compiles >= 2


def _fuel_outcome(*, jit: bool, fuel: int):
    runtime = LuaRuntime(jit=jit, jit_threshold=2, fuel=fuel)
    try:
        return "ok", runtime.execute(_TRACE_SOURCE)
    except LuaQuotaError as exc:
        return "quota", str(exc)


def test_trace_osr_preserves_every_nearby_fuel_boundary():
    for fuel in range(1, 420):
        assert _fuel_outcome(jit=True, fuel=fuel) == _fuel_outcome(
            jit=False, fuel=fuel
        )


def test_trace_compiler_negative_caches_unsupported_entry():
    runtime = LuaRuntime(jit=False)
    proto = runtime.compile("""-- luapyre: typed
global t: table
return t.value
""")
    traces = TraceJIT(threshold=1)
    traces.entry_counts[(id(proto), 0)] = (proto, 1)
    assert traces.maybe_trace(proto, 0) is None
    assert traces.cache[(id(proto), 0)] == (proto, None)
