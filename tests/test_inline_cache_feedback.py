from __future__ import annotations

from luapyre import LuaRuntime
from luapyre.inline_cache import CacheState, InlineCacheFeedback


def test_monomorphic_call_and_table_sites_report_hits():
    runtime = LuaRuntime(jit_threshold=10_000, fuel=2_000_000)
    result = runtime.execute("""
local t = {value = 1}
local function read(x) return x.value end
local total = 0
for i = 1, 100 do
    total = total + read(t)
end
return total
""")
    assert result == 100
    feedback = runtime.jit_feedback
    assert feedback is not None
    assert feedback["call_sites"] >= 1
    assert feedback["table_sites"] >= 1
    assert runtime.jit_stats.call_ic_hits > 0
    assert runtime.jit_stats.table_ic_hits > 0


def test_table_read_cache_observes_writes_without_version_invalidation():
    runtime = LuaRuntime(jit_threshold=10_000, fuel=2_000_000)
    result = runtime.execute("""
local t = {value = 1}
local total = 0
for i = 1, 20 do
    total = total + t.value
    if i == 10 then t.value = 3 end
end
return total
""")
    assert result == 40
    assert runtime.jit_stats.cache_invalidations == 0
    assert runtime.jit_feedback["invalidations"] == 0


def test_streaming_integer_keys_bypass_the_identity_pic():
    runtime = LuaRuntime(jit_threshold=10_000, fuel=2_000_000)
    result = runtime.execute("""
local t = {}
for i = 1, 100 do t[i] = i end
local total = 0
for i = 1, 100 do total = total + t[i] end
return total
""")
    assert result == 5050
    assert runtime.jit_feedback["table_sites"] == 0
    assert runtime.jit_feedback["states"]["megamorphic"] == 0


def test_call_site_widens_when_function_identity_changes():
    runtime = LuaRuntime(jit_threshold=10_000, fuel=2_000_000)
    result = runtime.execute("""
local function one(x) return x + 1 end
local function two(x) return x + 2 end
local f = one
local total = 0
for i = 1, 20 do
    if i == 11 then f = two end
    total = f(total)
end
return total
""")
    assert result == 30
    assert runtime.jit_feedback["states"]["polymorphic"] >= 1


def test_recreated_closures_with_same_proto_keep_call_site_monomorphic():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    proto = runtime.compile("""
local function bump(x) return x + 1 end
local total = 0
for i = 1, 100 do total = bump(total) end
return total
""")
    for _ in range(8):
        assert runtime.vm.run(proto) == 100
    feedback = runtime.jit_feedback
    assert feedback["states"]["monomorphic"] >= 1
    assert feedback["states"]["megamorphic"] == 0


def test_metatable_table_access_stays_on_exact_slow_path():
    runtime = LuaRuntime(jit_threshold=10_000, fuel=2_000_000)
    result = runtime.execute("""
local fallback = {value = 9}
local t = setmetatable({}, {__index = fallback})
local total = 0
for i = 1, 20 do total = total + t.value end
return total
""")
    assert result == 180


def test_polymorphic_cache_caps_and_becomes_megamorphic():
    feedback = InlineCacheFeedback(polymorphic_limit=2)
    site = feedback.call_site(object(), 3)
    first, second, third = object(), object(), object()
    site.install((first, type(first)))
    site.install((second, type(second)))
    assert site.state is CacheState.POLYMORPHIC
    site.install((third, type(third)))
    assert site.state is CacheState.MEGAMORPHIC
    assert site.entries == []


def test_deopt_feedback_is_reasoned_and_requests_retirement():
    feedback = InlineCacheFeedback(retire_after=3)
    proto = object()
    assert not feedback.record_deopt(proto, 7, "type_guard")
    assert not feedback.record_deopt(proto, 7, "type_guard")
    assert feedback.record_deopt(proto, 7, "type_guard")
    snapshot = feedback.snapshot()
    assert snapshot["deopt_sites"] == 1
    assert snapshot["deopt_reasons"] == {"type_guard": 3}


def test_interpreter_only_runtime_has_no_feedback_surface():
    runtime = LuaRuntime(jit=False)
    assert runtime.execute("return 1") == 1
    assert runtime.jit_feedback is None
