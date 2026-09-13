from __future__ import annotations

from luapyre import LuaRuntime


def test_hot_coroutine_runs_compiled_between_yields():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    result = runtime.execute("""
local function worker(n)
    local total = 0
    for i = 1, n do
        total = total + i
        if i % 5 == 0 then coroutine.yield(total) end
    end
    return total
end
local co = coroutine.create(worker)
local ok, value = coroutine.resume(co, 100)
while coroutine.status(co) ~= "dead" do
    ok, value = coroutine.resume(co)
end
return ok, value
""")
    assert result == (True, 5050)
    stats = runtime.jit_stats
    assert stats.coroutine_compiles == 1
    assert stats.coroutine_executions > 0
    assert stats.coroutine_yields == 20
    assert stats.coroutine_instructions > 0


def test_compiled_coroutine_resume_values_land_in_call_results():
    runtime = LuaRuntime(jit_threshold=1)
    result = runtime.execute("""
local co = coroutine.create(function (value)
    local left, right = coroutine.yield(value + 1, value + 2)
    return left * 10 + right
end)
local ok1, a, b = coroutine.resume(co, 3)
local ok2, result = coroutine.resume(co, 7, 8)
return ok1, a, b, ok2, result
""")
    assert result == (True, 4, 5, True, 78)
    assert runtime.jit_stats.coroutine_yields == 1


def test_coroutine_yield_identity_guard_deoptimizes_before_mutation():
    runtime = LuaRuntime(jit_threshold=1)
    result = runtime.execute("""
coroutine.yield = function (value) return value + 10 end
local co = coroutine.create(function ()
    local value = coroutine.yield(5)
    return value * 2
end)
return coroutine.resume(co)
""")
    assert result == (True, 30)
    assert runtime.jit_stats.coroutine_deopts >= 1


def test_debug_hook_keeps_coroutine_on_exact_interpreter_path():
    runtime = LuaRuntime(jit_threshold=1, debug_hooks=True)
    result = runtime.execute("""
local co = coroutine.create(function ()
    coroutine.yield(1)
    return 2
end)
debug.sethook(co, function () end, "l")
local ok1, first = coroutine.resume(co)
local ok2, second = coroutine.resume(co)
return ok1, first, ok2, second
""")
    assert result == (True, 1, True, 2)
    assert runtime.jit_stats.coroutine_compiles == 0
