from luapyre import LuaRuntime


def test_automatic_pacing_clears_unreachable_weak_values():
    lua = LuaRuntime(jit=False)
    result = lua.execute(
        '''
local weak = setmetatable({}, {__mode = "v"})
do local value = {}; weak[1] = value end
for i = 1, 4000 do local garbage = {i} end
return weak[1] == nil
'''
    )
    assert result is True
    assert lua.vm.gc.stats.automatic_cycles > 0
    assert lua.vm.gc.stats.minor_cycles > 0


def test_stopped_collector_defers_automatic_cycles_until_restart():
    lua = LuaRuntime(jit=False)
    cycles = lua.vm.gc.stats.cycles
    result = lua.execute(
        '''
collectgarbage("stop")
local weak = setmetatable({}, {__mode = "v"})
do local value = {}; weak[1] = value end
for i = 1, 1200 do local garbage = {i} end
local held = weak[1] ~= nil
collectgarbage("restart")
for i = 1, 1200 do local garbage = {i} end
return held, weak[1] == nil
'''
    )
    assert result == (True, True)
    assert lua.vm.gc.stats.cycles > cycles


def test_generational_barrier_keeps_young_value_reachable_from_old_table():
    for jit in (False, True):
        lua = LuaRuntime(jit=jit, jit_threshold=2)
        result = lua.execute(
            '''
local holder = {}
for i = 1, 1200 do local garbage = {i} end
local weak = setmetatable({}, {__mode = "v"})
local child = {answer = 42}
holder.child = child
weak[1] = child
child = nil
for i = 1, 2400 do local garbage = {i} end
return holder.child.answer, weak[1] ~= nil
'''
        )
        assert result == (42, True)
        assert lua.vm.gc.stats.remembered_writes > 0
        assert lua.vm.gc.stats.minor_cycles > 0


def test_generational_barrier_tracks_young_value_in_captured_cell():
    for jit in (False, True):
        lua = LuaRuntime(jit=jit, jit_threshold=1)
        result = lua.execute(
            '''
local slot = nil
local function keep(value) slot = value end
for i = 1, 40 do keep(nil) end
local weak = setmetatable({}, {__mode = "v"})
local child = {answer = 42}
keep(child)
weak[1] = child
child = nil
for i = 1, 2400 do local garbage = {i} end
return slot.answer, weak[1] ~= nil
'''
        )
        assert result == (42, True)
        assert lua.vm.gc.stats.remembered_writes > 0


def test_automatic_pacing_runs_finalizers():
    lua = LuaRuntime(jit=False)
    result = lua.execute(
        '''
local finalized = false
do
  local value = setmetatable({}, {__gc = function() finalized = true end})
end
for i = 1, 4000 do local garbage = {i} end
return finalized
'''
    )
    assert result is True
    assert lua.vm.gc.stats.finalized_objects > 0
