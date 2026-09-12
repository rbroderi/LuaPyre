from __future__ import annotations

import pytest

from luapyre import LuaRuntime


def test_weak_values_are_cleared_after_full_collection():
    lua = LuaRuntime()
    result = lua.execute(
        '''
local weak = setmetatable({}, {__mode = "v"})
local function seed()
  local value = {n = 42}
  weak[1] = value
end
seed()
collectgarbage("collect")
return weak[1] == nil
'''
    )
    assert result is True


def test_weak_keys_are_cleared_but_strings_are_not():
    lua = LuaRuntime()
    result = lua.execute(
        '''
local weak = setmetatable({}, {__mode = "k"})
local function seed()
  local key = {}
  weak[key] = "gone"
  weak["string-key"] = "kept"
end
seed()
collectgarbage()
local count = 0
local string_value = weak["string-key"]
for k, v in pairs(weak) do count = count + 1 end
return count, string_value
'''
    )
    assert result == (1, b"kept")


def test_all_weak_table_removes_pair_when_value_dies():
    lua = LuaRuntime()
    result = lua.execute(
        '''
local weak = setmetatable({}, {__mode = "kv"})
local keeper
local function seed()
  keeper = {}
  local value = {}
  weak[keeper] = value
end
seed()
collectgarbage()
return weak[keeper] == nil
'''
    )
    assert result is True


def test_ephemeron_value_does_not_keep_its_key_alive():
    lua = LuaRuntime()
    result = lua.execute(
        '''
local eph = setmetatable({}, {__mode = "k"})
local function seed()
  local key = {}
  local value = {key = key}
  eph[key] = value
end
seed()
collectgarbage()
return next(eph) == nil
'''
    )
    assert result is True


def test_ephemeron_value_survives_when_key_is_strongly_reachable():
    lua = LuaRuntime()
    result = lua.execute(
        '''
local eph = setmetatable({}, {__mode = "k"})
local key = {}
local function seed()
  local value = {n = 42}
  eph[key] = value
end
seed()
collectgarbage()
return eph[key].n
'''
    )
    assert result == 42


def test_finalizers_run_in_reverse_marking_order():
    lua = LuaRuntime()
    result = lua.execute(
        '''
local log = ""
local mt = {__gc = function(self) log = log .. self.name end}
local function seed()
  local a = setmetatable({name = "a"}, mt)
  local b = setmetatable({name = "b"}, mt)
end
seed()
collectgarbage()
return log
'''
    )
    assert result == b"ba"


def test_adding_gc_after_setmetatable_does_not_mark_object():
    lua = LuaRuntime()
    result = lua.execute(
        '''
local hits = 0
local mt = {}
local function seed()
  local value = setmetatable({}, mt)
end
seed()
mt.__gc = function(self) hits = hits + 1 end
collectgarbage()
return hits
'''
    )
    assert result == 0


def test_finalizer_can_resurrect_table_but_runs_only_once():
    lua = LuaRuntime()
    result = lua.execute(
        '''
local hits = 0
local rescued
local mt = {__gc = function(self)
  hits = hits + 1
  rescued = self
end}
local function seed()
  local value = setmetatable({n = 42}, mt)
end
seed()
collectgarbage()
local first = rescued.n
rescued = nil
collectgarbage()
return hits, first
'''
    )
    assert result == (1, 42)


def test_resurrected_table_can_be_marked_for_finalization_again():
    lua = LuaRuntime()
    result = lua.execute(
        '''
local hits = 0
local rescued
local mt
mt = {__gc = function(self)
  hits = hits + 1
  if hits == 1 then
    rescued = self
    setmetatable(self, mt)
  end
end}
local function seed()
  local value = setmetatable({}, mt)
end
seed()
collectgarbage()
rescued = nil
collectgarbage()
return hits
'''
    )
    assert result == 2


def test_weak_values_drop_finalized_objects_before_gc_callback():
    lua = LuaRuntime()
    result = lua.execute(
        '''
local weak = setmetatable({}, {__mode = "v"})
local seen
local mt = {__gc = function(self) seen = weak[1] end}
local function seed()
  local value = setmetatable({}, mt)
  weak[1] = value
end
seed()
collectgarbage()
return seen == nil
'''
    )
    assert result is True


def test_weak_key_for_finalized_object_survives_one_extra_cycle():
    lua = LuaRuntime()
    result = lua.execute(
        '''
local weak = setmetatable({}, {__mode = "k"})
local seen_during_gc = false
local mt = {__gc = function(self)
  seen_during_gc = weak[self] == 42
end}
local function seed()
  local value = setmetatable({}, mt)
  weak[value] = 42
end
seed()
collectgarbage()
local after_first = next(weak) ~= nil
collectgarbage()
local after_second = next(weak) ~= nil
return seen_during_gc, after_first, after_second
'''
    )
    assert result == (True, True, False)


def test_collectgarbage_control_surface():
    lua = LuaRuntime()
    result = lua.execute(
        '''
local was_running = collectgarbage("isrunning")
collectgarbage("stop")
local stopped = collectgarbage("isrunning")
collectgarbage("restart")
local restarted = collectgarbage("isrunning")
local oldmode = collectgarbage("generational")
local currentmode = collectgarbage("incremental")
local oldpause = collectgarbage("param", "pause", 321)
local pause = collectgarbage("param", "pause")
local completed = collectgarbage("step", 1)
local count_is_number = type(collectgarbage("count")) == "number"
return was_running, stopped, restarted, oldmode, currentmode,
       oldpause, pause, completed, count_is_number
'''
    )
    assert result == (
        True,
        False,
        True,
        b"generational",
        b"generational",
        200,
        321,
        True,
        True,
    )


def test_finalizer_errors_are_warnings_not_lua_errors():
    lua = LuaRuntime()
    with pytest.warns(RuntimeWarning, match="error in __gc metamethod"):
        result = lua.execute(
            '''
local mt = {__gc = function(self) error("gc boom") end}
local function seed()
  local value = setmetatable({}, mt)
end
seed()
collectgarbage()
return 42
'''
        )
    assert result == 42
