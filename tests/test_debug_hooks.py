from luapyre import LuaRuntime


def test_debug_hooks_are_disabled_by_default():
    lua = LuaRuntime()
    assert lua.execute("return debug.sethook, debug.gethook()") == (
        None,
        None,
        b"",
        0,
    )


def test_opt_in_hook_reports_call_line_return_and_does_not_reenter():
    lua = LuaRuntime(debug_hooks=True, jit=False)
    trace = lua.execute(
        """
        local trace = {}
        local function hook(event)
          trace[#trace + 1] = event
        end
        local function target()
          local value = 1
          return value
        end
        debug.sethook(hook, "clr")
        target()
        debug.sethook()
        return table.concat(trace, ",")
        """
    )
    assert trace == b"return,line,call,line,line,return,line,call"


def test_hooks_are_owned_by_the_selected_coroutine():
    lua = LuaRuntime(debug_hooks=True, jit=False)
    trace = lua.execute(
        """
        local trace = {}
        local co = coroutine.create(function ()
          coroutine.yield(10)
          return 20
        end)
        debug.sethook(co, function (event)
          trace[#trace + 1] = event
        end, "clr")
        repeat until not coroutine.resume(co)
        assert(not debug.gethook())
        return table.concat(trace, ",")
        """
    )
    assert trace == b"call,line,call,return,line,return"


def test_count_hook_round_trips_configuration():
    lua = LuaRuntime(debug_hooks=True, jit=False)
    result = lua.execute(
        """
        local count = 0
        local function hook(event)
          assert(event == "count")
          count = count + 1
        end
        debug.sethook(hook, "", 4)
        local configured, mask, interval = debug.gethook()
        local total = 0
        for i = 1, 20 do total = total + i end
        debug.sethook()
        return configured == hook, mask, interval, count > 0, total
        """
    )
    assert result == (True, b"", 4, True, 210)


def test_stripped_function_gets_one_nil_line_hook_and_anonymous_locals():
    lua = LuaRuntime(debug_hooks=True, jit=False)
    assert lua.execute(
        """
local source = [[
local debug = require "debug"
local value = 12
local name, seen = debug.getlocal(1, 1)
return name, seen == debug, value
]]
local stripped = assert(load(string.dump(assert(load(source)), true)))
local lines = 0
local saw_nil = false
debug.sethook(function (event, line)
  assert(event == "line")
  lines = lines + 1
  if line == nil then saw_nil = true end
end, "l")
local name, same, value = stripped()
debug.sethook()
return name, same, value, lines > 0, saw_nil
"""
    ) == (b"(temporary)", True, 12, True, True)


def test_traceback_compacts_deep_stacks_to_bounded_head_and_tail():
    lua = LuaRuntime(debug_hooks=True, jit=False)
    trace = lua.execute(
        """
local function deep(n)
  if n == 0 then return debug.traceback("message", 1) end
  local result = deep(n - 1)
  return result
end
return deep(60)
"""
    )
    assert b"...\t(skipping " in trace
    assert trace.count(b"\n") <= 23
