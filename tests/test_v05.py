from luapyre import LuaRuntime


def run(source):
    return LuaRuntime().execute(source)


def test_yield_resume_round_trip_and_status():
    assert run('''
local co = coroutine.create(function(a)
  local x, y = coroutine.yield(a + 1, a + 2)
  return x + y
end)
local ok1, a, b = coroutine.resume(co, 10)
local s1 = coroutine.status(co)
local ok2, c = coroutine.resume(co, 20, 22)
return ok1, a, b, s1, ok2, c, coroutine.status(co), type(co)
''') == (True, 11, 12, b"suspended", True, 42, b"dead", b"thread")


def test_yield_inside_nested_lua_call():
    assert run('''
local function inner()
  return coroutine.yield("inner")
end
local co = coroutine.create(function()
  return inner()
end)
local ok1, value = coroutine.resume(co)
local ok2, final = coroutine.resume(co, 42)
return ok1, value, ok2, final
''') == (True, b"inner", True, 42)


def test_running_main_and_coroutine_identity():
    assert run('''
local main, ismain = coroutine.running()
local co
co = coroutine.create(function()
  local self, child_is_main = coroutine.running()
  return self == co, child_is_main, coroutine.isyieldable()
end)
local ok, same, childmain, yieldable = coroutine.resume(co)
return type(main), ismain, coroutine.isyieldable(main), ok, same, childmain, yieldable
''') == (b"thread", True, False, True, True, False, True)


def test_parent_is_normal_while_child_runs():
    assert run('''
local parent
local child = coroutine.create(function()
  return coroutine.status(parent)
end)
parent = coroutine.create(function()
  local ok, status = coroutine.resume(child)
  return ok, status
end)
local ok, childok, status = coroutine.resume(parent)
return ok, childok, status
''') == (True, True, b"normal")


def test_wrap_yields_and_returns_without_status_boolean():
    assert run('''
local f = coroutine.wrap(function(x)
  local y = coroutine.yield(x + 1)
  return y + 1
end)
local first = f(10)
local second = f(41)
return first, second
''') == (11, 42)


def test_close_suspended_coroutine_runs_pending_close():
    assert run('''
local closed = 0
local co = coroutine.create(function()
  local resource <close> = setmetatable({}, {
    __close = function(self, err) closed = closed + 1 end
  })
  coroutine.yield("paused")
  closed = 100
end)
local ok1, value = coroutine.resume(co)
local ok2 = coroutine.close(co)
return ok1, value, ok2, closed, coroutine.status(co)
''') == (True, b"paused", True, 1, b"dead")


def test_error_keeps_stack_for_coroutine_close():
    result = run('''
local closed = 0
local seen
local co = coroutine.create(function()
  local resource <close> = setmetatable({}, {
    __close = function(self, err)
      closed = closed + 1
      seen = err
    end
  })
  coroutine.yield("ready")
  error("boom", 0)
end)
local ok1 = coroutine.resume(co)
local ok2, err = coroutine.resume(co)
local closeok, closeerr = coroutine.close(co)
return ok1, ok2, err, closeok, closeerr, closed, seen, coroutine.status(co)
''')
    assert result == (True, False, b"boom", False, b"boom", 1, b"boom", b"dead")


def test_resume_dead_coroutine_returns_failure():
    assert run('''
local co = coroutine.create(function() return 42 end)
local ok1, value = coroutine.resume(co)
local ok2, err = coroutine.resume(co)
return ok1, value, ok2, err
''') == (True, 42, False, b"cannot resume dead coroutine")


def test_running_coroutine_can_close_itself():
    assert run('''
local closed = 0
local co = coroutine.create(function()
  local resource <close> = setmetatable({}, {
    __close = function(self, err) closed = closed + 1 end
  })
  coroutine.close()
  closed = 100
  return 99
end)
local ok, value = coroutine.resume(co)
return ok, value, closed, coroutine.status(co)
''') == (True, None, 1, b"dead")
