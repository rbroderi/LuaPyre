import pytest

from luapyre import LuaRuntime, LuaRuntimeError, LuaSyntaxError, LuaTypeError


def test_const_local_is_read_only_prefix_and_postfix_forms():
    lua = LuaRuntime()
    assert lua.execute("local <const> x = 42; return x") == 42
    assert lua.execute("local x <const> = 42; return x") == 42
    with pytest.raises(LuaTypeError, match="read-only local"):
        lua.compile("local x <const> = 1; x = 2")


def test_close_runs_at_normal_block_exit_in_reverse_order():
    src = '''
local log = ""
local mt = {__close = function(self) log = log .. self.name end}
do
    local a <close> = setmetatable({name="a"}, mt)
    local b <close> = setmetatable({name="b"}, mt)
end
return log
'''
    assert LuaRuntime().execute(src) == b"ba"


def test_close_runs_before_return_and_preserves_return_value():
    src = '''
local log = ""
local mt = {__close = function(self) log = log .. self.name end}
local function f()
    local x <close> = setmetatable({name="x"}, mt)
    return 42
end
local value = f()
return value, log
'''
    assert LuaRuntime().execute(src) == (42, b"x")


def test_close_runs_on_break():
    src = '''
local log = ""
local mt = {__close = function(self) log = log .. self.name end}
while true do
    local x <close> = setmetatable({name="x"}, mt)
    break
end
return log
'''
    assert LuaRuntime().execute(src) == b"x"


def test_goto_backward_and_forward():
    src = '''
local i = 0
::again::
i = i + 1
if i < 3 then goto again end
goto done
::unused::
i = 100
::done::
return i
'''
    assert LuaRuntime().execute(src) == 3


def test_goto_out_of_scope_closes_values():
    src = '''
local log = ""
local mt = {__close = function(self) log = log .. self.name end}
do
    local x <close> = setmetatable({name="x"}, mt)
    goto done
end
::done::
return log
'''
    assert LuaRuntime().execute(src) == b"x"


def test_goto_cannot_enter_local_scope():
    with pytest.raises(LuaSyntaxError, match="jumps into the scope"):
        LuaRuntime().compile('''
goto target
local x = 1
::target::
return x
''')


def test_missing_and_duplicate_visible_labels_are_errors():
    lua = LuaRuntime()
    with pytest.raises(LuaSyntaxError, match="no visible label"):
        lua.compile("goto nowhere")
    with pytest.raises(LuaSyntaxError, match="already defined"):
        lua.compile("::x:: do ::x:: end")


def test_close_runs_during_error_unwind_and_receives_error_object():
    lua = LuaRuntime()
    src = '''
closed = ""
seen_error = ""
local mt = {
    __close = function(self, err)
        closed = closed .. self.name
        seen_error = err
    end
}
local function f()
    local x <close> = setmetatable({name="x"}, mt)
    error("boom")
end
f()
'''
    with pytest.raises(LuaRuntimeError, match="boom"):
        lua.execute(src)
    assert lua.get("closed") == b"x"
    assert lua.get("seen_error") == b"boom"


def test_error_in_close_still_closes_earlier_values():
    lua = LuaRuntime()
    src = '''
closed = ""
local good = {__close = function(self, err) closed = closed .. self.name end}
local bad = {__close = function(self, err) error("close failed") end}
local function f()
    local a <close> = setmetatable({name="a"}, good)
    local b <close> = setmetatable({name="b"}, bad)
    error("body failed")
end
f()
'''
    with pytest.raises(LuaRuntimeError, match="close failed"):
        lua.execute(src)
    assert lua.get("closed") == b"a"


def test_nonclosable_value_is_rejected_at_declaration():
    with pytest.raises(LuaRuntimeError, match="non-closable"):
        LuaRuntime().execute("local x <close> = {}; return 1")
    assert LuaRuntime().execute("local x <close> = nil; return 42") == 42
    assert LuaRuntime().execute("local x <close> = false; return 42") == 42


def test_global_declaration_disables_implicit_globals():
    lua = LuaRuntime()
    assert lua.execute("global x; x = 42; return x") == 42
    with pytest.raises(LuaSyntaxError, match="not declared"):
        lua.compile("global x; return y")


def test_global_wildcard_and_const_wildcard():
    lua = LuaRuntime()
    assert lua.execute("global<const> *; return type(42)") == b"number"
    with pytest.raises(LuaTypeError, match="read-only global"):
        lua.compile("global<const> *; x = 1")


def test_named_global_can_override_const_wildcard():
    src = '''
global x
global<const> *
x = 42
return x, type(x)
'''
    assert LuaRuntime().execute(src) == (42, b"number")


def test_global_initializer_rejects_existing_value():
    lua = LuaRuntime()
    with pytest.raises(LuaRuntimeError, match="already has a value"):
        lua.execute("x = 1; global x = 2")


def test_global_initializer_is_typed_and_global_const_is_read_only():
    lua = LuaRuntime()
    assert lua.execute("global x: integer = 42; return x") == 42
    with pytest.raises(LuaTypeError):
        lua.compile('global x: integer; x = "bad"')
    with pytest.raises(LuaTypeError, match="read-only global"):
        lua.compile("global x <const> = 1; x = 2")


def test_global_function_is_declared_before_its_body_for_recursion():
    src = '''
global function fact(n)
    if n <= 1 then return 1 end
    return n * fact(n - 1)
end
return fact(6)
'''
    assert LuaRuntime().execute(src) == 720


def test_close_runs_before_tailcall_replaces_frame():
    src = '''
local log = ""
local mt = {__close = function(self) log = log .. "x" end}
local function target(v) return v end
local function source(v)
    local x <close> = setmetatable({}, mt)
    return target(v)
end
return source(42), log
'''
    assert LuaRuntime().execute(src) == (42, b"x")
