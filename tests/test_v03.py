import pytest

from luapyre import LuaRuntime, LuaTypeError, LuaRuntimeError


def run(source):
    return LuaRuntime().execute(source)


def test_index_and_newindex_metamethods():
    src = '''
local backing = {x = 40}
local proxy = {}
setmetatable(proxy, {
  __index = function(t, k) return backing[k] end,
  __newindex = function(t, k, v) backing[k] = v end
})
proxy.x = proxy.x + 2
return proxy.x, rawget(proxy, "x"), backing.x
'''
    assert run(src) == (42, None, 42)


def test_index_and_newindex_table_chains():
    assert run('local p={x=42}; local t={}; setmetatable(t,{__index=p}); return t.x') == 42
    assert run('local p={}; local t={}; setmetatable(t,{__newindex=p}); t.x=42; return p.x,rawget(t,"x")') == (42, None)


def test_callable_table_used_as_index_is_still_indexed_as_a_table():
    source = '''
local methods = {answer = function() return 42 end}
setmetatable(methods, {__call = function() return "wrong" end})
local object = setmetatable({}, {__index = methods})
return object.answer()
'''
    assert run(source) == 42


def test_parentheses_collapse_multiple_results():
    source = '''
local function pair() return 20, 22 end
local function count(...) return select("#", ...), ... end
return count((pair()))
'''
    assert run(source) == (1, 20)


def test_native_style_library_calls_ignore_extra_arguments_and_optional_nil():
    source = '''
local t = {}
local mt = {}
return setmetatable(t, mt, "ignored") == t,
       string.sub("abc", 2, nil),
       tostring(42, "ignored"),
       string.upper("x", 1, 2)
'''
    assert run(source) == (True, b"bc", b"42", b"X")


def test_call_metamethod():
    assert run('local t={n=40}; setmetatable(t,{__call=function(self,x) return self.n+x end}); return t(2)') == 42


def test_arithmetic_and_unary_metamethods():
    src = '''
local mt = {
  __add = function(a,b) return a.n+b.n end,
  __unm = function(a) return -a.n end
}
local a=setmetatable({n=20},mt)
local b=setmetatable({n=22},mt)
return a+b, -b
'''
    assert run(src) == (42, -22)


def test_len_concat_and_comparison_metamethods():
    src = '''
local mt={
 __len=function(a) return 42 end,
 __concat=function(a,b) return a.s .. b.s end,
 __lt=function(a,b) return a.n < b.n end,
 __le=function(a,b) return a.n <= b.n end,
 __eq=function(a,b) return a.n == b.n end
}
local a=setmetatable({s="a",n=1},mt)
local b=setmetatable({s="b",n=2},mt)
local c=setmetatable({s="c",n=1},mt)
return #a, a..b, a<b, a<=c, a==c
'''
    assert run(src) == (42, b"ab", True, True, True)


def test_metatable_protection():
    assert run('local t={}; local mt={__metatable="locked"}; setmetatable(t,mt); return getmetatable(t)') == b"locked"
    with pytest.raises(LuaRuntimeError):
        run('local t={}; local mt={__metatable="locked"}; setmetatable(t,mt); setmetatable(t,{})')


def test_do_repeat_and_break():
    assert run('local x=1; do local x=40; x=x+2 end; return x') == 1
    assert run('local x=0; repeat x=x+1; if x==5 then break end until false; return x') == 5


def test_numeric_for_loops_and_readonly_control():
    assert run('local s=0; for i=1,10 do s=s+i end; return s') == 55
    assert run('local s=0; for i=5,1,-2 do s=s+i end; return s') == 9
    with pytest.raises(LuaTypeError):
        LuaRuntime().compile('for i=1,3 do i=7 end')


def test_generic_for_pairs_and_ipairs():
    assert run('local s=0; for k,v in pairs({a=20,b=22}) do s=s+v end; return s') == 42
    assert run('local s=0; for i,v in ipairs({10,20,12}) do s=s+v end; return s') == 42


def test_generic_for_stops_only_on_nil_control():
    src = '''
local n=0
local function iter(state, control)
  n=n+1
  if n==1 then return false,20 end
  if n==2 then return true,22 end
  return nil
end
local s=0
for k,v in iter,nil,nil do s=s+v end
return s
'''
    assert run(src) == 42


def test_anonymous_function_and_method_syntax():
    assert run('local f=function(x) return x+2 end; return f(40)') == 42
    assert run('local t={n=40}; function t:add(x) return self.n+x end; return t:add(2)') == 42
    assert run('local t={}; function t.add(a,b) return a+b end; return t.add(20,22)') == 42


def test_proper_tailcall_reuses_frame():
    lua = LuaRuntime(max_frames=8, fuel=200000)
    src = '''
local function loop(n, acc)
  if n == 0 then return acc end
  return loop(n-1, acc+1)
end
return loop(5000,0)
'''
    assert lua.execute(src) == 5000
    assert "TAILCALL" in lua.disassemble(src)


def test_index_chain_loop_guard():
    with pytest.raises(LuaRuntimeError, match="chain too long"):
        run('local a={}; local b={}; setmetatable(a,{__index=b}); setmetatable(b,{__index=a}); return a.x')
