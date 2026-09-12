"""Differential execution of official Lua 5.5 binary chunks."""

from __future__ import annotations

import pytest
from lupa.lua55 import LuaRuntime as ReferenceLuaRuntime

from luapyre import LuaRuntime


CASES = [
    b"return 1+2*3, 7//2, 7%4, 2^5",
    b"local x=10; if x>5 then x=x+2 else x=0 end; return x, x==12, x~=11",
    b"local s=0; for i=-3,9,2 do s=s+i end; return s",
    b"local s=0; for k,v in pairs({a=10,b=20,c=30}) do s=s+v end; return s",
    b"local x=1; local function bump(n) x=x+n; return x end; return bump(2),bump(4),x",
    b"local function f(...) return select('#',...), ... end; return f('a',nil,'c')",
    b"local t={1,2,3,x=4}; table.insert(t,2,9); return #t,t[1],t[2],t[3],t.x",
    b"return string.sub('abcdef',2,-2), string.match('id=42','(%d+)'), ('abc'):upper()",
    b"local mt={__index=function(_,k) return k..'!' end}; local t=setmetatable({},mt); return t.hello",
    b"local t=setmetatable({}, {__add=function(a,b) return b*2 end}); return t+21",
    b"local function tail(n,acc) if n==0 then return acc end return tail(n-1,acc+n) end; return tail(20,0)",
    b"local function values() return 1,nil,3 end; local a,b,c=values(); return a,b,c,select('#',values())",
    b"local s=''; for i,v in ipairs({'a','b','c'}) do s=s..i..v end; return s",
    b"local a=5; local b=9; return a&b,a|b,a~b,a<<2,b>>1,~a",
]


def _values(result):
    return result if isinstance(result, tuple) else (result,)


def _reference(source: bytes):
    return _values(ReferenceLuaRuntime(encoding=None).execute(source))


def _binary(source: bytes):
    ref = ReferenceLuaRuntime(encoding=None)
    dump = ref.eval("function(src) return string.dump(assert(load(src)), true) end")
    blob = dump(source)
    lua = LuaRuntime()
    lua.set("blob", blob)
    return _values(lua.execute('local f,e=load(blob,"@binary-reference","b"); assert(f,e); return f()'))


@pytest.mark.parametrize("source", CASES)
def test_puc_binary_execution_matches_lua_55(source):
    assert _binary(source) == _reference(source)
