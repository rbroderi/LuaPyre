"""Differential coverage for the 0.7 safe standard-library tranche."""

from __future__ import annotations

import pytest
from lupa.lua55 import LuaRuntime as ReferenceLuaRuntime

from luapyre import LuaRuntime


CASES = [
    '''
local t = setmetatable({}, {__tostring=function() return "custom" end})
local a,b = select(2,10,20,22)
return tostring(t), select("#",1,nil,3), a,b, select(-1,4,5,6)
''',
    '''
local seen = 0
local function iter(state, control)
  if control == nil then return "x",42 end
end
local t = setmetatable({}, {__pairs=function(self)
  seen = seen + 1
  return iter,self,nil,nil
end})
local total = 0
for k,v in pairs(t) do total = total + v end
local a,b,c,d = pairs(t)
return seen,total,type(a),b==t,c,d
''',
    '''
local t = setmetatable({}, {__index=function(_,k)
  if k <= 3 then return k*10 end
end})
local total = 0
for i,v in ipairs(t) do total = total + v end
return total
''',
    '''
local ok,a,b = pcall(function(x) return x,x+2 end,40)
local ok2,e = pcall(function() error("boom") end)
local ok3,msg = xpcall(function() error("bad") end, function(err) return "handled:"..err end)
return ok,a,b,ok2,type(e),ok3,msg
''',
    '''
local f,err = load("return x + 2", "chunk", "t", {x=40})
return err,f()
''',
    '''
local t = table.create(4,2)
local p = table.pack("a",nil,"c")
local u = {1,2,3}
table.insert(u,2,9)
local removed = table.remove(u,3)
local dst = {}
table.move(u,1,#u,2,dst)
local a,b,c = table.unpack(u)
return type(t),p.n,p[1],p[2],p[3],removed,a,b,c,dst[2],dst[3],dst[4],table.concat({1,"x",3},":")
''',
    '''
local a={4,1,3,2}; table.sort(a)
local b={1,4,2,3}; table.sort(b,function(x,y) return x>y end)
return table.concat(a,","),table.concat(b,",")
''',
    '''
local i,f=math.modf(-3.25)
local m,e=math.frexp(8)
return math.abs(-5),math.floor(3.9),math.ceil(-3.9),i,f,
       math.type(1),math.type(1.5),math.tointeger(4.0),math.ult(-1,0),
       math.max(1,5,3),math.min(1,5,3),m,e
''',
    '''
math.randomseed(123,456)
return math.random(0),math.random(),math.random(-10,10),math.random(1,1000000)
''',
    '''
local a,b,c=string.byte("ABC",1,3)
return a,b,c,string.char(a,b,c),string.sub("abcdef",-3,-1),
       string.rep("ab",3,"-"),string.reverse("abc"),string.lower("ABC"),string.upper("abc"),
       ("hello"):upper()
''',
    '''
local i,j,word=string.find("!! hello 123","(%a+)")
local digits=string.match("id=42","=(%d+)")
local words={}
for w in string.gmatch("one two three","%a+") do words[#words+1]=w end
local changed,n=string.gsub("a1 b22","(%a)(%d+)","%2%1")
return i,j,word,digits,table.concat(words,"|"),changed,n
''',
    '''
local balanced=string.match("x(a(b)c)y","%b()")
local word=string.match("!abc!","%f[%a]%a+%f[%A]")
local repeated=string.match("abc abc","(%a+)%s+%1")
local p1,p2=string.match("xxaaay","()a+()")
return balanced,word,repeated,p1,p2
''',
    '''
local packed=string.pack("<i4I2c3",-12345,65530,"xy")
local a,b,c,nextpos=string.unpack("<i4I2c3",packed)
return a,b,c,nextpos,#packed,string.packsize("<i4I2c3")
''',
    '''
return string.format("%04d %.2f %s %q %%",7,2.5,"ok","a\\nb")
''',
    '''
local s=utf8.char(65,8364,128578)
local a,b,c=utf8.codepoint(s,1,-1)
local positions={}
for p,cp in utf8.codes(s) do positions[#positions+1]=p end
local p1,e1=utf8.offset(s,2)
local p2,e2=utf8.offset(s,0,p1)
return a,b,c,utf8.len(s),table.concat(positions,","),p1,e1,p2,e2
''',
    '''
local s=utf8.char(2147483647)
local cp=utf8.codepoint(s,1,-1,true)
return #s,cp,utf8.len(s,1,-1,true)
''',
]


def _encode(value):
    if value is None:
        return "nil"
    if type(value) is bool:
        return "boolean:" + str(value).lower()
    if type(value) is int:
        return f"integer:{value}"
    if type(value) is float:
        return "float:" + value.hex()
    if isinstance(value, bytes):
        return "string:" + value.hex()
    return type(value).__name__


def _normalise(result):
    values = result if isinstance(result, tuple) else (result,)
    return [str(len(values)), *(_encode(value) for value in values)]


def _reference(source):
    lua = ReferenceLuaRuntime(encoding=None)
    return _normalise(lua.execute(source))


@pytest.mark.parametrize("source", CASES)
def test_safe_stdlib_matches_lua_55(source):
    assert _normalise(LuaRuntime().execute(source)) == _reference(source)
