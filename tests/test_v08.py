from __future__ import annotations

from lupa.lua55 import LuaRuntime as ReferenceLuaRuntime

from luapyre import LuaRuntime
from luapyre.binary_chunks import PUC_MAGIC
from luapyre.native_debug_chunks import PUC55_HEADER


def run(source):
    return LuaRuntime().execute(source)


def puc_dump(source: bytes) -> bytes:
    lua = ReferenceLuaRuntime(encoding=None)
    dump = lua.eval("function(src) return string.dump(assert(load(src)), true) end")
    return dump(source)


def run_puc(source: bytes):
    lua = LuaRuntime()
    lua.set("blob", puc_dump(source))
    return lua.execute('local f,e=load(blob,"puc","b"); assert(f,e); return f()')


def test_string_dump_roundtrips_luapyre_function():
    assert run('''
local dumped = string.dump(function(a,b) return a+b, a*b end)
local f,err = load(dumped, "roundtrip", "b")
return string.byte(dumped,1), err, f(6,7)
''') == (27, None, 13, 42)


def test_dump_uses_lua_55_header_and_strip_argument():
    lua = LuaRuntime()
    dumped = lua.execute('return string.dump(function() return 42 end, true)')
    assert dumped.startswith(PUC55_HEADER)


def test_dumped_function_gets_fresh_first_upvalue_from_load_environment():
    assert run('''
local dumped = string.dump(function() return answer end)
local f = assert(load(dumped, "env", "b", {answer=42}))
return f()
''') == 42


def test_load_modes_distinguish_text_and_binary():
    assert run('''
local dumped = string.dump(function() return 1 end)
local a,ea = load(dumped,nil,"t")
local b,eb = load("return 1",nil,"b")
return a, type(ea), b, type(eb)
''') == (None, b"string", None, b"string")


def test_binary_reader_function_can_supply_multiple_pieces():
    assert run('''
local dumped = string.dump(function() return 42 end)
local at = 1
local function reader()
  if at > #dumped then return nil end
  local part = string.sub(dumped, at, at + 4)
  at = at + 5
  return part
end
local f,err = load(reader,"reader","b")
return err, f()
''') == (None, 42)


def test_corrupt_native_chunk_fails_as_lua_load_error():
    lua = LuaRuntime()
    dumped = lua.execute('return string.dump(function() return 42 end)')
    lua.set("bad", dumped[:-3])
    result = lua.execute('local f,e=load(bad,nil,"b"); return f,type(e)')
    assert result == (None, b"string")


def test_puc_lua_55_simple_binary_chunk_executes():
    blob = puc_dump(b"return 40 + 2, 'ok'")
    assert blob.startswith(PUC_MAGIC)
    lua = LuaRuntime()
    lua.set("blob", blob)
    assert lua.execute('local f,e=load(blob,"puc","b"); return e,f()') == (None, 42, b"ok")


def test_puc_table_fields_and_updates():
    assert run_puc(b"local t={10,20,x=5}; t[2]=t[2]+1; return t[1],t[2],t.x") == (10, 21, 5)


def test_puc_numeric_for_and_branches():
    assert run_puc(b"local s=0; for i=1,10 do if i%2==0 then s=s+i end end; return s") == 30


def test_puc_closure_shared_upvalue_mutation():
    assert run_puc(b"local x=1; local function f() x=x+1; return x end; return f(),f()") == (2, 3)


def test_puc_vararg_multiple_results():
    assert run_puc(b"local function f(...) return select('#',...), ... end; return f(1,nil,3)") == (3, 1, None, 3)


def test_puc_generic_for():
    assert run_puc(b"local s=0; for k,v in pairs({10,20,30}) do s=s+v end; return s") == 60


def test_puc_arithmetic_metamethod_path():
    assert run_puc(b"local t=setmetatable({}, {__add=function(a,b) return 42 end}); return t+1") == 42


def test_puc_gc_keeps_live_values_across_numeric_and_generic_loops():
    assert run_puc(b'''
local weak=setmetatable({}, {__mode="v"})
local held={answer=42}
weak[1]=held
local total=0
for i=1,2 do
  collectgarbage()
  total=total+i
end
for _,v in pairs({3,4}) do
  collectgarbage()
  total=total+v
end
return weak[1] == held, held.answer, total
''') == (True, 42, 10)


def test_puc_tbc_value_is_a_gc_root_until_scope_close():
    assert run_puc(b'''
local weak=setmetatable({}, {__mode="v"})
local closed=0
do
  local value <close> = setmetatable({}, {__close=function() closed=closed+1 end})
  weak[1]=value
  collectgarbage()
  assert(weak[1] == value)
end
return closed
''') == 1


def test_truncated_puc_chunk_fails_closed():
    blob = puc_dump(b"return 42")
    lua = LuaRuntime()
    lua.set("bad", blob[:-5])
    assert lua.execute('local f,e=load(bad,nil,"b"); return f,type(e)') == (None, b"string")


def test_puc_version_and_platform_header_mismatch_fail_closed():
    blob = puc_dump(b"return 42")

    bad_version = bytearray(blob)
    bad_version[4] = 0x54
    lua = LuaRuntime()
    lua.set("bad", bytes(bad_version))
    assert lua.execute('local f,e=load(bad,nil,"b"); return f,type(e)') == (None, b"string")

    # signature(4) + version + format + LUAC_DATA(6) => sizeof(int) at byte 12
    bad_size = bytearray(blob)
    bad_size[12] = 8
    lua = LuaRuntime()
    lua.set("bad", bytes(bad_size))
    assert lua.execute('local f,e=load(bad,nil,"b"); return f,type(e)') == (None, b"string")
