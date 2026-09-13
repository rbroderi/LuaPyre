from __future__ import annotations

from luapyre import LuaRuntime


def test_io_is_memory_backed_and_cannot_read_or_write_host_files(tmp_path):
    host_file = tmp_path / "host.txt"
    host_file.write_bytes(b"host-secret")
    output = []
    lua = LuaRuntime(output=output.append)

    result = lua.execute(
        f'''local denied = io.open({str(host_file)!r}, "r")
local f = assert(io.open("virtual.lua", "w+"))
assert(f:write("return 42"))
assert(f:seek("set", 0) == 0)
local contents = f:read("a")
assert(f:close())
io.write("safe")
return denied, contents, dofile("virtual.lua"), io.type(f), os.execute
'''
    )

    assert result == (None, b"return 42", 42, b"closed file", None)
    assert output == [b"safe"]
    assert host_file.read_bytes() == b"host-secret"


def test_os_mutation_is_limited_to_virtual_files(tmp_path):
    host_file = tmp_path / "host.txt"
    host_file.write_bytes(b"unchanged")
    lua = LuaRuntime(output=lambda _data: None)

    result = lua.execute(
        f'''local f = assert(io.open("old", "w")); f:write("x"); f:close()
local renamed = os.rename("old", "new")
local removed = os.remove("new")
local denied = os.remove({str(host_file)!r})
return renamed, removed, denied, os.getenv("HOME"), os.setlocale("C"), os.setlocale("en_US")
'''
    )

    assert result == (True, True, None, None, b"C", None)
    assert host_file.read_bytes() == b"unchanged"


def test_os_getenv_reads_only_the_explicit_private_environment():
    lua = LuaRuntime(output=lambda _data: None)
    lua.set_environment({"VISIBLE": "sandbox-value"})
    assert lua.execute('return os.getenv("VISIBLE"), os.getenv("HOME")') == (
        b"sandbox-value",
        None,
    )


def test_pairs_and_dofile_callbacks_can_yield_without_host_access():
    lua = LuaRuntime(output=lambda _data: None)
    lua.capabilities.write_virtual_file(
        "yield.lua", b"local x = coroutine.yield(20); return x + 2"
    )
    assert lua.execute(
        '''local t = setmetatable({}, {__pairs=function ()
  coroutine.yield(10)
  return next, {}, nil
end})
local first = coroutine.wrap(function () for _ in pairs(t) do end end)
local second = coroutine.wrap(function () return dofile("yield.lua") end)
return first(), first(), second(), second(40)
'''
    ) == (10, None, 20, 42)


def test_debug_exposes_lua_state_without_python_or_hook_access():
    lua = LuaRuntime(output=lambda _data: None)
    result = lua.execute(
        '''local x = 10
local function f() return x end
local name, value = debug.getupvalue(f, 1)
local info = debug.getinfo(f)
return name, value, info.what, info.nparams,
       debug.getlocal, debug.setlocal, debug.sethook, debug.debug
'''
    )
    assert result[:4] == (b"x", 10, b"Lua", 0)
    assert result[4] is not None
    assert result[5:] == (None, None, None)
