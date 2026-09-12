from __future__ import annotations

import pytest

from luapyre import LuaRuntime, LuaRuntimeError
from luapyre.native_debug_chunks import DEBUG_NATIVE_MAGIC


def run(source):
    return LuaRuntime().execute(source)


def test_default_error_decorates_string_with_named_source_and_line():
    assert run('''
local f = assert(load("error('boom')", "=named"))
local ok, err = pcall(f)
return ok, err
''') == (False, b"named:1: boom")


def test_error_level_zero_preserves_raw_string_and_non_string_identity():
    assert run('''
local marker = {}
local ok1, e1 = pcall(function() error("boom", 0) end)
local ok2, e2 = pcall(function() error(marker) end)
return ok1, e1, ok2, e2 == marker
''') == (False, b"boom", False, True)


def test_error_level_two_points_at_calling_lua_frame():
    assert run('''
local f = assert(load([[local function outer()
  local function inner()
    error("boom", 2)
  end
  inner()
end
outer()]], "=levels"))
local ok, err = pcall(f)
return ok, err
''') == (False, b"levels:5: boom")


def test_assert_uses_call_site_location():
    assert run('''
local f = assert(load("assert(false)", "=assertion"))
local ok, err = pcall(f)
return ok, err
''') == (False, b"assertion:1: assertion failed!")


def test_vm_generated_error_gets_source_line_prefix():
    ok, err = run('''
local f = assert(load("local x=nil\\nreturn x+1", "=calc"))
return pcall(f)
''')
    assert ok is False
    assert err.startswith(b"calc:2: ")
    assert b"arithmetic" in err


def test_load_default_string_chunkname_is_source_text():
    ok, err = run('''
local f = assert(load("error('x')"))
return pcall(f)
''')
    assert ok is False
    assert err == b'[string "error(\'x\')"]:1: x'


def test_load_reader_default_chunkname_is_load():
    assert run('''
local done = false
local function reader()
  if done then return nil end
  done = true
  return "error('x')"
end
local f = assert(load(reader))
local ok, err = pcall(f)
return ok, err
''') == (False, b"(load):1: x")


def test_native_dump_preserves_debug_metadata_unless_stripped():
    lua = LuaRuntime()
    result = lua.execute('''
local maker = assert(load("return function()\\n  error('boom')\\nend", "=origin"))
local f = maker()
local full = string.dump(f, false)
local stripped = string.dump(f, true)
local a = assert(load(full, "=replacement", "b"))
local b = assert(load(stripped, "=replacement", "b"))
local oka, ea = pcall(a)
local okb, eb = pcall(b)
return full, stripped, oka, ea, okb, eb
''')
    full, stripped, oka, ea, okb, eb = result
    assert full.startswith(DEBUG_NATIVE_MAGIC)
    assert stripped.startswith(DEBUG_NATIVE_MAGIC)
    assert oka is False and ea == b"origin:2: boom"
    assert okb is False and eb == b"boom"


def test_structured_exception_captures_lua_call_stack_and_formats_traceback():
    lua = LuaRuntime()
    source = '''local function outer()
  local function inner()
    error("boom")
  end
  inner()
end
outer()
'''
    with pytest.raises(LuaRuntimeError) as raised:
        lua.execute(source, chunkname="=trace")
    error = raised.value
    assert error.value == b"trace:3: boom"
    assert [(frame.name, frame.line) for frame in error.trace[:3]] == [
        ("inner", 3),
        ("outer", 5),
        ("<chunk>", 7),
    ]
    rendered = lua.traceback(error)
    assert rendered.startswith(b"trace:3: boom\nstack traceback:")
    assert b"trace:3: in function 'inner'" in rendered
    assert b"trace:5: in function 'outer'" in rendered
    assert b"trace:7: in main chunk" in rendered


def test_load_syntax_error_reports_requested_chunk_name_and_line():
    f, typ, err = run('''
local f, err = load("local = 1", "=bad")
return f, type(err), err
''')
    assert f is None
    assert typ == b"string"
    assert err.startswith(b"bad:1: ")
