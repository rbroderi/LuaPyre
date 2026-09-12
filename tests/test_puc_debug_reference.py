from __future__ import annotations

from lupa.lua55 import LuaRuntime as ReferenceLuaRuntime

from luapyre import LuaRuntime
from luapyre.puc55 import load_puc55_chunk


SOURCE = b'''local function fail()
  local marker = 40
  error("boom")
end
fail()
'''

NESTED_SOURCE = b'''local function outer()
  local function inner()
    local x = 1
    error("nested")
  end
  inner()
end
outer()
'''


def _dump(source: bytes, name: bytes, *, strip: bool) -> bytes:
    ref = ReferenceLuaRuntime(encoding=None)
    dumper = ref.eval(
        "function(src,name,strip) "
        "local f=assert(load(src,name,'t')); return string.dump(f,strip) end"
    )
    return dumper(source, name, strip)


def _protected_luapyre(blob: bytes):
    lua = LuaRuntime()
    lua.set("blob", blob)
    return lua.execute(
        "local f,e=load(blob,'@override-name.lua','b'); assert(f,e); "
        "local ok,err=pcall(f); return ok,err"
    )


def _protected_reference(blob: bytes):
    ref = ReferenceLuaRuntime(encoding=None)
    runner = ref.eval(
        "function(blob) local f,e=load(blob,'@override-name.lua','b'); "
        "assert(f,e); local ok,err=pcall(f); return ok,err end"
    )
    return runner(blob)


def test_unstripped_puc_chunk_preserves_exact_error_source_line():
    blob = _dump(SOURCE, b"@puc_debug.lua", strip=False)
    assert _protected_luapyre(blob) == _protected_reference(blob)
    assert _protected_luapyre(blob) == (False, b"puc_debug.lua:3: boom")


def test_nested_puc_prototypes_preserve_source_and_lines():
    blob = _dump(NESTED_SOURCE, b"@nested_debug.lua", strip=False)
    assert _protected_luapyre(blob) == _protected_reference(blob)
    assert _protected_luapyre(blob) == (False, b"nested_debug.lua:4: nested")

    proto = load_puc55_chunk(blob)
    assert proto.source == b"@nested_debug.lua"
    assert len(proto.lineinfo) == len(proto.code)
    assert proto.children
    outer = proto.children[0]
    assert outer.source == b"@nested_debug.lua"
    assert len(outer.lineinfo) == len(outer.code)
    assert outer.children
    inner = outer.children[0]
    assert inner.source == b"@nested_debug.lua"
    assert len(inner.lineinfo) == len(inner.code)
    assert 4 in inner.lineinfo


def test_stripped_puc_chunk_has_no_error_location():
    blob = _dump(SOURCE, b"@stripped_debug.lua", strip=True)
    assert _protected_luapyre(blob) == _protected_reference(blob)
    assert _protected_luapyre(blob) == (False, b"boom")

    proto = load_puc55_chunk(blob)
    assert proto.source is None
    assert proto.lineinfo
    assert set(proto.lineinfo) == {-1}


def test_translated_expansion_keeps_line_table_aligned_with_vm_code():
    source = b'''local t={1,2,3}
local function fail()
  return t.missing + 1
end
return fail()
'''
    blob = _dump(source, b"@expanded_debug.lua", strip=False)
    proto = load_puc55_chunk(blob)

    def walk(p):
        assert len(p.lineinfo) == len(p.code)
        for child in p.children:
            walk(child)

    walk(proto)
    assert _protected_luapyre(blob) == _protected_reference(blob)
