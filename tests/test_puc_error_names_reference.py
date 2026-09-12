from __future__ import annotations

import pytest
from lupa.lua55 import LuaRuntime as ReferenceLuaRuntime

from luapyre import LuaRuntime


CASES = [
    b"return missing_value + 1",
    b"local captured; local function f() return captured + 1 end; return f()",
]


def _dump(source: bytes) -> bytes:
    ref = ReferenceLuaRuntime(encoding=None)
    dumper = ref.eval(
        "function(src) local f=assert(load(src,'@names.lua','t')); "
        "return string.dump(f,false) end"
    )
    return dumper(source)


def _reference(blob: bytes):
    ref = ReferenceLuaRuntime(encoding=None)
    runner = ref.eval(
        "function(blob) local f=assert(load(blob,'@ignored.lua','b')); "
        "local ok,err=pcall(f); return ok,err end"
    )
    return runner(blob)


def _luapyre(blob: bytes):
    lua = LuaRuntime()
    lua.set("blob", blob)
    return lua.execute(
        "local f,e=load(blob,'@ignored.lua','b'); assert(f,e); "
        "local ok,err=pcall(f); return ok,err"
    )


@pytest.mark.parametrize("source", CASES)
def test_puc_named_arithmetic_errors_match_lua_55(source):
    blob = _dump(source)
    assert _luapyre(blob) == _reference(blob)


def test_global_and_upvalue_suffixes_are_preserved():
    global_error = _luapyre(_dump(CASES[0]))
    upvalue_error = _luapyre(_dump(CASES[1]))
    assert global_error == (
        False,
        b"names.lua:1: attempt to perform arithmetic on a nil value (global 'missing_value')",
    )
    assert upvalue_error == (
        False,
        b"names.lua:1: attempt to perform arithmetic on a nil value (upvalue 'captured')",
    )
