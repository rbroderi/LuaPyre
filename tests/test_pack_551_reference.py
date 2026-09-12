from __future__ import annotations

from lupa.lua55 import LuaRuntime as ReferenceLuaRuntime

from luapyre import LuaRuntime


def _values(result):
    return result if isinstance(result, tuple) else (result,)


def test_wide_integer_unpack_uses_lua_integer_low_bits_and_extension_rules():
    source = r'''
      local n = 0x0807060504030201
      local neg = -n
      local packed = string.pack("<j", neg)
      return
        string.unpack("<i9", packed .. "\xFF"),
        string.unpack("<I9", packed .. "\0")
    '''
    reference = _values(ReferenceLuaRuntime(encoding=None).execute(source.encode()))
    actual = _values(LuaRuntime().execute(source))
    assert actual == reference
    assert actual[0] == -0x0807060504030201
    assert actual[1] == -0x0807060504030201
    assert actual[2] == 10


def test_wide_integer_unpack_rejects_non_extension_bytes():
    source = r'''
      local ok1, err1 = pcall(string.unpack, "<I9", ("\0"):rep(8) .. "\1")
      local ok2, err2 = pcall(string.unpack, ">i9", "\1" .. ("\0"):rep(8))
      return ok1, string.find(err1, "does not fit") ~= nil,
             ok2, string.find(err2, "does not fit") ~= nil
    '''
    expected = _values(ReferenceLuaRuntime(encoding=None).execute(source.encode()))
    assert _values(LuaRuntime().execute(source)) == expected == (False, True, False, True)
