from __future__ import annotations

from lupa.lua55 import LuaRuntime as ReferenceLuaRuntime

from luapyre import LuaRuntime


def _values(result):
    return result if isinstance(result, tuple) else (result,)


def test_gsub_percent_one_is_whole_match_without_explicit_captures():
    source = "return string.gsub('abc', '%w', '%1%0'), string.gsub('abc', '%w+', '%0%1')"
    expected = _values(ReferenceLuaRuntime(encoding=None).execute(source.encode()))
    actual = _values(LuaRuntime().execute(source))
    # Only the last call in a return list expands its multiple results.
    assert actual == expected == (b"aabbcc", b"abcabc", 1)
