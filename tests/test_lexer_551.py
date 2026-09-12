from __future__ import annotations

import math

import pytest

from luapyre.errors import LuaSyntaxError
from luapyre.lexer import Lexer
from luapyre import LuaRuntime


def _values(source: str):
    return [token.value for token in Lexer(source).tokens() if token.kind == "NUMBER"]


def _strings(source: str):
    return [token.value for token in Lexer(source).tokens() if token.kind == "STRING"]


def test_lua_55_numeric_literal_forms():
    values = _values("0xff 0xFFFFFFFFFFFFFFFF 1e3 0x1p4 .5 0x1.8p1")
    assert values == [255, -1, 1000.0, 16.0, 0.5, 3.0]


def test_large_hex_integer_literal_falls_back_to_float():
    value = _values("0x10000000000000000")[0]
    assert type(value) is float
    assert value == float.fromhex("0x10000000000000000")


def test_lua_55_short_string_escapes():
    values = _strings(
        r'''"\200\210" "\xFF" "\u{D800}" "\u{7FFFFFFF}" "a\z  
          b"'''
    )
    assert values[0] == bytes((200, 210))
    assert values[1] == b"\xff"
    assert values[2] == bytes.fromhex("ed a0 80")
    assert values[3] == bytes.fromhex("fd bf bf bf bf bf")
    assert values[4] == b"ab"


def test_long_strings_and_comments_support_equals_delimiters():
    source = """--[=[ ignored [nested-ish] text ]=]\nreturn [==[first\nsecond]==]"""
    strings = _strings(source)
    assert strings == [b"first\nsecond"]
    assert LuaRuntime().execute(source) == b"first\nsecond"


def test_long_string_drops_one_initial_newline():
    assert _strings("[=[\nabc]=]") == [b"abc"]


@pytest.mark.parametrize(
    "source, message",
    [
        (r'''"\q"''', "invalid escape sequence"),
        (r'''"\300"''', "decimal escape too large"),
        (r'''"\u{80000000}"''', "UTF-8 value too large"),
        ("0x", "malformed number"),
        ("1e+", "malformed number"),
        ("123abc", "malformed number"),
    ],
)
def test_invalid_lua_55_literals_are_rejected(source, message):
    with pytest.raises(LuaSyntaxError, match=message):
        Lexer(source).tokens()


def test_literals_execute_through_parser_and_vm():
    result = LuaRuntime().execute(
        r'''return 0xff, 0xFFFFFFFFFFFFFFFF, 0x1p4, "\200\210", "\u{D800}"'''
    )
    assert result == (255, -1, 16.0, bytes((200, 210)), bytes.fromhex("ed a0 80"))
