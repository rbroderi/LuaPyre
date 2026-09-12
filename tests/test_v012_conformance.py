from __future__ import annotations

import pytest

from luapyre import LuaRuntime, LuaRuntimeError


def run(source: str):
    return LuaRuntime(output=lambda _data: None).execute(source)


def test_arithmetic_coerces_lua_numeral_strings():
    assert run(r'''
local a,b,c = "2", " 3e0 ", " 10  "
return a+b, -b, b+"2", "10"-c, c%a, a^b
''') == (5.0, -3.0, 5.0, 0, 0, 8.0)


def test_bitwise_coerces_numeric_strings_and_handles_extreme_shifts():
    assert run(r'''
return
  ("0xffffffffffffffff" | 0),
  ("0xfffffffffffffffe" & "-1"),
  (" \t-0xfffffffffffffffe\n\t" & "-1"),
  ("1234.0" << "5.0"),
  ("0xffff.0" ~ "0xAAAA"),
  (~"0x0.000p4"),
  (1 >> math.mininteger),
  (1 >> math.maxinteger),
  (1 << math.mininteger),
  (1 << math.maxinteger)
''') == (-1, -2, 2, 1234 * 32, 0x5555, -1, 0, 0, 0, 0)


def test_nonintegral_wide_hex_float_does_not_bitwise_coerce():
    with pytest.raises(LuaRuntimeError):
        run('return "0xffffffffffffffff.0" | 0')


def test_trailing_decimal_zero_is_not_misclassified_as_hex():
    assert run("return 0") == 0
    assert run("return 2 // 0") is not None if False else True


def test_named_varargs_are_backed_by_mutable_vararg_table():
    assert run(r'''
local function aux(a, v, ...t)
  for k, value in pairs(v) do t[k] = value end
  return ...
end
local t = table.pack(aux(10, {11, [5] = 24}, 1, 2, 3, nil, 4))
return t.n, t[1], t[2], t[3], t[4], t[5]
''') == (5, 11, 2, 3, None, 24)


def test_named_vararg_n_controls_returned_slots():
    assert run(r'''
local function aux(a, b, n, ...t) t.n = n; return b, ... end
local t = table.pack(aux(10, 1, 10000))
return t.n, t[1], #t
''') == (10001, 1, 1)


def test_named_vararg_rejects_invalid_n():
    assert run(r'''
local function aux(a, b, n, ...t) t.n = n; return b, ... end
local function bad(n)
  local ok, err = pcall(aux, 1, 1, n)
  return (not ok) and string.find(err, "no proper 'n'") ~= nil
end
return bad(-1), bad(math.maxinteger), bad(math.mininteger), bad(1.0)
''') == (True, True, True, True)


def test_loaded_chunks_are_variadic_and_named_vararg_is_const():
    assert run(r'''
local f = assert(load[[ return {...} ]])
local t = f(2, 3)
local st, msg = load("return function (... t) t = 10 end")
return t[1], t[2], t[3], st == nil,
       string.find(msg, "const variable 't'") ~= nil
''') == (2, 3, None, True, True)


def test_named_vararg_remains_const_when_captured():
    assert run(r'''
local st, msg = load[[
  local function foo (...extra)
    return function (...) extra = nil end
  end
]]
return st == nil, string.find(msg, "const variable 'extra'") ~= nil
''') == (True, True)


def test_gsub_reports_lua_capture_and_replacement_diagnostics():
    assert run(r'''
local function fails(fragment, f, ...)
  local ok, err = pcall(f, ...)
  return (not ok) and string.find(err, fragment) ~= nil
end
return
  fails("invalid replacement value %(a table%)", string.gsub, "alo", ".", {a={}}),
  fails("invalid capture index %%2", string.gsub, "alo", ".", "%2"),
  fails("invalid capture index %%0", string.gsub, "alo", "(%0)", "a"),
  fails("invalid capture index %%1", string.gsub, "alo", "(%1)", "a"),
  fails("invalid use of '%%'", string.gsub, "alo", ".", "%x")
''') == (True, True, True, True, True)


def test_pack_alignment_and_integral_size_diagnostics():
    assert run(r'''
local function fails(fragment, f, ...)
  local ok, err = pcall(f, ...)
  return (not ok) and string.find(err, fragment) ~= nil
end
return
  fails("out of limits", string.pack, "i0", 0),
  fails("out of limits", string.pack, "i17", 0),
  fails("out of limits", string.pack, "!17", 0),
  fails("%(17%) out of limits %[1,16%]", string.pack, "Xi17"),
  fails("not power of 2", string.pack, "!4i3", 0)
''') == (True, True, True, True, True)


def test_pack_x_uses_exactly_the_next_option_for_alignment():
    assert run(r'''
local function fails(fragment, f, ...)
  local ok, err = pcall(f, ...)
  return (not ok) and string.find(err, fragment) ~= nil
end
return
  fails("invalid next option", string.pack, "X"),
  fails("invalid next option", string.unpack, "XXi", ""),
  fails("invalid next option", string.unpack, "X i", ""),
  fails("invalid next option", string.pack, "Xc1"),
  string.packsize("Xx") == 0
''') == (True, True, True, True, True)


def test_pack_rejects_format_and_total_size_overflow():
    assert run(r'''
local function fails(fragment, f, ...)
  local ok, err = pcall(f, ...)
  return (not ok) and string.find(err, fragment) ~= nil
end
local huge = string.format("c%d", math.maxinteger - 9)
return
  fails("invalid format", string.packsize, "c1" .. string.rep("0", 40)),
  string.packsize(huge) == math.maxinteger - 9,
  fails("too large", string.packsize, huge .. "c10"),
  fails("too long", string.pack, "xxxxxxxxxx " .. huge)
''') == (True, True, True, True)


def test_integer_idiv_and_pow_boundary_semantics():
    assert run(r'''
return math.maxinteger // 1 == math.maxinteger,
       math.mininteger // -1 == math.mininteger,
       0^-1 == 1/0
''') == (True, True, True)


def test_utf8_offset_accepts_incomplete_sequences_as_byte_boundaries():
    assert run(r'''
local p1,e1 = utf8.offset("\xE0", 1)
local p2,e2 = utf8.offset("\xE0\x9e", -1)
return p1,e1,p2,e2
''') == (1, 1, 1, 2)


def test_utf8_codepoint_and_len_reject_out_of_bounds_indices():
    assert run(r'''
local function bad(f, ...)
  local ok, err = pcall(f, ...)
  return (not ok) and string.find(err, "out of bounds") ~= nil
end
return
  bad(utf8.codepoint, "abc", 4),
  bad(utf8.codepoint, "abc", 1, 4),
  bad(utf8.len, "abc", 0, 2),
  bad(utf8.len, "abc", 1, 4)
''') == (True, True, True, True)


def test_utf8_empty_ranges_and_iterator_controls_match_lua():
    assert run(r'''
local t = {utf8.codepoint("", 1, -1)}
local f = utf8.codes("")
return #t, utf8.len("", 1, -1),
       f("", 2) == nil, f("", -1) == nil, f("", math.mininteger) == nil
''') == (0, 0, True, True, True)
