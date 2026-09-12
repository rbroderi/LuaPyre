from __future__ import annotations

from luapyre import LuaRuntime


def run(source: str):
    return LuaRuntime(output=lambda _data: None).execute(source)


def test_collectgarbage_count_is_stable_across_nonallocating_named_vararg_call():
    assert run(r'''
local function notab(keys, t, ...v)
  for _, k in pairs(keys) do assert(t[k] == v[k]) end
  assert(t.n == v.n)
  return ...
end
local t = table.pack(10, 20, 30)
local keys = {-1, 0, 1, t.n, t.n + 1, 1.0, 1.1, "n", print, "k", "1"}
notab(keys, t, 10, 20, 30)
local m = collectgarbage"count"
notab(keys, t, 10, 20, 30)
return m == collectgarbage"count"
''') is True


def test_math_huge_integer_conversion_names_field():
    assert run(r'''
local ok, err = pcall(function () return math.huge << 1 end)
return not ok and string.find(err, "field 'huge'") ~= nil
''') is True


def test_tonumber_leading_zero_decimal_matches_source_arithmetic():
    assert run("return tonumber('-012')") == -12
    assert run("return -010-2") == -12
    assert run("return tonumber('-012') == -010-2") is True


def test_tonumber_plain_hexadecimal_overflow_wraps_to_lua_integer():
    assert run('return tonumber("0x1000000000000000000000000000000")') == 0
    assert run('return tonumber("0xffffffffffffffff")') == -1
    assert run('return tonumber("-0xffffffffffffffff")') == 1


def test_tonumber_long_hexadecimal_float_overflow_returns_infinity():
    assert run("return tonumber('0x' .. string.rep('f', 300) .. '.0') == math.huge") is True
    assert run("return tonumber('-0x' .. string.rep('f', 300) .. '.0') == -math.huge") is True
    assert run("return tonumber('0xe03' .. string.rep('0', 1000) .. 'p-4000')") == 3587.0


def test_float_modulo_matches_lua_nan_and_infinity_edges():
    assert run("local x = 0.0 % 0; return x ~= x") is True
    assert run("local x = 1.3 % 0; return x ~= x") is True
    assert run("local x = math.huge % 1; return x ~= x") is True
    assert run("return 1 % math.huge") == 1.0
    assert run("return 1e30 % math.huge") == 1e30
    assert run("return 1e30 % -math.huge") == float("-inf")
    assert run("return -1 % math.huge") == float("inf")
    assert run("return -1 % -math.huge") == -1.0


def test_integer_modulo_by_zero_still_errors():
    assert run("local ok = pcall(function () return 1 % 0 end); return ok") is False
