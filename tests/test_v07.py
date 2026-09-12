from __future__ import annotations

from luapyre import LuaRuntime


def run(source):
    return LuaRuntime().execute(source)


def test_safe_library_tables_are_installed_without_host_io_libraries():
    assert run('return type(string), type(table), type(math), type(utf8), io, os, type(package), debug') == (
        b'table', b'table', b'table', b'table', None, None, b'table', None
    )


def test_string_metatable_enables_method_syntax():
    assert run('return ("hello"):upper(), ("ABCDE"):sub(2, -2), getmetatable("").__index == string') == (
        b'HELLO', b'BCD', True
    )


def test_tostring_metamethod_and_select():
    assert run('''
local t = setmetatable({}, {__tostring=function() return "custom" end})
local a,b = select(2, 10,20,22)
return tostring(t), select("#", 1,nil,3), a,b, select(-1, 4,5,6)
''') == (b'custom', 3, 20, 22, 6)


def test_pairs_honors_pairs_metamethod_and_fourth_result_shape():
    assert run('''
local seen = 0
local function iter(state, control)
  if control == nil then return "x", 42 end
end
local t = setmetatable({}, {__pairs=function(self)
  seen = seen + 1
  return iter, self, nil, nil
end})
local value
for k,v in pairs(t) do value = k .. v end
local a,b,c,d = pairs(t)
return seen, value, type(a), b == t, c, d
''') == (2, b'x42', b'function', True, None, None)


def test_ipairs_uses_regular_indexing():
    assert run('''
local t = setmetatable({}, {__index=function(_, k)
  if k <= 3 then return k * 10 end
end})
local total = 0
for i,v in ipairs(t) do total = total + v end
return total
''') == 60


def test_pcall_and_xpcall_preserve_error_objects():
    assert run('''
local ok,a,b = pcall(function(x) return x, x+2 end, 40)
local marker = {}
local ok2,err = pcall(function() error(marker) end)
local ok3,msg = xpcall(function() error("boom", 0) end, function(e) return "handled:" .. e end)
return ok,a,b,ok2,err == marker,ok3,msg
''') == (True, 40, 42, False, True, False, b'handled:boom')


def test_load_text_chunk_with_explicit_environment():
    assert run('''
local f,err = load("return x + 2", "chunk", "t", {x=40})
return err, f()
''') == (None, 42)


def test_table_library_core_operations():
    assert run('''
local t = table.pack("a", nil, "c")
local n = t.n
local u = {1,2,3}
table.insert(u, 2, 9)
local removed = table.remove(u, 3)
local dst = {}
table.move(u, 1, #u, 2, dst)
local a,b,c = table.unpack(u)
return n, t[1], t[2], t[3], removed, a,b,c, dst[2],dst[3],dst[4], table.concat({1,"x",3}, ":")
''') == (3, b'a', None, b'c', 2, 1, 9, 3, 1, 9, 3, b'1:x:3')


def test_table_sort_default_and_callback():
    assert run('''
local a = {4,1,3,2}
table.sort(a)
local b = {1,4,2,3}
table.sort(b, function(x,y) return x > y end)
return table.concat(a, ","), table.concat(b, ",")
''') == (b'1,2,3,4', b'4,3,2,1')


def test_math_core_and_integer_preservation():
    result = run('''
local i,f = math.modf(-3.25)
return math.abs(-5), math.floor(3.9), math.ceil(-3.9), i, f,
       math.type(1), math.type(1.5), math.tointeger(4.0),
       math.ult(-1, 0), math.max(1,5,3), math.min(1,5,3),
       math.frexp(8)
''')
    assert result[:11] == (5, 3, -3, -3, -0.25, b'integer', b'float', 4, False, 5, 1)
    assert result[11:] == (0.5, 4)


def test_math_randomseed_is_reproducible():
    assert run('''
math.randomseed(123, 456)
local a,b,c = math.random(0), math.random(), math.random(-10,10)
math.randomseed(123, 456)
return a == math.random(0), b == math.random(), c == math.random(-10,10)
''') == (True, True, True)


def test_string_byte_char_sub_rep_reverse_case():
    assert run('''
local a,b,c = string.byte("ABC", 1, 3)
return a,b,c,string.char(a,b,c),string.sub("abcdef",-3,-1),
       string.rep("ab",3,"-"),string.reverse("abc"),string.lower("ABC"),string.upper("abc")
''') == (65,66,67,b'ABC',b'def',b'ab-ab-ab',b'cba',b'abc',b'ABC')


def test_string_patterns_find_match_gmatch_and_gsub():
    assert run('''
local i,j,word = string.find("!! hello 123", "(%a+)")
local digits = string.match("id=42", "=(%d+)")
local words = {}
for w in string.gmatch("one two three", "%a+") do words[#words+1] = w end
local changed,n = string.gsub("a1 b22", "(%a)(%d+)", "%2%1")
return i,j,word,digits,table.concat(words,"|"),changed,n
''') == (4,8,b'hello',b'42',b'one|two|three',b'1a 22b',2)


def test_string_pattern_balanced_frontier_backreference_and_position_capture():
    assert run('''
local balanced = string.match("x(a(b)c)y", "%b()")
local word = string.match("!abc!", "%f[%a]%a+%f[%A]")
local repeated = string.match("abc abc", "(%a+)%s+%1")
local p1,p2 = string.match("xxaaay", "()a+()")
return balanced,word,repeated,p1,p2
''') == (b'(a(b)c)', b'abc', b'abc', 3, 6)


def test_string_pack_unpack_and_packsize():
    assert run('''
local packed = string.pack("<i4I2c3", -12345, 65530, "xy")
local a,b,c,nextpos = string.unpack("<i4I2c3", packed)
return a,b,c,nextpos,#packed,string.packsize("<i4I2c3")
''') == (-12345, 65530, b'xy\0', 10, 9, 9)


def test_string_format_common_specifiers_and_q():
    result = run('return string.format("%04d %.2f %s %q %%", 7, 2.5, "ok", "a\\nb")')
    expected = b'0007 2.50 ok "a' + b'\\' + b'\n' + b'b" %'
    assert result == expected


def test_utf8_char_codepoint_len_codes_and_offset():
    assert run('''
local s = utf8.char(65, 8364, 128578)
local a,b,c = utf8.codepoint(s,1,-1)
local positions = {}
for p,cp in utf8.codes(s) do positions[#positions+1] = p end
local p1,e1 = utf8.offset(s,2)
local p2,e2 = utf8.offset(s,0,p1)
return a,b,c,utf8.len(s),table.concat(positions,","),p1,e1,p2,e2
''') == (65,8364,128578,3,b'1,2,5',2,4,2,4)


def test_utf8_len_reports_first_invalid_byte():
    lua = LuaRuntime()
    lua.set('bad', b'a\xffb')
    assert lua.execute('local n,p=utf8.len(bad); return n,p') == (None, 2)
