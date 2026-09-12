from lupa.lua55 import LuaRuntime as ReferenceLuaRuntime

from luapyre.binary_chunks import _decode_puc_chunk, _op, _a, _b, _c, _k


def test_show_vararg_return_layout():
    lua = ReferenceLuaRuntime(encoding=None)
    dump = lua.eval("function(src) return string.dump(assert(load(src)), true) end")
    blob = dump(b"local function f(...) return select('#',...), ... end; return f(1,nil,3)")
    root = _decode_puc_chunk(blob)
    child = root.children[0]
    decoded = [(i, _op(w), _a(w), _b(w), _c(w), _k(w), hex(w)) for i, w in enumerate(child.code)]
    raise AssertionError(decoded)
