from __future__ import annotations

import pytest

from luapyre import LuaRuntime


def test_output_sink_receives_exact_lua_formatted_bytes():
    output = []
    lua = LuaRuntime(output=output.append)
    lua.execute(r'''print("\255", 42, true, nil)''')
    assert output == [b"\xff\t42\ttrue\tnil\n"]


def test_warning_sink_and_warning_controls():
    warnings = []
    lua = LuaRuntime(warning=warnings.append)
    lua.execute('warn("a", "b"); warn("@off"); warn("hidden"); warn("@on"); warn("c")')
    assert warnings == [b"ab", b"c"]


def test_runtime_capabilities_reject_noncallables():
    with pytest.raises(TypeError, match="output sink"):
        LuaRuntime(output=42)
    with pytest.raises(TypeError, match="warning sink"):
        LuaRuntime(warning=42)
    with pytest.raises(TypeError, match="file loader"):
        LuaRuntime(file_loader=42)


def test_preload_require_and_loaded_cache():
    lua = LuaRuntime(output=lambda _data: None)
    lua.preload("answer", "return {value = 42}")
    result = lua.execute(
        """
        local first, where = require('answer')
        local second = require('answer')
        return first.value, where, first == second, require('math') == math
        """
    )
    assert result == (42, b":preload:", True, True)


def test_python_callable_can_be_preloaded():
    seen = []
    lua = LuaRuntime(output=lambda _data: None)

    def loader(name, loader_data):
        seen.append((name, loader_data))
        return {"value": 7}

    lua.preload("hostmod", loader)
    assert lua.execute("return require('hostmod').value") == 7
    assert seen == [(b"hostmod", b":preload:")]


def test_default_package_has_virtual_only_file_capability():
    lua = LuaRuntime(output=lambda _data: None)
    assert lua.execute(
        "return loadfile == nil, dofile == nil, package.searchers[1] ~= nil, package.searchers[2] == nil"
    ) == (False, False, True, False)
    assert lua.execute("return loadfile('/etc/passwd')")[:1] == (None,)


def test_file_loader_enables_loadfile_dofile_and_require_searcher():
    files = {
        "answer.lua": b"return 42",
        "nested/module.lua": b"return {value = 23}",
    }
    lua = LuaRuntime(file_loader=files.get, output=lambda _data: None)
    result = lua.execute(
        """
        local f = assert(loadfile('answer.lua'))
        local m, filename = require('nested.module')
        return f(), dofile('answer.lua'), m.value, filename
        """
    )
    assert result == (42, 42, 23, b"nested/module.lua")


def test_file_loader_can_be_added_and_removed_dynamically():
    lua = LuaRuntime(output=lambda _data: None)
    assert lua.get("loadfile") is not None
    lua.set_file_loader(lambda name: b"return 9" if name == "x.lua" else None)
    assert lua.execute("return dofile('x.lua'), package.searchers[2] ~= nil") == (9, True)
    lua.set_file_loader(None)
    assert lua.execute("return loadfile ~= nil, dofile ~= nil, package.searchers[2] ~= nil") == (
        True, True, True,
    )


def test_file_loader_is_not_installed_when_safe_stdlib_is_disabled():
    lua = LuaRuntime(safe_stdlib=False, file_loader=lambda _name: b"return 1")
    assert lua.get("package") is None
    assert lua.get("loadfile") is None
