from __future__ import annotations

from luapyre import LuaRuntime


def test_print_uses_python_stdout_with_lua_formatting(capsys):
    lua = LuaRuntime()
    assert lua.execute('print("hello", 42, true, nil)') is None
    assert capsys.readouterr().out == "hello\t42\ttrue\tnil\n"


def test_print_without_arguments_emits_blank_line(capsys):
    LuaRuntime().execute("print()")
    assert capsys.readouterr().out == "\n"


def test_print_honors_lua_tostring_metamethod(capsys):
    LuaRuntime().execute('''
      local value = setmetatable({}, {
        __tostring = function() return "rendered" end
      })
      print(value)
    ''')
    assert capsys.readouterr().out == "rendered\n"


def test_print_is_absent_when_safe_stdlib_is_disabled():
    lua = LuaRuntime(safe_stdlib=False)
    assert lua.get("print") is None
