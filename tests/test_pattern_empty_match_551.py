from luapyre import LuaRuntime


def test_gsub_empty_match_progression_matches_lua_533_plus_semantics():
    lua = LuaRuntime()
    assert lua.execute('return string.gsub("a b cd", " *", "-")') == (b"-a-b-c-d-", 5)
