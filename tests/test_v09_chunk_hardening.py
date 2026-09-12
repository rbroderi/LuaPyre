from __future__ import annotations

from luapyre import LuaRuntime


def test_corrupt_native_debug_metadata_fails_as_load_error():
    lua = LuaRuntime()
    dumped = lua.execute('return string.dump(function() return 42 end, false)')
    # Damage the JSON trailer while leaving the validated VM payload intact.
    bad = dumped[:-1] + b"!"
    lua.set("bad", bad)
    assert lua.execute('local f,e=load(bad,nil,"b"); return f,type(e)') == (
        None,
        b"string",
    )
