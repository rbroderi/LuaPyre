# LuaPyre 0.11 embedding capabilities

LuaPyre remains sandbox-first, but 0.11 separates safe Lua library semantics from ambient host access more cleanly.

## Output and warnings

`print` is part of the normal safe base environment. Values are converted through Lua's `tostring` semantics (including `__tostring`), separated by tabs, terminated with a newline, and then delivered to a byte-oriented output sink.

```python
from luapyre import LuaRuntime

captured = []
lua = LuaRuntime(output=captured.append)
lua.execute('print("hello", 42)')
assert captured == [b"hello\t42\n"]
```

With no output sink supplied, the default sink decodes with UTF-8 replacement and delegates to Python `print`. A custom sink receives the exact already-formatted Lua bytes, so embedders can preserve arbitrary byte strings.

`warn` uses the same model through a separate warning sink. The Lua 5.5 `@off` and `@on` controls are preserved.

```python
warnings = []
lua = LuaRuntime(warning=warnings.append)
lua.execute('warn("hello")')
assert warnings == [b"hello"]
```

Sinks can also be changed after construction with `set_output_sink()` and `set_warning_sink()`.

## Safe modules

The default safe runtime exposes `package` and `require`, but only an in-memory preload searcher is active. It does not search the host filesystem or load native libraries.

```python
lua = LuaRuntime()
lua.preload("answer", "return { value = 42 }")
assert lua.execute("return require('answer').value") == 42
```

`LuaRuntime.preload()` accepts Lua source text/bytes, an existing LuaPyre closure/host function, or a Python callable. `package.loaded` caches module results with Lua 5.5-style `require` behavior. The built-in safe libraries are already represented in `package.loaded`.

The default sandbox deliberately does not expose `package.loadlib`, native/C searchers, host environment-derived paths, or filesystem loading. `package.cpath` is empty.

## Explicit file loading capability

`loadfile`, `dofile`, and the Lua-file `require` searcher are absent until the host supplies a file loader:

```python
files = {
    "answer.lua": b"return 42",
    "game/vector.lua": b"return { x = 1, y = 2 }",
}

lua = LuaRuntime(file_loader=files.get)
assert lua.execute("return dofile('answer.lua')") == 42
assert lua.execute("return require('game.vector').x") == 1
```

A file loader receives a logical UTF-8 name and may return `bytes`, `str`, or `None`. LuaPyre does not interpret the name as a Python filesystem path. The embedding application decides whether it represents a real file, virtual filesystem entry, package resource, database row, archive member, or another read-only source.

The capability can be added or removed dynamically:

```python
lua.set_file_loader(files.get)
lua.set_file_loader(None)
```

Removing it removes `loadfile`, `dofile`, and the file searcher again. In-memory `package.preload` and `require` remain available.

## Security boundary

A default `LuaRuntime()` still has no ambient filesystem, networking, process execution, Python import/eval, `io`, `os`, `debug`, native library loading, or Python object introspection. `print`, `warn`, and preload-only modules are safe library facilities. Host filesystem access exists only when an embedding application explicitly installs `file_loader`.

The official Lua 5.5.1 conformance harness now uses these same production APIs. It supplies a suite-rooted read-only file loader and a captured output sink instead of replacing `package`, `require`, `loadfile`, `dofile`, or `print` with harness-specific Lua implementations.
