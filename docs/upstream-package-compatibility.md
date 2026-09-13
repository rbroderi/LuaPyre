# Upstream package compatibility

LuaPyre runs pinned, unchanged tests from real Lua packages in addition to its
Lua 5.5 conformance suite. Run the current package check with:

```bash
python tools/upstream_compat.py
```

The script clones exact revisions, executes upstream Lua source through
`LuaRuntime`, and fails when a test in the supported profile fails.

## Penlight

- Repository: `lunarmodules/Penlight`
- Revision: `dfa483dddc0d751be1047667f4580dd90319169c`
- Result: **23/23 portable upstream tests pass unchanged**

The passing set covers Penlight's vector and class implementations,
comprehensions, configuration parsing, data transforms, functional helpers,
lexer, lists, maps, ordered maps, sequences, SIP matching, string IO, table
utilities, templates, and URL handling.

Penlight's old suite contains 37 test files. Fourteen are outside the portable
profile because they explicitly use unrestricted host services such as
`io.open`, `os.execute`, process exit, LuaFileSystem directory mutation, date
and locale access, or the full `debug` API. The compatibility runner supplies
small output, timing, stack-inspection, and LuaFileSystem placeholders needed
by Penlight's own assertion helper, but it does not grant filesystem or process
access to package code. Those exclusions preserve LuaPyre's sandbox contract.

The upstream run exposed and now guards six ordinary-Lua compatibility bugs:

- `global` remains an ordinary identifier outside LuaPyre's typed syntax;
- a table-valued `__index`/`__newindex` is indexed even when it has `__call`;
- parentheses collapse calls and varargs to one result;
- simultaneous assignment snapshots aliased locals before storing;
- Python implementations of Lua standard functions ignore surplus arguments
  when their native Lua counterparts do.
- package scripts may retain their Unix shebang line.

## lua-cjson

- Repository: `mpx/lua-cjson`
- Revision: `718f27293a981fb5e9e662e9aec0b7cf78317da6`
- Result: **unsupported: native Lua C module**

The unchanged upstream test starts with `require "cjson"` and
`require "cjson.safe"`. Those names are implemented by lua-cjson's compiled C
module, not by its Lua helper files. LuaPyre deliberately exposes no native
module searcher and no `lua_State` C ABI, so the actual package cannot load.
The compatibility check verifies and reports that boundary rather than
substituting a Python JSON implementation and calling the result compatible.

Supporting lua-cjson would require a defined native-extension bridge or a
separate API-compatible Lua/Python module. Only the first option would make the
real lua-cjson package and its own tests a valid compatibility result.
