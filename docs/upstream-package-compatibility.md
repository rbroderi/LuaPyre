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

## luatest

- Repository: `mblayman/luatest`
- Revision: `d063c547b31d4df1dce1b0679ccf964d53ff5c1e`
- Result: **5/5 selected core module tests pass unchanged**

The gate runs the three upstream `main.update_package_path` tests and both
upstream executor tests, including the passing and failing-test paths. Small
in-memory adapters implement the narrow contracts used from `luassert`, its
stub helper, and `pl.tablex`; they do not replace any luatest source under
test.

This proves that luatest's core executor can load and run tests on LuaPyre and
that its package-path helper behaves correctly. The complete command-line
runner is not yet a compatibility claim: discovery, configuration, output
capture, and coverage require its full LuaRocks dependency set plus filesystem
and temporary-file capabilities that the default sandbox does not expose.

## LuaCov

- Repository: `lunarmodules/luacov`
- Revision: `645a98468aad737de035a49a578c245bb3e555fb`
- Result: **24/24 line-scanner specs pass unchanged; collection is unsupported**

The entire upstream `spec/linescanner_spec.lua` file runs through a minimal
Busted-style `describe`/`it` adapter. This exercises LuaCov's real pure-Lua
source scanner, including long strings, comments, functions, labels, and
inline enable/disable directives.

Loading LuaCov's collector reaches `require("debug")`. Actual coverage
collection needs `debug.sethook` line events, `debug.getinfo`, exit/finalizer
behavior, and report-file I/O. LuaPyre does not currently expose those hooks,
so the runner records this as partial compatibility instead of treating a
no-op debug shim as coverage support.

## Are We Fast Yet

- Repository: `smarr/are-we-fast-yet`
- Revision: `74306fec151070fd07157cefeacf19e7e0bcdc89`
- Result: **11/11 selected Lua benchmarks pass their upstream result checks**

The unchanged upstream `harness.lua` runs Bounce, DeltaBlue, Json, List,
Mandelbrot, NBody, Permute, Queens, Sieve, Storage, and Towers with one measured
iteration and the upstream validation for each result. The host supplies only
the monotonic clock used by the harness. This is an application-style semantic
gate across object protocols, deep table graphs, recursion, parsing, numeric
loops, and allocation-heavy workloads; its timings are not used as CI
performance thresholds.
