# LuaPyre

LuaPyre is a clean-slate Lua runtime written in Python. It targets Lua 5.5.1 semantics, a sandbox-first embedding model, and optional gradual type annotations that feed the optimizer without creating a second runtime.

**Python 3.13+**.

LuaPyre is still pre-alpha and is not yet a complete Lua 5.5.1 implementation. The runtime is deliberately built around a register VM and explicit Lua frames so later quickening and JIT work can specialize stable semantics rather than replace an AST interpreter.

## Language model

Plain Lua is dynamic. Optional annotations add static guarantees and optimization facts:

```lua
local dynamic = get_value()       -- Any
local count: integer = 10

local function add(a: integer, b: integer): integer
    return a + b
end
```

Missing annotations mean `Any`. Typed and untyped code use the same parser, compiler, register bytecode, and VM.

Lua 5.5 declarations and LuaPyre annotations can be combined:

```lua
local limit: integer <const> = 10
global answer: integer
answer = 42
```

## Python embedding

```python
from luapyre import LuaRuntime

lua = LuaRuntime()
lua.expose("double_from_python", lambda n: n * 2)

result = lua.execute("""
local function add(a: integer, b: integer): integer
    return a + b
end

local x = double_from_python(10)
return add(x, 22)
""")

assert result == 42
```

Host capabilities are explicit. The default runtime does not expose filesystem, networking, process execution, Python import/eval, package loading, `io`, `os`, `debug`, or Python object introspection.

## 0.8 binary chunks

The 0.8 tranche adds binary-function serialization and PUC-Lua 5.5 binary input without making PUC bytecode the internal execution format:

- `string.dump` serializes LuaPyre closures into a bounded, non-pickle LuaPyre-native binary format
- `load` accepts LuaPyre-native binary chunks through ordinary `"b"`/`"bt"` mode handling
- loaded native chunks receive fresh root upvalues, with the first upvalue initialized from the supplied `load(..., env)` environment
- reader-function loading supports binary chunks assembled from multiple pieces, with a bounded total chunk size
- PUC-Lua 5.5 binary chunks are validated and translated into LuaPyre register bytecode before execution
- the PUC reader validates version/format sentinels, platform sizes, integer/number representations, string references, prototype nesting, instruction counts, constant counts, and truncation
- PUC compiler patterns such as mandatory `NEWTABLE`/`EXTRAARG`, tail-call/dead-return pairs, `LFALSESKIP`/`LOADTRUE`, numeric and generic `for`, varargs, shared closure upvalues, metamethod companion instructions, and `<close>` state are normalized or lowered explicitly
- compatibility opcodes exist only where PUC frame layout differs materially from LuaPyre's internal bytecode; ordinary operations lower back into the existing VM instruction set
- the Lua-aware GC liveness pass understands the PUC compatibility operations, so translated loop state, varargs, and pending to-be-closed registers remain semantic roots when required
- malformed or unsupported chunks fail as Lua `load` errors instead of being trusted as executable host data

LuaPyre intentionally keeps the two formats separate. `string.dump` does **not** claim to emit an official PUC-Lua chunk; it emits LuaPyre's own serialization of LuaPyre prototypes. PUC-Lua chunks are an interoperability input path that is decoded and translated into LuaPyre bytecode before execution. This preserves the custom VM/optimizer/JIT architecture while still allowing compiled Lua 5.5 code to enter through `load`.

The PUC compatibility suite differentially executes compiler-generated binary chunks against `lupa.lua55`, covering arithmetic and comparison expressions, tables, branches, numeric and generic loops, closures/upvalues, varargs and multiple results, metamethod fallback, tail calls, `<close>`, and GC-sensitive liveness. Exact Lua 5.5.1 remains the release-level format authority.

## 0.7 safe standard libraries

The 0.7 tranche fills out the deterministic, in-memory part of Lua 5.5's standard library while keeping the default runtime sandboxed:

- expanded base library with `tostring`/`__tostring`, `tonumber`, `select`, `pcall`, `xpcall`, `warn`, and text-mode `load`
- `pairs` honors `__pairs` and Lua 5.5's four-result iterator protocol
- `ipairs` uses ordinary indexing, including `__index`
- string values use the standard string metatable, so method syntax such as `("text"):upper()` works
- `table.concat`, `table.create`, `table.insert`, `table.move`, `table.pack`, `table.remove`, `table.sort`, and `table.unpack`
- the safe `math` library, including integer-aware helpers and Lua 5.5-compatible explicit-seed xoshiro256** random sequences
- the safe `string` library, including byte-string operations, Lua pattern matching, replacement callbacks/tables, formatting, and `pack`/`packsize`/`unpack`
- native Lua-pattern support for captures, sets/classes, greedy and minimal repetition, balanced matches, frontier patterns, backreferences, and position captures
- Lua 5.5 `%q` literal serialization, including exact control-character/newline treatment, minimum-integer handling, and hexadecimal floating-point literals
- the `utf8` library with `char`, `charpattern`, `codes`, `codepoint`, `len`, and Lua 5.5's two-result `offset`
- synchronous Lua callbacks from library functions reuse the existing shared opcode handlers and remain visible to the Lua-aware GC root tracer
- `load(..., env)` correctly initializes the loaded chunk's lexical `_ENV`

The default sandbox still deliberately omits `io`, `os`, `package`, and `debug`, along with file loaders and unrestricted host output. Binary loading in 0.8 remains in-memory and does not add filesystem access or other ambient host capabilities.

Standard-library callbacks are currently synchronous continuation boundaries. Calling Lua from `__pairs`, `__tostring`, `table.sort`, `pcall`/`xpcall`, pattern replacements, and similar library paths is supported, but yielding through one of those native-library callback boundaries is rejected. Full C/API-style yieldable continuations remain a later compatibility tranche rather than being approximated unsafely.

## 0.6 weak tables and finalization

The 0.6 tranche adds the GC-observable semantics needed by Lua programs while leaving physical memory ownership to Python:

- Lua-aware tracing from globals, active Lua frames, suspended coroutine stacks, closures/upvalues, varargs, and pending `<close>` state
- bytecode liveness analysis for GC roots, so stale physical VM register slots do not keep logically dead Lua objects alive
- weak-value tables with `__mode = "v"`
- weak-key tables with `__mode = "k"`
- all-weak tables with `__mode = "kv"`
- Lua-compatible treatment of strings and other non-object values in weak tables
- ephemeron semantics for weak-key/strong-value tables, including fixed-point convergence
- two-phase weak-table processing around finalization, including the different resurrection rules for weak keys and weak values
- table `__gc` finalizers when the metatable already contains `__gc` at `setmetatable` time
- reverse finalization order for objects collected in the same cycle
- resurrection and explicit re-marking for later finalization
- non-yieldable finalizers; attempts to collect recursively from a finalizer are rejected
- finalizer errors become warnings instead of propagating as ordinary Lua errors
- `collectgarbage` support for `collect`, `stop`, `restart`, `isrunning`, `count`, `step`, `incremental`, `generational`, and `param`
- `LuaRuntime.collect()` for embedders that want an explicit full Lua-observable collection cycle

LuaPyre intentionally uses its own Lua reachability model instead of Python weak references or refcounts. This is required for ephemerons, resurrection, finalizer ordering, and suspended Lua stacks to match Lua semantics even though Python remains responsible for reclaiming the underlying Python objects.

In 0.6, collection is explicit and deterministic at `collectgarbage`/host collection safe points. `collectgarbage("step")` completes a full observable cycle, `count` is an approximate Lua-reachable-memory estimate, and incremental/generational mode parameters are represented by the compatibility control surface; automatic byte-debt scheduling of Lua's incremental/generational collectors is not yet modeled.

## 0.5 coroutine core

The 0.5 tranche adds persistent Lua threads on top of the explicit-frame VM:

- `LuaThread` values with Lua `thread` type identity
- `coroutine.create`
- `coroutine.resume`
- `coroutine.yield`, including yields from nested Lua calls
- resume arguments becoming the return values of the suspended `coroutine.yield(...)` expression
- correct handling of `return coroutine.yield(...)` in tail position
- `coroutine.status` with `running`, `suspended`, `normal`, and `dead`
- `coroutine.running`, including main-thread identification
- `coroutine.isyieldable`
- `coroutine.wrap`
- `coroutine.close` for suspended and errored threads
- self-closing running coroutines, where `coroutine.close()` does not return to the coroutine body
- preservation of an errored coroutine's Lua stack until it is explicitly closed
- integration with 0.4 `<close>` state so pending `__close` methods run when a suspended or errored coroutine is closed
- the original Lua error object is supplied to pending `__close` methods during coroutine error cleanup

Coroutine stacks own the same Lua `Frame` objects used by the register VM, so registers, closures, upvalues, program counters, pending close state, and nested Lua calls survive suspension directly. Ordinary main-chunk execution continues through the established VM path; coroutine execution reuses the same opcode and semantic helpers with a persistent per-thread frame list.

The 0.4 tranche already provides `goto`/labels, local `<const>`/`<close>`, Lua 5.5 global declarations, strict-global behavior, and unified scope/error unwinding. The 0.3 tranche provides metatable/metamethod dispatch, additional control flow, anonymous functions, method syntax, generic iteration, and proper tail calls. Earlier semantic-core work includes lexical closures/upvalues, multiple returns, Lua 5.5 named varargs, lexical `_ENV`, split Lua tables, byte strings, signed 64-bit integer behavior, optional type guards, explicit host capabilities, and fuel/frame limits.

## Compatibility testing

CI runs on Python 3.13 and 3.14 and installs Lupa 2.8+, then explicitly imports `lupa.lua55`. This gives the differential suite an in-process PUC-Lua 5.5 oracle without compiling or launching a separate Lua executable. CI verifies that the selected backend reports `_VERSION == "Lua 5.5"` before running tests.

Differential coverage now includes safe standard libraries, Lua patterns, packing/unpacking, explicit-seed random output, UTF-8 iteration/offsets, `__pairs`, protected calls, text loading environments, weak keys/values, ephemerons, finalizer ordering, resurrection interactions, weak-table behavior across finalization, and execution of compiler-generated PUC-Lua 5.5 binary chunks through LuaPyre's translator. For exact micro-release conformance, LuaPyre targets Lua 5.5.1 and the official 5.5.1 tests/reference implementation remain the final authority. Lupa is the fast per-commit differential oracle; release-level conformance will additionally be checked against the exact 5.5.1 distribution.

The full official Lua test suite is **not** expected to pass yet. Major remaining work includes automatic incremental/generational GC pacing, detailed source-location/error/debug compatibility, yieldable native-library/C-API continuations, userdata/C-API GC behavior outside the sandbox value model, and substantially broader coverage of the official Lua 5.5.1 suite. The unsafe host-facing libraries remain intentionally absent from the default sandbox rather than being treated as missing semantic work.

## Performance roadmap

Correct semantics come first. Once the runtime is broad enough for the Lua 5.5.1 test suite, the performance path is:

1. adaptive quickening and inline caches
2. table/global/call specialization
3. generated superinstructions for common opcode sequences
4. hot-function compilation to Python code using Python locals for Lua registers
5. deoptimization back to the VM when specialization assumptions fail

Typed code can skip many dynamic guards because its annotations survive into compiler optimization metadata.
