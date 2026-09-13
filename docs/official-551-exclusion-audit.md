# Official Lua 5.5.1 exclusion audit

LuaPyre runs the checksum-pinned, unchanged upstream suite with `_U`, `_soft`,
`_port`, and `_nomsg` enabled.  Exclusion applies to a top-level driver, not to
the underlying language rule: portable rules are represented by an unchanged
upstream file where practical and otherwise by a bounded Python regression.

| Upstream file | Why the full file is excluded | Safe semantic disposition |
| --- | --- | --- |
| `all.lua` | Suite orchestrator invokes every native, process, and stress driver. | The harness performs the same manifest accounting and fresh-runtime isolation directly. |
| `api.lua` | Returns immediately without Lua's internal `T` C-test module. | Ordinary call, stack, value, and error rules are exercised by required upstream files and unit tests. |
| `big.lua` | Returns immediately in `_soft` mode; full mode deliberately allocates huge tables and strings. | Parser/register limits and bounded large-table behavior are unit-tested. |
| `code.lua` | Depends throughout on `T.listcode`, `T.listk`, and other internal compiler inspection APIs. | Runtime semantics are covered; PUC opcode-shape assertions are implementation-specific. |
| `cstack.lua` | Deliberately drives recursive patterns, callbacks, and coroutine chains to host stack exhaustion. | Protected stack recovery, recursive `xpcall` handlers, parser nesting, and coroutine error recovery have bounded regressions. |
| `gc.lua` | Mixes long allocation loops with internal collector controls and non-portable pacing assertions. | Weak tables, ephemerons, finalization/resurrection, mode changes, parameter extremes, automatic pacing, and remembered-set barriers are bounded tests. |
| `heavy.lua` | Purpose-built memory/string/compiler exhaustion workload. | Size, syntax-depth, local, return, register, and upvalue limits fail deterministically in bounded tests. |
| `main.lua` | Tests the standalone PUC executable, shell, environment, readline, and physical filesystem. | Source/BOM/load behavior and virtual `io`/`os` capabilities are tested without granting ambient host access. |
| `memerr.lua` | Requires allocator-failure injection and the internal C test API. | Lua error unwinding and quota failures are tested; deterministic host allocator failure is outside the sandbox contract. |
| `tracegc.lua` | Helper module for internal GC tracing and host stderr behavior. | Finalizer order, warnings, nested-collection deferral, and output capabilities are covered directly. |

`gengc.lua` was removed from this list: its substantial portable prefix runs to
completion without `T` and is now part of the required unchanged-upstream gate.

There are no remaining tracked-gap files. `coroutine.lua`, `db.lua`, and
`locals.lua` now run unchanged to completion and are mandatory release-gate
members. This includes weak coroutine-wrapper collection, exact line/count/
call/return hook behavior, stripped-chunk debug inspection, bounded traceback
formatting, and wrapped-coroutine close-error propagation.
