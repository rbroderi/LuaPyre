# Branch, table, and Python boundary overhead (0.31)

LuaPyre 0.31 follows the remaining profiles after the 0.30 fast paths. It keeps
the Python-only implementation, ordinary Lua integer wrapping, and explicit
`integer`/`LuaInt` contracts.

## Changes

1. **Batch fuel charges by branch path.** A structured diamond loop charges
   its prefix, branch, selected arm, jump, and numeric backedge once per
   iteration. Every guard side exit still reports the exact cost before its
   instruction and spills registers at that instruction's PC. Short budgets
   resume the interpreter, including when the two arms have different costs.
2. **Use types established on the current path.** Numeric equality and
   ordering use direct Python comparisons when preceding constants, arithmetic,
   or guards prove their representations. Checked integer modulo proves its
   result type. Unknown values retain representation checks and Lua equality;
   table metamethod cases still deopt. Neither profiles nor a previous iteration
   establish these facts. Boolean/numeric equality remains distinct.
3. **Read invariant constant table fields once per runner entry.** A structured
   loop may hoist a plain-table constant read when the table register is stable
   and the body has no table writes or calls. This includes dense integer keys
   and hash keys. Guards and instruction charges stay in the loop. Reads are
   refreshed on every entry; any table write disables hoisting for all tables
   in the region, so aliases cannot produce stale values. Metatables retain
   their existing interpreter fallback.
4. **Streamline scalar conversion.** Built-in scalar return annotations bypass
   `typing.get_origin/get_args`. Python integers already in signed-64 range
   bypass wrapping; larger untyped integers still wrap. Explicit typed integer
   inputs retain their range check without routing through result conversion.
   Union/container conversion, strict bool/int distinctions, and errors remain
   intact.

Repeated-call coverage also exposed a pre-existing table PIC issue: a positive
integral float key could hit a hash-only cache even though its value lived in
the integer array. Those keys now follow the integer lookup path, including
when filling a gap migrates a key from hash to array storage.

## Measurements

CPython 3.14.7 on Linux x86-64; the same benchmark script and interpreter were
used for both source checkouts. Each process used 7 warmups and 31 timed samples
with Python GC disabled only during timing. Processes shared one CPU affinity
and `PYTHONHASHSEED=0`. Runs used A/B/B/A order; values below are the median of
each revision's two run medians. These are workload measurements, not an
application-wide speed claim.

Baseline: the merged 0.30 code including its correctness follow-ups
(`364d5c2`). Raw run metadata and medians are in
[`speed_031_ab.json`](../benchmarks/results/speed_031_ab.json).

| Workload | 0.30 | 0.31 | Change in elapsed time |
| --- | ---: | ---: | ---: |
| Diamond branch loop | 14.113 ms | 6.588 ms | 53.3% less |
| Constant hash field | 3.902 ms | 2.108 ms | 46.0% less |
| Constant array field | 2.751 ms | 2.064 ms | 25.0% less |
| 1,000 Python calls, `int` result | 4.258 ms | 2.848 ms | 33.1% less |
| 1,000 Python calls, inferred result | 3.861 ms | 2.798 ms | 27.5% less |
| 1,000 Python calls, union result | 5.175 ms | 3.774 ms | 27.1% less |
| Inlined typed-call control | 0.699 ms | 0.689 ms | 1.5% less |
| Recursive-call control | 17.685 ms | 18.173 ms | 2.8% more |

No recursive-call improvement is claimed. Individual recursion runs ranged
from 17.508–17.862 ms on 0.30 and 17.426–18.920 ms on 0.31; the ranges overlap.
The typed-call control also remains essentially unchanged. These controls are
included to keep the comparison visible beyond the targeted wins.

Run the same script against each checkout, from the 0.31 directory:

```bash
PYTHONPATH=/path/to/030/src python benchmarks/speed_031_ab.py \
  --label 0.30 --warmups 7 --repeats 31 --json /tmp/speed030.json
PYTHONPATH=/path/to/031/src python benchmarks/speed_031_ab.py \
  --label 0.31 --warmups 7 --repeats 31 --json /tmp/speed031.json
```

## Validation

- Full regression suite passed on CPython 3.13.15 and 3.14.7, followed by the
  final 28-case 0.31 regression suite on both versions.
- All 24 required official Lua 5.5.1 baseline probes passed.
- Penlight 23/23, luatest 5/5, LuaCov portable scanner specs 24/24, and Are We
  Fast Yet 11/11 passed. The existing LuaCov collection and native lua-cjson
  exclusions remain reported by the upstream runner.
- New cases compare interpreter and JIT results/errors over small fuel
  budgets, unequal branch paths, ascending/descending/float loops, and modulo
  failures in the prefix and either arm. They also cover mixed primitive
  comparisons, aliased table writes, field changes between calls, metatable
  fallback, sparse-to-dense key migration, typed integer bounds, and untyped
  wrapping.

The next large untouched cost remains recursive compiled-call/frame overhead.
This tranche does not expand the admitted opcode set or change debug-hook,
coroutine, or sandbox capabilities.
