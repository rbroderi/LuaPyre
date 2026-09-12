# LuaPyre 0.22 trace-shaped CFG compilation and OSR

LuaPyre 0.22 adds a profile-guided trace tier for certified scalar CFGs. Its
purpose is to optimize a hot loop during its first long-running invocation and
to turn repeated side exits into explicit successor profiles rather than
waiting for a future function call.

## Edge profiles and trace shape

The tiered VM records taken edges at typed `JMP`, `JMPIF`, `JMPIFNOT`, and
`JMPIFNIL` instructions. Once a target reaches the configured JIT threshold,
`TraceJIT` follows the hottest observed successor at each conditional and emits
a linear Python trace. A backedge to the entry closes the trace into one native
Python `while` loop.

Trace admission reuses `CFGValueIRCompiler`. Only certified scalar CFGs enter;
calls, cells, tables/metatables, varargs, unsupported arithmetic, child-closure
creation, and unproven operations fail closed and are negatively cached.

## Exact on-stack replacement

Branch handlers are synchronization points for typed main-thread chunks. After
the triggering Lua branch has consumed its ordinary instruction fuel, the VM
passes the existing `Frame` directly to the trace runner. The runner reads and
writes the same register list and begins at the already-selected target PC.
There is no synthetic frame, replayed prologue, reconstructed argument list, or
second Lua state.

Every traced instruction performs a preflight against the synchronized budget
before execution. If the budget cannot cover it, the runner stores that exact
PC and returns to Tier 0; the next interpreter tick raises at the same boundary
as interpreter-only execution.

## Side exits

Conditional traces guard the profiled successor. If execution selects the
other edge, the trace writes the alternate PC into the live frame, returns the
number of Lua instructions already consumed, and records the side-exit site.
The interpreter resumes without replaying the branch. Repeated alternate paths
naturally become hot OSR targets and may receive their own trace.

Trace exits are also exact at unsupported instructions, end-of-trace PCs, and
returns. A traced `RETURN` consumes its source instruction and then uses the
VM's ordinary `_return` frame/result protocol.

## Scope and overhead boundary

The edge/OSR handlers are installed only when the root chunk is certified with
`-- luapyre: typed`. Ordinary Lua retains the unchanged dispatch table. Lua
coroutines currently retain the established loop/function tiers; trace OSR is
limited to the main execution budget until coroutine-specific suspension tests
establish the same contract.

New `jit_stats` fields expose trace compilations, executions, side exits, and
OSR entries.

## Validation and performance

Permanent tests cover live-register OSR, cyclic trace formation, exact exit-PC
feedback, traced returns, fail-closed ordinary Lua/unsupported CFG behavior,
and JIT/interpreter agreement across every nearby fuel boundary.

On CPython 3.13.15, the permanent 30,000-iteration typed scalar `while` probe
measured 24.5 ms versus 111.1 ms on merged 0.21: approximately **4.5× faster**
on the same runner. The gain comes from entering the trace during the first hot loop
instead of interpreting the entire invocation.

## Next compiler work

1. stitch hot alternate traces directly instead of returning through Tier 0;
2. admit guarded table/call PIC facts into trace CFGs;
3. add coroutine-aware trace suspension and resumption;
4. add range predication and dominance-frontier PRE; and
5. lower the same trace/CFG contracts to a native backend.
