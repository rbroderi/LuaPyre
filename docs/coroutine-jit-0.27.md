# LuaPyre 0.27: resumable compiled coroutines

LuaPyre 0.27 adds a generated-Python state-machine tier for hot coroutine
bodies. The compiler groups eligible bytecode into basic blocks and executes
those blocks directly between suspension points instead of dispatching every
instruction through Tier 0.

## State and suspension contract

The ordinary Lua `Frame` remains the authoritative activation record. The
compiled runner reads and writes its register array and commits `frame.pc`
before every yield, return, error-capable helper, or interpreter fallback.
Consequently:

- resume arguments land in the ordinary CALL result registers;
- the Lua GC sees the same roots as interpreted execution;
- debug hooks can force the exact interpreter path without reconstruction;
- unsupported or mutated call targets deopt before observable work;
- stack traces and coroutine closing continue to use normal Lua frames; and
- bytecode fuel is charged for the exact instructions completed by a block.

CALL admission is deliberately narrow. Generated code executes a suspension
only when the runtime value is the exact sandbox-owned `coroutine.yield`
function. Replacing that table field, invoking another host function, entering
a protected/closing frame, or enabling a debug hook returns control to Tier 0.

The first opcode boundary covers scalar arithmetic and comparison, dense/raw
table reads without metatables, upvalue reads, numeric loops, branches, direct
yield calls, and returns. Other shapes are negative-cached and interpreted.

## Performance

On the committed coroutine benchmark (2,000 loop iterations and 200 yields),
steady-state execution on the profiling host changed from approximately
19.6 ms to 9.2 ms. The compiled coroutine is about 2.1x faster than the prior
JIT path and 1.8x faster than LuaPyre's interpreter on that host.

The runtime exposes `coroutine_compiles`, `coroutine_executions`,
`coroutine_yields`, `coroutine_instructions`, and `coroutine_deopts` in its JIT
statistics so the tier remains observable without adding counters inside each
generated instruction.

