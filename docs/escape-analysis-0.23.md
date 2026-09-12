# LuaPyre 0.23 escape analysis and allocation sinking

LuaPyre 0.23 adds a backend-neutral escape plan for statically resolved lexical
calls. The first consumer is the generated-Python CALL IR backend, which now
keeps eligible child-frame registers in scalar Python locals. A normal return
does not allocate a VM `Frame`.

Eligibility is deliberately exact. The child must be a fully typed,
non-vararg lexical leaf with no upvalues, nested closures, calls, tables, or
other heap effects. The admitted scalar CFG includes branches and numeric loops,
so this extends beyond the tiny straight-line calls already inlined by Value IR.
Unsupported calls continue through the real-frame backend.

## Materialization points

Virtual state becomes real state only where the interpreter must observe it:

- the normal stack-limit check still occurs before child execution;
- insufficient fuel reconstructs the child `Frame` with its exact registers and
  resume PC, then returns to Tier 0;
- a Lua runtime error reconstructs the child frame at the faulting PC so source
  diagnostics and unwinding retain the lexical call;
- argument checks occur in the same order as `_new_frame` and retain their
  existing diagnostics.

The source bytecode remains the fuel authority. Generated blocks preflight their
full bytecode cost; when the remaining allowance is smaller, materialization
happens before the block and the interpreter consumes the partial allowance one
instruction at a time.

## Virtual multiple results

An open-result `CALL` normally wraps its result tuple in `MultiValue`. Escape
analysis now proves when that wrapper is used only by `UNPACK` or `RETURNV`.
The backend then keeps the tuple in a promoted local and reads/returns its scalar
components directly. If the caller suspends while the open result is live, the
spill path allocates the ordinary `MultiValue` wrapper exactly once.

`JITStats` exposes `escape_plans`, `virtual_frame_elisions`,
`virtual_frame_materializations`, `virtual_multivalue_elisions`, and
`virtual_multivalue_materializations` for validation and profiling.

## Performance

On the same CPython runner, the focused `escape_analysis_ab.py` benchmark
measured the branch-call workload at 57.3 ms versus 72.7 ms on 0.22 (21% faster)
and the open-result workload at 61.4 ms versus 103.3 ms (41% faster). Each 0.23
sample reported 96,000 virtual-frame elisions; the open-result sample also
reported 96,000 `MultiValue` elisions.
