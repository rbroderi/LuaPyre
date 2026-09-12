# CPython-specialization-aware code generation in 0.25

LuaPyre's JIT emits Python functions, so the shape of those functions controls
whether CPython can specialize them. CPython 3.13 and 3.14 adapt operations such
as binary arithmetic and constant-index list access after warmup. Its optional
copy-and-patch JIT consumes a lower tier of the same specializing interpreter;
LuaPyre does not require that optional JIT for correctness.

## Promoted registers

The dense loop and leaf tiers now rewrite constant `regs[n]` reads and writes to
real Python locals before `compile()`. Inputs are loaded once. Loops spill the
locals on every exit, and leaves spill before a sentinel deoptimization. Object
allocation sites spill before entering the VM's GC safe point so active Lua
registers remain visible to semantic collection.

Higher typed CFG, whole-function, value-IR, direct-call, structured-loop, and
virtual-frame tiers already used scalar locals. Lua tables, closures, cells,
frames, host functions, threads, protos, and multivalues already use slots, so
0.25 keeps their representation and adds a regression audit for it.

## Integer intervals

`range_analysis.py` adds backend-neutral signed-64 interval facts. Integer
parameters start at the complete signed-64 range. Constants are exact, copies
retain their interval, and typed add/subtract/multiply propagate bounds. An
operation is marked overflow-free only when both result bounds fit Lua's signed
64-bit domain.

Literal numeric loops seed the induction variable and constant limit/step at the
loop header. Loop-assigned values are killed first, which prevents the initial
accumulator value from being mistaken for a loop invariant. Facts are cleared
at control-flow merges; this deliberately gives up optimization instead of
making a path-sensitive assumption. Unproven operations retain the exact mask
and sign-fix sequence.

The fact is attached to typed instructions and value nodes and is consumed by
dense, AST, structured, whole-function, value-IR, CFG, direct-call, and virtual
frame code generation. Range-proven integer loop backedges also emit the integer
path directly, removing the repeated runtime int/float selection.

## Guards and audits

Guard failure remains an explicit return state (`DEOPT`, `_DEOPT_STATE`, or
`_FUNC_SUSPEND`). Python exceptions remain reserved for Lua runtime errors and
for materializing exact traceback state. This keeps successful hot paths free
of exception-based control flow.

`tests/test_cpython_specialization.py` warms a generated leaf and inspects
`dis.get_instructions(..., adaptive=True)`. It requires CPython's specialized
integer-add opcode, verifies that successful leaf execution has no register-list
stores, checks wrap elision and overflow preservation, and audits slots and
sentinel deoptimization. `benchmarks/cpython_specialization_ab.py` is the
same-runner release comparison for the affected dense-loop, typed-leaf, and
literal-range shapes.

CPython bytecode names are implementation details, so this audit is intentionally
version-gated to the supported CPython 3.13/3.14 matrix. See the official
[`dis` adaptive-bytecode documentation](https://docs.python.org/3/library/dis.html),
the [Python 3.13 experimental JIT notes](https://docs.python.org/3/whatsnew/3.13.html#an-experimental-just-in-time-jit-compiler),
and [PEP 659](https://peps.python.org/pep-0659/) for the underlying interpreter
model.
