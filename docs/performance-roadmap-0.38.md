# Performance roadmap: 0.38

## Goal

Implement the next Python/AST-only tranche from the 0.37 roadmap in this
order: pure base-case entry, scalar internal calls, dense-region proofs, and
record-construction improvements. Direct CPython bytecode generation and
native repository runtime code remain out of scope.

## Implemented path

1. Recognize a narrow leading typed-integer comparison whose taken arm returns
   only its argument or an integer constant. Charge its exact instruction cost
   and return the scalar without allocating a child `Frame`. Stack, type, fuel,
   debug-hook, and fallback behavior stay live.
2. Give fixed one/two-argument, one-result materialized compiled calls a scalar
   entry, avoiding the parent's argument tuple and result-sequence extraction.
   Reuse the existing real-frame pool for non-base calls and retain the generic
   tuple path for unsupported shapes and suspension.
3. Prove stable table aliases, positive dense index ranges, and primitive
   writes for a straight-line numeric-loop region. Guard table identity,
   metatable absence, and dense extent once, bind the backing array once, then
   emit direct array operations. Reject calls, holes, growth, nil/collectable
   writes, unknown aliases, and unproved bounds.
4. Preserve constant-key facts across whole-function call continuations. Mark
   unique literal constructor keys as proven fresh, lazily allocate iteration
   metadata, and use a fresh pre-hashed setter that retains version increments,
   GC barriers, and allocation accounting.

## Acceptance and rejected forms

Every change uses generated Python source/AST and CPython's own adaptive
specialization. The reusable `benchmarks/speed_038_ab.py` harness runs the same
checked workload against baseline and candidate checkouts in three paired
processes per CPython version, with alternating Lua/Python timing order, seven
warmups, and 31 samples.

Scalar recursion is admitted only when at least two recursive call sites make
base-case frame elision frequent enough to amortize entry selection; linear
recursion stayed on the prior path. Read-only dense loops keep their established
checked-array lowering because one-time region binding did not beat it. A
broader modulo-heavy nested-region experiment was also removed after regressing
table mix. These exclusions are part of the cost model, not semantic limits.

See [the 0.38 performance record](speed-0.38.md) for measurements and
validation.
