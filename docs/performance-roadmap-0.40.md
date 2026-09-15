# Performance roadmap: corrected 0.40

## Status

The three whole-algorithm compilers have been removed. The production compiler
contains no full-function `expected_ops` signature or handwritten substitute
for matrix multiplication or binary-tree construction/traversal.

## General IR path

1. **Pure scalar expression DAGs.** Inline bounded, side-effect-free typed
   callees from their instruction graph. Register copies and equivalent
   temporary layouts do not define the optimization.
2. **Nested reductions.** Lower typed nested loop CFGs through the ordinary IR
   region compiler. Continue replacing state dispatch with Python control flow
   only from loop, dominance, type, range, and effect facts.
3. **Record layouts.** Treat fresh constructor writes using source-proven unique
   keys and arbitrary layout size. Base-case analysis follows register value
   provenance and accepts reordered independent loads and aliases.
4. **Recursive calls.** Reduce scalar entry, frame reset, pooling, and safe base
   arms generically. Future non-tail frame elimination must be derived from call
   and continuation IR; binary-tree shape is not a proof.
5. **Sparse sets.** Keep 0.39 provisionally, with unrelated sieve, visited-set,
   slot-map, and occupancy-map admission coverage.

## Release gates

- Each claimed optimization must accelerate at least three structurally
  different programs in isolated A/B measurements.
- Metamorphic source mutations must retain admission and results.
- Every benchmark report includes per-workload admission counts.
- Any path admitted by only one benchmark is labeled experimental.
- The full semantic suite and exact fuel comparisons remain mandatory.

The detailed permanent rules are in
[performance-generality-policy.md](performance-generality-policy.md).
