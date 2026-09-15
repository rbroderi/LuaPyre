# Performance optimization generality policy

Every production fast path must be expressed as an instruction-, IR-, CFG-,
type-, alias-, range-, or effect-level transformation. A compiler must not
match an entire function with an `expected_ops` tuple and replace it with a
handwritten implementation of the recognized algorithm. Benchmark names,
function names, fixed field counts, and exact temporary-register allocation are
never admissible proof facts.

Before an optimization is accepted it must:

1. improve at least three structurally different programs that exercise the
   same reusable compiler fact, by the preregistered A/B threshold (2% by
   default, so timer noise does not count as acceleration);
2. retain admission and correctness under applicable semantic-preserving
   mutations, including introduced temporaries, reversed comparisons,
   reordered independent statements, changed record arity, and equivalent
   expression trees;
3. preserve fuel, errors, metatables, debug hooks, stack limits, GC behavior,
   allocation identity, and deoptimization boundaries; and
4. publish static fast-path admission counts for every benchmark workload.

Benchmark aggregation classifies a path admitted by fewer than three distinct
workloads as `experimental`. Experimental paths may be measured and retained
behind ordinary semantic guards, but they are not evidence for a general
performance claim and must not justify release headline results.

The 0.38 pure-base prefix remains because it skips only a proven local arm; its
matching is now def-use based for scalar comparisons and fresh nil records.
Further work should move the remaining table-prefix proof fully into CFG IR.
The 0.39 sparse integer/boolean representation remains provisional and is
covered by sieve, visited-set, slot-map, and occupancy-map shapes plus admission
telemetry.
