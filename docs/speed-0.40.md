# LuaPyre 0.40 corrective performance tranche

0.40 retains the general GC/accounting, scalar-entry, record-construction, and
typed-IR work, but withdraws the earlier Binary Trees and Spectral Norm headline
results. Those results came from three whole-function algorithm replacements:
a dense matrix product, a binary-record builder, and a binary-record counter.
All three compilers have been removed. Matrix and tree programs now pass through
the ordinary typed CFG compiler.

The retained and generalized paths are:

- def-use analysis for integer base cases, insensitive to local temporaries and
  reversed comparisons;
- field-count-independent fresh nil-record base cases, using value provenance
  rather than fixed instruction positions;
- arbitrary straight-line pure scalar expression DAG inlining;
- typed CFG lowering of nested numeric regions and reductions;
- generic compiled scalar calls and pooled materialized frames;
- fresh-record setters, exact allocation accounting, cached empty-table sizing,
  primitive barrier elision, and safe child ownership handling; and
- sparse integer-to-boolean regions, provisionally retained with unrelated
  visited/slot/occupancy tests.

`native_headroom.py` and `python_headroom.py` now report fast-path admissions per
workload. The native aggregate also lists every path's distinct workloads and
labels paths with fewer than three as `experimental`. See the permanent
[performance generality policy](performance-generality-policy.md).

## Acceptance coverage

Tests exercise three structurally different scalar DAGs, three nested
reductions, three record arities/layouts, and three unrelated sparse-set uses.
Metamorphic variants introduce temporaries, reverse comparisons, reorder record
fields, alter expression trees, and change record arity. Exact fuel boundaries,
invalid values, allocation identity, and accounting remain covered.

The old 0.40 JSON files are retained only as historical evidence and must not be
quoted as performance of the corrected implementation. New release numbers
must be generated after the general paths satisfy the three-program speed gate.
