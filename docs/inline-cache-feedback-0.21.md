# LuaPyre 0.21 inline caches and feedback

LuaPyre 0.21 adds adaptive call/table inline caches to the tiered VM and turns
guard exits into durable profile feedback. Tier 0 remains the semantic oracle;
the caches only bypass dispatch after proving the same raw operation at the
same bytecode site.

## Bounded call and table PICs

Every `CALL`, `CALLV`, `TAILCALL`, and `TAILCALLV` site records executable
identity: `Proto` identity for Lua closures and object identity for host
functions. Every raw, metatable-free `GETTABLE` and `SETTABLE` site records the
receiver identity plus Lua's normalized key token. Sites progress through four
states:

1. empty;
2. monomorphic;
3. polymorphic (up to four entries); and
4. megamorphic.

The megamorphic state is permanent for the VM lifetime and uses ordinary exact
dispatch. It prevents an unstable site from continually replacing cache
entries. Metatable-sensitive access, nil/NaN keys, and non-table receivers
always use the interpreter operation.

Active frames bind directly to a shared, per-`Proto` call-site array. After the
first access, a monomorphic call therefore requires no tuple construction or
site-dictionary lookup: the VM performs one indexed load and one callable
identity guard. A cached compiled leaf also bypasses the redundant global leaf
cache lookup.

Read entries carry `LuaTable.version` and the cached result. Any raw mutation
invalidates the value entry before reuse. Writes cache only the proven raw
receiver/key route and continue through `rawset`, preserving array/hash
migration, nil deletion, NaN rejection, and version updates.

## Deoptimization feedback

Loop side exits are recorded by `(Proto, pc, reason)`. A region that reaches the
same failing site three times is retired to Tier 0, avoiding compile/deopt
thrashing. The first reason vocabulary distinguishes entry guards from other
side exits and is deliberately extensible as later IR tiers expose richer guard
identities.

`LuaRuntime.jit_feedback` returns a stable snapshot containing site counts,
cache-state counts, aggregate hits/misses/invalidations, and deoptimization
reason totals. Existing low-overhead counters remain available through
`LuaRuntime.jit_stats` and now include call/table IC activity, megamorphic
transitions, invalidations, and retired regions.

## Correctness boundary

Inline caches do not change Lua fuel, frame creation, stack limits, tail-call
behavior, coroutine suspension, metamethod chains, error provenance, or table
semantics. Interpreter-only runtimes allocate no cache state and expose
`jit_feedback` as `None`.

## Validation

Permanent tests cover monomorphic hits, mutation invalidation without stale
reads, bounded polymorphism and megamorphic transition, reason aggregation and
retirement thresholds, and the cache-free interpreter surface. The complete
Lua 5.5 reference/differential suite remains the release gate.

`benchmarks/inline_cache_ab.py` provides repeatable same-runner probes for
monomorphic calls, stable table reads, and a version-invalidated table read.

## Next compiler work

1. feed stable PIC targets into typed/CALL IR cloning and guarded direct calls;
2. widen numeric/table specialization from observed deopt reason histories;
3. dominance-frontier/PRE and broader global value numbering;
4. loop predication/range analysis using induction/limit metadata;
5. hot side-exit compilation and CFG stitching; and
6. CFG OSR, escape analysis, intrinsics, then native lowering from the same IR.
