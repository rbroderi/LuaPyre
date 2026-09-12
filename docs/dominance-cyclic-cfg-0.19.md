# LuaPyre 0.19 dominance and cyclic CFG Value IR

LuaPyre 0.19 extends the 0.18 CFG Value IR from forward DAGs to **reducible typed scalar cycles** and makes dominance an explicit optimization proof.

The certified typed-function pipeline is now:

```text
-- luapyre: typed source
        ↓
LuaPyre bytecode / source-certified scalar types
        ↓
TypedIRPlan
        ↓
CFGValueIRPlan
  basic blocks
  predecessor/successor graph
  liveness
  dominators
  natural backedges
  scalar value identities
  branch and loop phi values
        ↓
dominance-qualified folding / CSE
        ↓
simple diamond       → Python if/else
simple natural loop  → Python while
other reducible CFG  → predecessor-tracked state machine
        ↓
exact block-entry live-state side exit to Tier 0
```

Python AST remains a backend. Dominance, loop structure, value identity, phi inputs, liveness, and exact side-exit state are represented before Python lowering so a later native backend can consume the same correctness contract.

## Dominators

0.19 computes classic block dominators over the reachable CFG. Block 0 dominates only itself initially; every other block starts with the universal set and iterates to the intersection of predecessor dominators plus itself.

Expression value numbering now retains a computed expression across a block boundary only when the expression's defining block dominates the block currently being compiled. Constants, literals, and arguments remain globally reusable because they are immutable roots.

This is deliberately stricter than semantic equivalence. Two sibling branches may independently compute `a + b`, but neither branch dominates the other or the join. 0.19 therefore does **not** reuse either sibling expression after the join. A later PRE/GVN tranche may prove such equivalence explicitly; ordinary CSE does not guess it.

## Backedges and reducibility

An edge `latch -> header` is a natural backedge only when `header` dominates `latch`.

After identifying all such edges, 0.19 temporarily removes them and requires the remaining graph to be acyclic. If a cycle remains, the CFG is irreducible and the tier fails closed to the older JIT/interpreter pipeline.

The first cyclic tranche also requires each loop header to have:

- exactly one natural backedge/latch, and
- exactly one predecessor outside the natural loop.

This intentionally excludes multi-latch loops such as some `continue`-heavy shapes until multi-input cyclic phis are represented and audited explicitly.

Nested natural loops are supported when each header satisfies those rules.

## Natural-loop construction

For each backedge, the natural loop starts with the header and latch and walks predecessor edges backward from the latch until the header boundary. Loop exits are every edge from a loop block to a block outside that set.

The resulting `CFGNaturalLoop` records:

- header block
- latch block
- member blocks
- external preheader predecessor
- exit edges

This is optimization metadata rather than Python syntax.

## Loop-carried phi values

0.18 could build phis only after all predecessor exit states already existed. Cycles break that source-order assumption.

0.19 compiles the CFG in topological order **with natural backedges removed**. When it reaches a loop header, it creates provisional phi values only for registers that are:

1. live at the header, and
2. written somewhere inside that natural loop.

Registers that are live but never written by the loop retain their preheader value identity and are loop invariants.

The loop body then compiles against those symbolic phi IDs. Once the latch exit state exists, the provisional phi is patched with its exact backedge input. The final graph can therefore contain the expected SSA cycle:

```text
phi(i_preheader, i_next)
             ↑        |
             |        v
             +--- add(phi, 1)
```

No runtime mutation is hidden in this representation; it is a symbolic value cycle that the backend materializes according to predecessor identity.

## Dominance-aware CSE inside loops

Dominance makes cross-block reuse safe in cyclic CFGs as well:

- a preheader expression may be reused in the header/body because the preheader dominates them;
- a header expression may be reused later in that iteration because the header dominates the loop body;
- a latch-only expression cannot be reused at the header because the latch does not dominate the first entry to the header.

The generated Python assignment therefore executes on every dynamic path required by the value's dominance proof.

## Structured natural-loop lowering

The common whole-function shape

```text
preheader
    ↓
 header/condition ─────→ exit/return
    ↓
 linear body
    ↓
  latch
    └──────────────→ header
```

lowers directly to a Python `while True` with an explicit break on the CFG exit condition. Loop-carried phi assignments are emitted once on preheader entry and again on each latch-to-header transition.

This removes the synthetic `_state` dispatch from the common typed `while` loop while preserving the same CFG/value proof used by the generic backend.

Branchy or nested reducible loops remain in the generic predecessor-tracked state machine. They still use the same loop phi nodes and exact fuel/side-exit contract; only the final control-flow lowering differs.

The older structured hot-loop compiler remains available for top-level/region compilation and for shapes that never enter whole-function CFG Value IR. 0.19 therefore begins the unification without deleting a proven fallback prematurely.

## Exact fuel and side exits

Fuel remains Lua-bytecode fuel.

Every compiled block retains its original instruction count. Before a block executes, the backend checks whether the whole pure block fits in the remaining budget. If it does not:

- none of that block is charged;
- bytecode-live entry registers are reconstructed from the current value graph;
- `frame.pc` is set to the exact block start;
- execution suspends to Tier 0.

At a loop header, predecessor-specific phi assignments occur before that preflight. A quota side exit therefore reconstructs the exact loop-carried state for the iteration that Tier 0 is about to execute.

Differential tests compare JIT and interpreter outcomes across every nearby fuel value for the structured natural-loop path.

## Fail-closed boundary

0.19 does not broaden trust for ordinary Lua or binary chunks. The tier is available only to the existing fully typed source contract.

The first cyclic Value IR still rejects or delegates:

- irreducible CFGs
- multi-latch loop headers
- multiple external entries to one loop header
- calls inside CFG Value IR
- tables and mutable alias-sensitive values
- closures/upvalues
- varargs
- generic/metamethod-sensitive arithmetic
- unsupported scalar type joins

Those cases continue through the older exact JIT tiers or Tier 0.

## Validation

The 0.19 tests cover:

- positive dominance-qualified cross-block CSE
- negative sibling-branch CSE without dominance
- loop-carried integer phi construction
- natural backedge/dominator relationships
- direct structured `while` lowering
- branchy reducible cyclic fallback through the generic CFG backend
- nested natural loops in one cyclic value graph
- exact JIT/interpreter quota agreement across nearby loop fuel boundaries

The official Lua 5.5.1 conformance suite remains a release gate.

## Next compiler work

The next high-value steps are:

1. multi-latch loop phis and `continue`-heavy natural loops
2. dominance frontiers and more general SSA placement
3. PRE/GVN across sibling branches where equivalence can be proven
4. loop-invariant code motion using the explicit natural-loop metadata
5. scalar induction-variable recognition and strength reduction
6. CFG Value IR support for typed CALL IR
7. guarded table/shape facts inside cyclic regions
8. progressively replace older region loop specializers when the CFG tier matches their coverage and speed
9. native x86-64/AArch64 or LLVM lowering from the same typed/value/CFG/CALL representation

Exact Lua 5.5.1 semantics, signed-64 arithmetic, path-dependent fuel, frame state, and fail-closed fallback remain mandatory for every extension.
