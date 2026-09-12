# LuaPyre 0.18 CFG-aware value IR

LuaPyre 0.18 extends the 0.17 value/expression compiler across **acyclic typed scalar control flow**. The new tier is deliberately narrow: it proves exact merge and deoptimization semantics for forward branches before loop-carried SSA, mutable tables, calls, closures, or guards are admitted.

The certified typed-function pipeline is now:

```text
-- luapyre: typed source
        ↓
LuaPyre bytecode / static type facts
        ↓
TypedIRPlan
  CFG / type / alias facts
        ↓
CFGValueIRPlan
  basic blocks / liveness / scalar values / phi-like joins
        ↓
block-local folding + CSE / live-in merge proof
        ↓
structured Python if/else for simple diamonds
        OR generic forward-DAG state machine
        ↓
exact block-entry side exit to Tier 0
```

Python AST remains a code-generation backend. CFG structure, value identity, join semantics, liveness, and side-exit state are represented before Python lowering so future native backends can consume the same correctness contract.

## Admitted control flow

The first 0.18 CFG tier accepts fully typed functions whose reachable bytecode contains only scalar-safe operations plus forward control flow:

- `LOADK`, `MOVE`, `LOCAL`
- typed integer and floating add/subtract/multiply
- `NOT`, `TOBOOL`
- scalar `EQ`, `LT`, `LE`
- `JMP`, `JMPIF`, `JMPIFNOT`, `JMPIFNIL`
- `RETURN`, `HALT`

Every reachable edge must move forward. Backedges, calls, tables, closures, upvalues, mutable effects, varargs, generic/metamethod-sensitive arithmetic, and unsupported joins fail closed to the existing 0.17/0.16/0.15 tiers or Tier 0.

This restriction is intentional. Acyclic CFGs let merge values and side exits be audited independently before loop-carried values introduce a fixed-point SSA problem.

## Basic blocks and liveness

`CFGValueIRCompiler` splits reachable bytecode into basic blocks and computes predecessor/successor relationships. It then performs standard backward block liveness using the same register read/write model already used by typed IR.

Liveness is essential because LuaPyre is a register VM. Compiler scratch registers may contain unrelated stale values on sibling paths even when no later bytecode can observe them. Requiring every physical register to agree at a join would therefore reject valid programs and confuse register-allocation artifacts with Lua semantics.

0.18 merges and rematerializes **only bytecode-live block inputs**. Dead physical scratch state does not participate in semantic join proof.

## Phi-like merge values

When all predecessors provide the same value identity for a live register, that identity flows directly into the successor.

When predecessor identities differ, 0.18 creates a phi-like `ValueNode` if and only if:

- the register is live at the join,
- every incoming value has the same proven scalar type, and
- that type is safe for the value tier.

The node records both incoming value IDs and predecessor block IDs. If type or value safety cannot be proven, the CFG tier rejects the function and falls back.

Copies remain aliases. Integer/float/boolean/scalar expressions reuse the 0.17 value representation, including exact signed-64 folding and value numbering.

## Dominance and CSE

0.18 intentionally keeps expression CSE **block-local**.

Arguments, constants, and scalar literals may be shared globally because they dominate every use. Computed expression intern entries are cleared at each basic-block boundary. This prevents an expression computed only in one sibling branch from being reused in another branch or after a join without a dominance proof.

A later tranche can add explicit dominators and dominance-aware cross-block CSE. The first CFG release prefers an obviously correct boundary over speculative global reuse.

## Exact fuel and side exits

Fuel remains defined by Lua bytecode, not generated Python operations.

Each compiled basic block carries its original instruction count. Before executing a block, the backend checks whether that entire pure block fits in the remaining budget:

- if it fits, the block executes and exactly its original bytecode count is added to the shared meter;
- if it does not fit, the backend consumes none of that block, reconstructs the successor block's bytecode-live input registers, sets `frame.pc` to the block start, and suspends to Tier 0.

Tier 0 then executes the original bytecode instruction by instruction and reaches the same quota boundary. Differential tests compare JIT and interpreter outcomes across every nearby fuel value for both simple diamonds and larger forward DAGs.

Path-dependent cost is preserved: only blocks on the actually selected branch are charged.

## Structured diamonds

A common four-block shape

```text
       entry / condition
          /       \
       then       else
          \       /
             join
```

is lowered directly to a Python `if`/`else`. Phi values are assigned from the selected predecessor before the join. Exact block preflight and live-state rematerialization remain in each path.

This avoids synthetic `_state` dispatch overhead for the most common reducible branch shape.

More general acyclic forward CFGs lower through a compact state machine with explicit predecessor tracking. The same phi and fuel contracts apply, so optimization of additional reducible structures can happen later without changing semantics.

## Performance

Focused same-runner CPython 3.13.15 measurements compare merged 0.17, the generic CFG state-machine backend, and the structured-diamond backend on the same GitHub runner. The merge-heavy `cfg_phi_branch` medians were:

- merged 0.17: **106.73 ms**
- generic 0.18 CFG value IR: **94.63 ms**
- structured 0.18 CFG value IR: **93.28 ms**

The generic CFG tier is about **11.3% faster than merged 0.17** on this probe. Structured diamond lowering is about **12.6% faster than merged 0.17** and a further **1.4% faster than the generic CFG state dispatcher** on the same runner. The existing 0.17 value/CALL probes remain in the same performance band, so the new tier does not trade away the earlier hot paths.

The main purpose of 0.18 remains architectural: value optimization can now cross branch joins with exact live-state reconstruction. Larger performance gains are expected when loop-carried values and dominance-aware optimization can reuse this representation.

## Fail-closed boundary

0.18 does not make ordinary Lua statically trusted. The CFG value tier is reached only through the existing fully typed source contract. PUC/binary chunks and ordinary dynamic Lua retain their existing guarded/interpreter semantics.

Unsupported control flow or values return `None` from the new compiler tier and immediately continue through older proven JIT tiers. The exact Lua 5.5.1 interpreter remains the universal semantic oracle.

## Next compiler work

The next natural steps are:

1. explicit dominators and dominance-aware cross-block value reuse
2. constant-condition edge pruning where the condition is an immutable value
3. guard/side-exit blocks with exact mid-function rematerialization
4. cyclic CFG construction and loop-carried phi values
5. structured loop recovery from cyclic CFG value IR
6. table/shape/cache facts in the value graph
7. CALL IR integration inside CFG regions
8. recursive/self-call identity through the existing real-frame machinery
9. native x86-64/AArch64 or LLVM lowering from the same typed/value/CFG/CALL IR

Exact Lua 5.5.1 semantics, signed-64 arithmetic, fuel accounting, frame state, and fail-closed fallback remain release gates for every extension.
