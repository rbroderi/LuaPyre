# LuaPyre 0.20 loop optimization

LuaPyre 0.20 builds on the 0.19 dominance/cyclic CFG Value IR with a small backend-neutral loop optimizer. The goal remains a **small statically typed IR compiler targeting optimized Python AST**, with Python AST treated as a backend rather than the semantic optimizer.

The certified typed-function pipeline is now:

```text
-- luapyre: typed source
        ↓
TypedIRPlan
        ↓
CFGValueIRPlan
  liveness
  dominators
  natural loops
  branch/loop phi values
        ↓
CFGLoopOptimizer
  LICM proof
  canonical integer induction variables
        ↓
optimized Python AST
  invariant materialization in preheader
  direct wrapped induction-phi update
        ↓
exact side exit to Tier 0
```

Table/environment-heavy loop regions continue to use the existing TypedIR loop path, where stable `_ENV` constant-key reads already carry backend-neutral invariance facts and their raw table/metatable guard is emitted before the generated Python loop. 0.20 therefore extends the same architectural principle instead of duplicating table semantics in CFG Value IR.

## Loop-invariant code motion

`CFGLoopOptimizer` operates on the 0.19 natural-loop and dominance metadata. A Value-IR expression defined inside a loop is hoistable only when:

1. its operation is explicitly on the LICM-safe whitelist;
2. every operand is either already proven invariant or is defined outside the loop;
3. an outside expression definition dominates the loop header; and
4. the value graph is the certified scalar CFG domain, with no dynamic/metamethod-sensitive operation hidden behind the node.

The first whitelist is deliberately small:

- signed-64 `add_i`, `sub_i`, `mul_i`
- typed `add_f`, `sub_f`, `mul_f`
- `not`, `tobool`
- typed `eq`, `lt`, `le`

This whitelist is fail-closed. Adding a new Value-IR opcode does not make it hoistable automatically. It must be audited for purity, exception behavior, and Lua observability first.

The Python backend materializes proven invariant expressions once after the function preheader and omits their repeated Python assignments inside the loop.

### Fuel remains source-bytecode fuel

LICM changes generated Python operations, not Lua execution accounting. Each CFG block retains its original bytecode cost and the JIT still charges that entire cost whenever the source block executes. A moved or eliminated Python assignment therefore does **not** refund quota.

The same rule applies to zero-trip loops: an admitted invariant scalar expression may be materialized speculatively because the current LICM whitelist is pure/non-throwing under certified operand types, but the loop body's Lua fuel is never charged unless the body actually executes.

Near-boundary differential tests compare JIT and interpreter outcomes across hundreds of fuel values.

## Canonical integer induction variables

0.20 recognizes a loop-header integer phi when its backedge value has the canonical form:

```text
phi(init, update)
update = add_i(phi, integer_constant)
```

or:

```text
update = sub_i(phi, integer_constant)
```

The signed step is represented with Lua's exact modulo-2**64 integer behavior.

If the backedge expression's only consumer is the loop phi, a backend may eliminate the temporary update value and update the phi directly at the latch. The Python backend emits the same explicit mask/sign correction used elsewhere in LuaPyre's integer JIT paths.

If another expression consumes the update node, the optimization is declined and the ordinary SSA value remains materialized.

## Induction limit recognition

For future range/predication work, the induction descriptor also records a typed `lt`/`le` comparison when the loop condition compares the phi against an invariant bound.

Source lowering sometimes wraps the comparison in pure `tobool`/`not` nodes before `JMPIF`/`JMPIFNOT`. Induction analysis peels those normalization nodes for recognition while the backend continues to execute the original condition graph.

This metadata does not yet remove the loop comparison or prove trip counts; it establishes the IR contract for later range checks, strength reduction, and loop predication.

## Guard hoisting

0.20 keeps guard specialization at the IR layer that owns the relevant semantics.

For table/environment loop regions, `TypedIRCompiler` already proves invariant global reads only when the region contains no operation that can mutate or re-enter `_ENV` (`SETTABLE`, global/upvalue mutation, calls, and related barriers are conservative blockers). The Python backend then:

1. resolves the stable table/key once before the generated loop;
2. guards that the receiver is the expected raw `LuaTable` shape with no metatable;
3. deoptimizes to the exact loop start before consuming Lua fuel if the guard fails; and
4. reuses the loaded value inside the loop while still charging the original GETTABLE instruction each iteration.

This is deliberately speculative deoptimization, not eager Lua execution. A failed hoisted guard therefore cannot manufacture a Lua error on a path/iteration that has not executed yet.

CFG Value IR remains scalar-only in this tranche. Table/shape facts can move into the cyclic CFG representation later once alias/guard snapshots are represented there explicitly.

## Python backend

The optimized structured natural-loop backend reuses the 0.19 generated-code filename for tooling compatibility but has a distinct function name (`_jit_cfg_value_ir_loop_opt`). Its sequence is:

```text
preheader preflight + original expressions
hoisted invariant expressions once
initialize header phis
while True:
    header preflight + non-hoisted expressions
    exit check
    body/latch preflights + non-hoisted expressions
    direct induction updates when proven
    ordinary phi copies otherwise
exit block
```

Unsupported shapes fall through to the 0.19 CFG backend, older structured loop tiers, or Tier 0.

## Validation

0.20 adds permanent tests for:

- LICM proof of a scalar `a + b` invariant inside a natural loop;
- canonical integer induction recognition;
- comparison/limit association through boolean normalization;
- direct induction-phi update admission only when the update value has no other consumer;
- optimized backend selection;
- zero-trip loop behavior; and
- exact JIT/interpreter quota agreement across nearby fuel boundaries.

Existing TypedIR tests continue to cover invariant `_ENV` reads, mutation barriers that disable hoisting, global-read execution, and table cache invalidation.

## Same-runner performance

CPython 3.13.15 on Ubuntu 24.04, 3 warmups and 9 timed samples against merged 0.19 (`5c68b08`):

| probe | 0.19 | 0.20 | change |
| --- | ---: | ---: | ---: |
| `licm_invariant` | 231.531 ms | 194.750 ms | **-15.9% (~1.19x)** |
| `induction_while` | 235.507 ms | 225.432 ms | **-4.3% (~1.04x)** |

The LICM workload deliberately performs an invariant typed addition in every loop iteration. The induction probe has no additional invariant body computation and isolates the smaller benefit from eliminating the temporary induction update/copy sequence.

## Fail-closed boundary

0.20 does not change trust for ordinary Lua, binary chunks, dynamic arithmetic, or unsupported CFGs. In particular, CFG LICM still does not admit:

- calls or re-entry;
- tables/metatables;
- closures/upvalues;
- varargs;
- throwing/dynamic arithmetic;
- irreducible or multi-latch cyclic CFGs outside the 0.19 contract; or
- any future Value-IR operation not explicitly audited for LICM.

Those paths continue through proven older tiers or Tier 0.

## Next compiler work

The HotSpot/LuaJIT-inspired next steps are:

1. profile-backed monomorphic/polymorphic inline caches for calls and table shapes;
2. deoptimization reason/site feedback so repeated guard misses widen specialization instead of thrashing;
3. dominance-frontier/PRE and broader global value numbering;
4. loop predication/range analysis using the induction/limit metadata;
5. hot side-exit compilation and CFG stitching;
6. CFG OSR using the existing exact Frame/PC/register contract;
7. escape analysis and virtual Frame/MultiValue materialization;
8. standard-library intrinsics; and
9. eventual native lowering from the same typed/value/CALL/CFG/loop IR.

Exact Lua 5.5.1 semantics, signed-64 arithmetic, path-dependent fuel, frame state, and fail-closed fallback remain mandatory for every extension.
