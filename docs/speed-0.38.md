# LuaPyre 0.38 performance tranche

0.38 implements four narrow Python/AST-only paths from the
[0.38 roadmap](performance-roadmap-0.38.md). It does not generate CPython
bytecode directly and adds no native runtime component.

## Accepted changes

1. **Pure recursive bases avoid frames.** A certified typed integer base arm
   charges exact fuel and returns its scalar directly. Debug hooks, inadequate
   fuel, invalid types, and other branches use the existing materialized path.
2. **Scalar compiled-call entries.** Fixed unary/binary, one-result compiled
   calls pass Python scalars without the parent's argument tuple or generic
   result-sequence extraction. Real frames and the child's internal return
   representation remain for non-base execution and suspension recovery.
3. **Dense primitive-write regions.** Stable aliases and positive bounded keys
   are proved once. The generated loop guards dense extent and metatable state,
   binds the array once, and writes primitive values directly while incrementing
   the table version.
4. **Cheaper record construction.** Constant facts now survive recursive call
   continuations. Unique literal fields use a fresh pre-hashed setter; ordinary
   tables allocate iteration/deletion metadata only if traversal requires it.
   GC barriers and allocation accounting remain intact.

## Measurements

The baseline is merged `main` at `10abe6061f7c8937e23ab07fc6b9ab5c6cb8d168`.
Each result is the median of three paired process medians with alternating
Lua/Python timing order, seven warmups, and 31 checked samples. Python GC is
disabled only during steady timing; Lua GC remains active. Full medians are retained in
[`speed_038.json`](../benchmarks/results/speed_038.json).

| Target | Python | 0.37 | 0.38 | Change | 0.38 / Python |
| --- | --- | ---: | ---: | ---: | ---: |
| Balanced recursive calls | 3.13.15 | 1.658 ms | 1.293 ms | **22.0% faster** | 30.20× |
| Balanced recursive calls | 3.14.7 | 1.336 ms | 1.086 ms | **18.7% faster** | 33.15× |
| Recursive record continuations | 3.13.15 | 7.089 ms | 6.703 ms | **5.4% faster** | 47.81× |
| Recursive record continuations | 3.14.7 | 5.950 ms | 5.759 ms | **3.2% faster** | 42.29× |
| Dense aliased primitive writes | 3.13.15 | 3.342 ms | 3.289 ms | **1.6% faster** | 26.12× |
| Dense aliased primitive writes | 3.14.7 | 2.691 ms | 2.581 ms | **4.1% faster** | 24.47× |

Controls remained below the 5% regression gate: linear recursion was -0.9%
and -1.5%, materialized binary scalar calls -0.3%/+0.5%, read-only dense loops
-0.1%/+0.6%, and modulo-heavy table mix -0.4%/-2.7% on 3.13/3.14 respectively.

## Validation

- All 598 Python tests pass on CPython 3.13.15 and 3.14.7.
- Deterministic additions cover frame-free base entry, all tested fuel
  boundaries, unary/binary scalar entry, dense code shape, fresh nil writes,
  GC adoption/accounting, iteration, and duplicate/dynamic constructor keys.
- All 24 required unchanged official Lua 5.5.1 probes pass on both versions.
- Penlight passes 23/23, luatest 5/5, LuaCov scanner specs 24/24, and Are We
  Fast Yet 11/11 on both versions. Established native-module and unsafe-I/O
  exclusions remain unchanged.

## Deferred work

This is base-case frame elimination, not general lazy activation recovery.
Effectful, mutually recursive, yielding, and single-chain calls retain real
frames. Dense proof excludes collectable/nil writes, calls, holes, growth,
metatables, and unproved aliases. General record-layout replacement remains
deferred because identity, weak tables, finalizers, and iteration order are
observable Lua semantics.
