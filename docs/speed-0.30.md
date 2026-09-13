# Python fast paths (0.30)

LuaPyre 0.30 follows the remaining CPython profiles without changing the
sandbox or Lua semantics:

1. Generated constant type guards become direct representation checks, so hot
   typed calls do not enter the generic `type_matches` helper.
2. A numeric loop containing one forward `if`/`else` diamond becomes a Python
   `while` plus `if`, rather than a state/match dispatcher. Each source path is
   still charged its exact bytecode fuel; insufficient batches resume Tier 0.
3. Compiled call sites cache the exact Closure/compiled-runner pair. Normally
   returned recursive Frames re-enter their bounded pool without clearing state
   that the admitted compiled opcode set cannot mutate.
4. Constant table keys are hashed during compilation. Generated reads perform
   a direct array/dictionary access after the ordinary plain-table/metatable
   guard instead of repeating Lua key normalization.
5. Python calls to a fully typed straight-line leaf enter its cached generated
   runner directly. Argument validation, quota checks, result conversion, GC
   safepoints, and the exact interpreter fallback remain at the host boundary.

All five paths fail closed when their static or runtime preconditions do not
hold. Dynamic Lua and unsupported typed control flow continue through the prior
JIT tiers or the interpreter.
