# Automatic generational GC pacing in 0.24

LuaPyre has two garbage-collection layers. CPython owns physical allocation and
reclamation. LuaPyre's semantic collector decides the behavior Lua can observe:
weak tables, ephemerons, finalizers, resurrection, and the `collectgarbage`
control surface.

0.24 charges Lua tables, closures, captured cells, coroutine objects, and table
growth to an allocation-debt counter. Allocation sites in the interpreter and
all generated JIT tiers poll that debt before publishing a new object. The poll
is a safe point: active frames, live registers, host-call values, globals, and
coroutine stacks remain roots for the collection.

The default mode is generational. The first semantic collection establishes an
old-generation baseline. Young objects survive through new and survival ages,
then become old. Table, metatable, and captured-cell writes record old objects
that acquire young references. Minor collections scan those remembered objects
and active stacks while skipping the rest of the old graph. Accumulated
allocation growth periodically requests a full major collection.

Programs without weak tables or finalizers cannot observe Lua-level tracing.
Their debt polls perform only a CPython generation-0 step and reset the debt.
Once a weak table or `__gc` metatable appears, LuaPyre immediately schedules a
baseline semantic collection and retains the conservative observable mode for
the life of the runtime. This avoids repeated Python-level graph traces in
ordinary table-allocation workloads while preserving weak-reference and
finalizer behavior.

`collectgarbage("stop")` and `collectgarbage("restart")` gate automatic polls.
Explicit `collectgarbage("collect")` still completes a major semantic cycle.
In generational mode, `collectgarbage("step")` completes a young cycle after a
baseline exists; incremental mode completes a major cycle at each requested
step. The existing collector parameters control debt and major pacing, with a
64 KiB minimum interval to amortize Python-level tracing overhead.

Use `python benchmarks/gc_pacing_ab.py` to compare stopped and automatically
paced execution on allocation-heavy binary trees and weak-value churn. The
binary-tree case verifies that unobservable table garbage stays on the cheap
CPython young-generation path; weak churn verifies that automatic semantic
cycles clear unreachable values without an explicit collection.
