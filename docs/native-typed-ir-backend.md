# Native typed-IR backend evaluation

Status: **evaluation complete; deferred by project policy**.

LuaPyre is remaining a Python-only runtime. No native extension, native code
generator, or runtime compiler is planned for the current roadmap. The
analysis below is retained as an architectural record only, should that policy
ever be revisited.

LuaPyre's typed/value/CALL/CFG IR is already backend-neutral, but its current
lowering produces Python functions operating on Python objects. Profiling shows
that generated Python is effective at removing VM dispatch, while arithmetic,
register boxing, and transitions back into Python remain far more expensive
than native Lua.

## Options considered

| Option | Strength | Main problem | Decision |
|---|---|---|---|
| Keep generated Python only | Zero build dependency; exact deopt is mature | Cannot remove Python numeric boxing or interpreter overhead | Retain as universal JIT backend |
| Runtime LLVM through llvmlite | Real machine code and mature optimization | Large/version-coupled dependency; Python-object callbacks erase gains on mixed regions | Do not make this the first native backend |
| Runtime C compiler or executable-memory emitter | Potentially small fast kernels | Requires a toolchain or writable/executable memory at runtime; poor sandbox and deployment properties | Reject |
| Ahead-of-time Cython/mypyc of the VM | Speeds general Python implementation | Does not consume LuaPyre IR or specialize individual regions | Useful independently, not the typed-IR backend |
| Optional prebuilt native region executor | No runtime compiler; wheels can use Python's supported extension ABI; one call can execute a whole region | Requires an unboxed shadow-frame ABI and deopt materialization | Recommended prototype |

Relevant implementation constraints are documented by Python's
[Stable ABI](https://docs.python.org/3/c-api/stable.html),
[Limited API](https://docs.python.org/3/c-api/stable.html#limited-c-api), and
[vectorcall protocol](https://docs.python.org/3/c-api/call.html#vectorcall-protocol).
LLVM remains a possible later compiler through
[llvmlite's binding layer](https://llvmlite.readthedocs.io/en/latest/user-guide/binding/index.html),
but it should compete against the native-executor prototype rather than become
a prerequisite by assumption.

## Previously evaluated prototype

Build an optional extension module that accepts a validated, immutable region
descriptor produced by the existing IR pipeline. At region entry it converts
proven scalar live-ins into a compact shadow frame:

```text
tag: nil | boolean | int64 | float64 | object
payload: union(int64, float64, PyObject*)
```

The native executor handles integer/float arithmetic, comparisons, branches,
loop induction, fuel accounting, and guard checks without allocating Python
numbers per instruction. Object/table/call operations initially side-exit.
On return or deoptimization, only live values from the existing materialization
map are boxed and written to the canonical Python `Frame`.

The extension must be optional. Import or platform failure selects the current
generated-Python backend, which remains both the semantic oracle and the
portable distribution.

## Required gates

The prototype should not become a supported backend unless it demonstrates:

1. at least 5x improvement over generated Python on typed arithmetic loops;
2. at least 2x on one mixed algorithm after entry/exit conversion costs;
3. identical results, fuel boundaries, deopt PCs, overflow, NaN, and error
   behavior across the Python test and official Lua gates;
4. no runtime compiler, subprocess, or writable/executable-memory requirement;
5. prebuilt wheels for supported CPython versions on Windows, macOS, and Linux,
   including x86-64 and ARM64; and
6. no Python callback from inside the scalar instruction loop.

If the executor clears those gates, the next experiment is direct native code
generation for the same shadow-frame ABI. Cranelift or LLVM can then be judged
on compile latency, wheel size, architecture support, and additional speed,
without changing the Lua semantics or deoptimization contract again.
