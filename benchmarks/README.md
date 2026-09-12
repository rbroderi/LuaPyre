# LuaPyre benchmarks

## Four-way runtime comparison

`compare_runtimes.py` is the standard end-to-end runtime benchmark for LuaPyre. It executes the same Lua workloads against four backends:

1. **LuaPyre JIT** — the tiered Python-code JIT enabled.
2. **LuaPyre interpreter** — `LuaRuntime(jit=False)`, which remains the semantic oracle and deoptimization target.
3. **Lua 5.5** — native Lua 5.5 through `lupa.lua55`.
4. **LuaJIT** — LuaJIT 2.1 through `lupa.luajit21` when available, falling back to 2.0.

Each workload is checked for the expected result before its timing is accepted. Warmup runs happen first, garbage collection is disabled during timed samples, and the report prints the median and records the best sample. The default is 3 warmups and 7 timed samples.

Install the development dependencies and run locally with:

```bash
python -m pip install -e '.[dev]'
python benchmarks/compare_runtimes.py --require-all --json benchmark-results.json
```

Useful overrides:

```bash
python benchmarks/compare_runtimes.py --warmups 5 --repeats 15 --require-all --json benchmark-results.json
```

The JSON report uses a versioned schema and includes the Python/platform information, GitHub commit SHA when available, backend identities, benchmark settings, and median/best timing for every workload/backend pair. This is intended to make future performance comparisons scriptable instead of relying on log scraping.

### GitHub Actions

The **Four-way runtime benchmark** workflow can be run manually from Actions and can also be called from another workflow. It runs under CPython 3.13, requires all four backends, writes the text report into the Actions job summary, and uploads both the text and JSON reports as an artifact.

For performance comparisons, compare runs on the same runner class and Python version. The Lua 5.5 and LuaJIT measurements execute native runtimes through Lupa, whereas LuaPyre is implemented in Python and its JIT emits Python execution paths, so the native runtimes are reference points rather than implementation-equivalent baselines.

## Other microbenchmarks

- `vm_programs.py` contains the shared workload definitions used by the four-way benchmark.
- `vm_dispatch.py` measures Python opcode-dispatch strategies in isolation.
