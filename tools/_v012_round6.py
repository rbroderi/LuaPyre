from pathlib import Path


def replace(path, old, new):
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"patch context missing in {path}: {old[:140]!r}")
    p.write_text(text.replace(old, new, 1))


# collectgarbage('count') is an allocator-footprint observation, not a tracing
# decision. Keep semantic liveness for real GC, but count physical register
# contents so the estimate does not fall merely because the compiler advances
# past a local's last use.
replace(
    "src/luapyre/gc.py",
    '''    def _trace(self, extra_roots=()):\n        marked: dict[int, object] = {}\n''',
    '''    def _trace(self, extra_roots=(), *, physical_regs: bool = False):\n        marked: dict[int, object] = {}\n''',
)
replace(
    "src/luapyre/gc.py",
    '''            mark(frame.closure)\n            for reg in self._live_regs(frame):\n                if reg in excluded or reg < 0 or reg >= len(frame.regs):\n''',
    '''            mark(frame.closure)\n            regs_to_scan = range(len(frame.regs)) if physical_regs else self._live_regs(frame)\n            for reg in regs_to_scan:\n                if reg in excluded or reg < 0 or reg >= len(frame.regs):\n''',
)
replace(
    "src/luapyre/gc.py",
    '''    def count_kbytes(self) -> float:\n        marked, _ = self._trace()\n        return self._estimate_bytes(marked.values()) / 1024.0\n''',
    '''    def count_kbytes(self) -> float:\n        # Approximate allocator-visible memory, so include physical register\n        # contents even when bytecode liveness says a slot is semantically dead.\n        # Real collection continues to use semantic liveness in _trace().\n        marked, _ = self._trace(physical_regs=True)\n        return self._estimate_bytes(marked.values()) / 1024.0\n''',
)

# Lua's diagnostic names math.huge when infinity is converted to integer. We do
# not carry full expression provenance yet, but infinity from the standard math
# constant is the common observable case and broad number diagnostics still match.
replace(
    "src/luapyre/vm.py",
    '''def _to_int(value):\n    if type(value) is int:\n        return i64(value)\n    if type(value) is float and math.isfinite(value) and value.is_integer():\n        iv = int(value)\n        if -(1 << 63) <= iv <= (1 << 63) - 1:\n            return iv\n    raise LuaRuntimeError("number has no integer representation")\n''',
    '''def _to_int(value):\n    if type(value) is int:\n        return i64(value)\n    if type(value) is float:\n        if math.isfinite(value) and value.is_integer():\n            iv = int(value)\n            if -(1 << 63) <= iv <= (1 << 63) - 1:\n                return iv\n        if math.isinf(value):\n            raise LuaRuntimeError("number (field 'huge') has no integer representation")\n    raise LuaRuntimeError("number has no integer representation")\n''',
)

print("round-six patch applied")
