from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src/luapyre/gc.py"
TEST = ROOT / "tests/test_dispatch_architecture.py"
WORKFLOW = ROOT / ".github/workflows/_gc-dispatch-refactor-once.yml"
SELF = Path(__file__)

source = TARGET.read_text(encoding="utf-8")
tree = ast.parse(source)
lines = source.splitlines()
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LuaGC")
methods = {
    node.name: node
    for node in cls.body
    if isinstance(node, ast.FunctionDef) and node.name in {"_ins_reads_writes", "_successors"}
}
if set(methods) != {"_ins_reads_writes", "_successors"}:
    raise RuntimeError("GC opcode analysis methods changed")

replacement_reads = '''    @staticmethod
    def _ins_reads_writes(proto: Proto, ins) -> tuple[set[int], set[int]]:
        return _RW_HANDLERS[ins.op](proto, ins)'''
replacement_succ = '''    @staticmethod
    def _successors(code, index: int) -> tuple[int, ...]:
        ins = code[index]
        return _SUCCESSOR_HANDLERS[ins.op](code, index, ins)'''

for name, replacement in sorted(
    (("_ins_reads_writes", replacement_reads), ("_successors", replacement_succ)),
    key=lambda pair: methods[pair[0]].lineno,
    reverse=True,
):
    node = methods[name]
    start = node.lineno - 2  # include @staticmethod
    end = node.end_lineno
    lines[start:end] = replacement.splitlines()

updated = "\n".join(lines) + ("\n" if source.endswith("\n") else "")

helpers = r'''

def _rw_empty(_proto: Proto, _ins) -> tuple[set[int], set[int]]:
    return set(), set()


def _rw_write_a(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return set(), {ins.a}


def _rw_read_a(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.a}, set()


def _rw_read_b(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.b}, set()


def _rw_read_b_write_a(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.b}, {ins.a}


def _rw_setcell(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.a, ins.b}, {ins.a}


def _rw_closure(proto: Proto, ins) -> tuple[set[int], set[int]]:
    reads = {
        desc.index
        for desc in proto.children[ins.b].upvalues
        if desc.kind == "local"
    }
    return reads, {ins.a}


def _rw_gettable(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.b, ins.c}, {ins.a}


def _rw_settable(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.a, ins.b, ins.c}, set()


def _rw_setlistv(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.a, ins.c}, set()


def _rw_forprep(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    regs = {ins.a, ins.b, ins.c}
    return set(regs), regs


def _rw_forloop(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.a, ins.b, ins.c}, {ins.a}


def _rw_pforprep(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    regs = {ins.a, ins.a + 1, ins.a + 2}
    return set(regs), regs


def _rw_pforloop(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.a, ins.a + 1, ins.a + 2}, {ins.a, ins.a + 2}


def _rw_ptforprep(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    regs = {ins.a + 2, ins.a + 3}
    return set(regs), regs


def _rw_ptforloop(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.a + 3}, set()


def _rw_call(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    reads = {ins.b, *range(ins.c, ins.c + max(0, ins.d))}
    if ins.e == 0:
        writes = set()
    elif ins.e == -1:
        writes = {ins.a}
    else:
        writes = set(range(ins.a, ins.a + max(0, ins.e)))
    return reads, writes


def _rw_callv(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    reads = {ins.b, ins.e, *range(ins.c, ins.c + max(0, ins.d))}
    return reads, {ins.a}


def _rw_tailcall(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.b, *range(ins.c, ins.c + max(0, ins.d))}, set()


def _rw_tailcallv(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.b, ins.e, *range(ins.c, ins.c + max(0, ins.d))}, set()


def _rw_vararg(proto: Proto, ins) -> tuple[set[int], set[int]]:
    reads = {proto.vararg_name_reg} if proto.vararg_name_reg >= 0 else set()
    writes = {ins.a} if ins.b == -1 else set(range(ins.a, ins.a + max(0, ins.b)))
    return reads, writes


def _rw_pvararg(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    reads = {ins.c} if ins.c >= 0 else set()
    writes = {ins.a} if ins.b == -1 else set(range(ins.a, ins.a + max(0, ins.b)))
    return reads, writes


def _rw_pgetvarg(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.b}, {ins.a}


def _rw_unpack(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return {ins.b}, set(range(ins.a, ins.a + max(0, ins.c)))


def _rw_return(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    return set(range(ins.a, ins.a + max(0, ins.b))), set()


def _rw_returnv(_proto: Proto, ins) -> tuple[set[int], set[int]]:
    reads = set(range(ins.a, ins.a + max(0, ins.b)))
    reads.add(ins.c)
    return reads, set()


_RW_HANDLERS = {op: _rw_empty for op in Op}
_RW_HANDLERS.update({
    Op.LOADK: _rw_write_a,
    Op.MOVE: _rw_read_b_write_a,
    Op.LOCAL: _rw_read_b_write_a,
    Op.GETGLOBAL: _rw_write_a,
    Op.SETGLOBAL: _rw_read_a,
    Op.GETUPVAL: _rw_write_a,
    Op.SETUPVAL: _rw_read_b,
    Op.GETCELL: _rw_read_b_write_a,
    Op.SETCELL: _rw_setcell,
    Op.CLOSURE: _rw_closure,
    Op.NEWTABLE: _rw_write_a,
    Op.GETTABLE: _rw_gettable,
    Op.SETTABLE: _rw_settable,
    Op.SETLISTV: _rw_setlistv,
    Op.FORPREP: _rw_forprep,
    Op.FORLOOP: _rw_forloop,
    Op.PFORPREP: _rw_pforprep,
    Op.PFORLOOP: _rw_pforloop,
    Op.PTFORPREP: _rw_ptforprep,
    Op.PTFORLOOP: _rw_ptforloop,
    Op.CALL: _rw_call,
    Op.CALLV: _rw_callv,
    Op.TAILCALL: _rw_tailcall,
    Op.TAILCALLV: _rw_tailcallv,
    Op.VARARG: _rw_vararg,
    Op.PVARARG: _rw_pvararg,
    Op.PGETVARG: _rw_pgetvarg,
    Op.UNPACK: _rw_unpack,
    Op.TBC: _rw_read_a,
    Op.PTBC: _rw_read_a,
    Op.CHECKNIL: _rw_read_a,
    Op.RETURN: _rw_return,
    Op.RETURNV: _rw_returnv,
    Op.GUARD: _rw_read_a,
})
for _op in _BINARY_OPS:
    _RW_HANDLERS[_op] = _rw_gettable
for _op in _UNARY_OPS:
    _RW_HANDLERS[_op] = _rw_read_b_write_a
for _op in _CONDITIONAL_JUMPS:
    _RW_HANDLERS[_op] = _rw_read_b


def _succ_fallthrough(code, index: int, _ins) -> tuple[int, ...]:
    return (index + 1,) if index + 1 < len(code) else ()


def _succ_terminate(_code, _index: int, _ins) -> tuple[int, ...]:
    return ()


def _succ_jump(code, _index: int, ins) -> tuple[int, ...]:
    return (ins.a,) if 0 <= ins.a < len(code) else ()


def _succ_conditional(code, index: int, ins) -> tuple[int, ...]:
    out = []
    if index + 1 < len(code):
        out.append(index + 1)
    if 0 <= ins.a < len(code):
        out.append(ins.a)
    return tuple(out)


def _succ_loop(code, index: int, ins) -> tuple[int, ...]:
    out = []
    if index + 1 < len(code):
        out.append(index + 1)
    if 0 <= ins.d < len(code):
        out.append(ins.d)
    return tuple(out)


def _succ_ptforprep(code, _index: int, ins) -> tuple[int, ...]:
    return (ins.d,) if 0 <= ins.d < len(code) else ()


_SUCCESSOR_HANDLERS = {op: _succ_fallthrough for op in Op}
for _op in _TERMINATORS:
    _SUCCESSOR_HANDLERS[_op] = _succ_terminate
_SUCCESSOR_HANDLERS[Op.JMP] = _succ_jump
for _op in _CONDITIONAL_JUMPS:
    _SUCCESSOR_HANDLERS[_op] = _succ_conditional
for _op in (Op.FORPREP, Op.FORLOOP, Op.PFORPREP, Op.PFORLOOP, Op.PTFORLOOP):
    _SUCCESSOR_HANDLERS[_op] = _succ_loop
_SUCCESSOR_HANDLERS[Op.PTFORPREP] = _succ_ptforprep

if set(_RW_HANDLERS) != set(Op) or set(_SUCCESSOR_HANDLERS) != set(Op):
    raise RuntimeError("GC opcode analysis dispatch table mismatch")
'''

marker = "\n\n@dataclass(slots=True)\nclass GCStats:"
if marker not in updated:
    raise RuntimeError("GC helper insertion marker changed")
updated = updated.replace(marker, helpers + marker, 1)
ast.parse(updated)
TARGET.write_text(updated, encoding="utf-8")

architecture = TEST.read_text(encoding="utf-8")
architecture += r'''


def test_gc_opcode_analysis_uses_complete_dispatch_tables():
    from luapyre.bytecode import Op
    from luapyre.gc import LuaGC, _RW_HANDLERS, _SUCCESSOR_HANDLERS

    assert set(_RW_HANDLERS) == set(Op)
    assert set(_SUCCESSOR_HANDLERS) == set(Op)
    assert "elif op" not in inspect.getsource(LuaGC._ins_reads_writes)
    assert "elif op" not in inspect.getsource(LuaGC._successors)
'''
TEST.write_text(architecture, encoding="utf-8")

SELF.unlink()
if WORKFLOW.exists():
    WORKFLOW.unlink()
