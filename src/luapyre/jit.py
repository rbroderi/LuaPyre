from __future__ import annotations

from dataclasses import dataclass
from types import FunctionType

from .bytecode import Cell, Ins, Op, Proto
from .jit_policy import JIT_LOOP_BODY_OPS
from .table import LuaTable
from .values import i64, lua_equal, type_matches


DEOPT = object()


@dataclass(frozen=True, slots=True)
class IRInstruction:
    """One LuaPyre instruction lowered into the tier-2 IR."""

    pc: int
    ins: Ins
    specialization: str | None = None


@dataclass(frozen=True, slots=True)
class IRBlock:
    """A straight-line basic block.

    0.13 intentionally keeps the first JIT IR small.  Control-flow structure is
    represented by region boundaries rather than by Python source details, so a
    future native backend can consume the same IR.
    """

    start_pc: int
    end_pc: int
    instructions: tuple[IRInstruction, ...]


@dataclass(frozen=True, slots=True)
class IRLoop:
    start_pc: int
    backedge_pc: int
    exit_pc: int
    body: IRBlock
    loop_ins: Ins


@dataclass(slots=True)
class JITStats:
    loop_compiles: int = 0
    loop_executions: int = 0
    loop_iterations: int = 0
    leaf_compiles: int = 0
    leaf_executions: int = 0
    deopts: int = 0
    compile_failures: int = 0
    call_ic_hits: int = 0
    call_ic_misses: int = 0
    table_ic_hits: int = 0
    table_ic_misses: int = 0
    cache_invalidations: int = 0
    megamorphic_sites: int = 0
    retired_regions: int = 0


@dataclass(frozen=True, slots=True)
class CompiledLoop:
    ir: IRLoop
    cost_per_iteration: int
    runner: FunctionType


@dataclass(frozen=True, slots=True)
class CompiledLeaf:
    proto: Proto
    instruction_cost: int
    runner: FunctionType


_LOOP_BODY_OPS = JIT_LOOP_BODY_OPS

_LEAF_OPS = frozenset({
    Op.LOADK,
    Op.MOVE,
    Op.LOCAL,
    Op.ADD,
    Op.ADD_I,
    Op.ADD_F,
    Op.SUB,
    Op.SUB_I,
    Op.SUB_F,
    Op.MUL,
    Op.MUL_I,
    Op.MUL_F,
    Op.MOD,
    Op.EQ,
    Op.LT,
    Op.LE,
    Op.NOT,
    Op.TOBOOL,
    Op.GUARD,
    Op.RETURN,
})


class PythonJIT:
    """Tiered pure-Python JIT for hot LuaPyre regions.

    Tier 0 remains the exact interpreter.  This tier lowers hot natural numeric
    loops and hot straight-line leaf functions into a deliberately small IR and
    then emits Python code.  Speculative operations guard before observable
    mutation and return DEOPT to the interpreter on a miss.

    Native source compilation marks *_I/*_F instructions as type-proven.  Those
    instructions can omit redundant guards; binary chunks and manually created
    Proto objects do not receive that trust marker.
    """

    def __init__(self, *, threshold: int = 32, enabled: bool = True):
        if threshold < 1:
            raise ValueError("jit threshold must be at least 1")
        self.threshold = threshold
        self.enabled = enabled
        self.stats = JITStats()
        self._loop_maps: dict[int, tuple[Proto, dict[int, int]]] = {}
        self._loop_hot: dict[tuple[int, int], int] = {}
        self._loop_cache: dict[tuple[int, int], CompiledLoop | None] = {}
        self._leaf_hot: dict[int, int] = {}
        self._leaf_cache: dict[int, tuple[Proto, CompiledLeaf | None]] = {}

    @staticmethod
    def _profile_binary(regs, ins: Ins) -> str | None:
        left, right = regs[ins.b], regs[ins.c]
        if type(left) is int and type(right) is int:
            return "int"
        if type(left) in (int, float) and type(right) in (int, float):
            return "number"
        if isinstance(left, bytes) and isinstance(right, bytes):
            return "bytes"
        return None

    def _loop_map(self, proto: Proto) -> dict[int, int]:
        ident = id(proto)
        cached = self._loop_maps.get(ident)
        if cached is not None and cached[0] is proto:
            return cached[1]
        entries: dict[int, int] = {}
        for pc, ins in enumerate(proto.code):
            if ins.op in (Op.FORLOOP, Op.JFORLOOP) and 0 <= ins.d < pc:
                entries[ins.d] = pc
        self._loop_maps[ident] = (proto, entries)
        return entries

    def _lower_loop(self, frame, start_pc: int, backedge_pc: int) -> IRLoop | None:
        proto = frame.proto
        body_ins = proto.code[start_pc:backedge_pc]
        if not body_ins or any(ins.op not in _LOOP_BODY_OPS for ins in body_ins):
            return None
        lowered = []
        for offset, ins in enumerate(body_ins):
            pc = start_pc + offset
            specialization = None
            if ins.op in (Op.ADD, Op.SUB, Op.MUL, Op.MOD, Op.EQ, Op.LT, Op.LE):
                specialization = self._profile_binary(frame.regs, ins)
            lowered.append(IRInstruction(pc, ins, specialization))
        loop_ins = proto.code[backedge_pc]
        if loop_ins.op not in (Op.FORLOOP, Op.JFORLOOP) or loop_ins.d != start_pc:
            return None
        return IRLoop(
            start_pc,
            backedge_pc,
            backedge_pc + 1,
            IRBlock(start_pc, backedge_pc, tuple(lowered)),
            loop_ins,
        )

    @staticmethod
    def _guard_deopt(lines: list[str], condition: str, pc: int, offset: int) -> None:
        lines.append(f"        if not ({condition}):")
        lines.append(f"            frame.pc = {pc}")
        lines.append(f"            return used + {offset}, False")

    @staticmethod
    def _emit_typed_arith(
        lines: list[str], ir: IRInstruction, symbol: str, *, floating: bool, trusted: bool,
    ) -> None:
        ins, pc = ir.ins, ir.pc
        if not trusted:
            if floating:
                PythonJIT._guard_deopt(
                    lines,
                    f"type(regs[{ins.b}]) in _NUM_TYPES and type(regs[{ins.c}]) in _NUM_TYPES",
                    pc,
                    0,
                )
            else:
                PythonJIT._guard_deopt(
                    lines,
                    f"type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int",
                    pc,
                    0,
                )
        expr = f"regs[{ins.b}] {symbol} regs[{ins.c}]"
        if floating:
            lines.append(f"        regs[{ins.a}] = float({expr})")
        else:
            lines.append(f"        regs[{ins.a}] = _i64({expr})")

    @staticmethod
    def _emit_generic_binary(
        lines: list[str], ir: IRInstruction, symbol: str, offset: int,
    ) -> bool:
        ins, pc, specialization = ir.ins, ir.pc, ir.specialization
        if specialization == "int":
            PythonJIT._guard_deopt(
                lines,
                f"type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int",
                pc,
                offset,
            )
            lines.append(
                f"        regs[{ins.a}] = _i64(regs[{ins.b}] {symbol} regs[{ins.c}])"
            )
            return True
        if specialization == "number":
            PythonJIT._guard_deopt(
                lines,
                f"type(regs[{ins.b}]) in _NUM_TYPES and type(regs[{ins.c}]) in _NUM_TYPES",
                pc,
                offset,
            )
            lines.append(f"        _a = regs[{ins.b}]; _b = regs[{ins.c}]")
            lines.append(f"        _v = _a {symbol} _b")
            lines.append(
                f"        regs[{ins.a}] = _i64(_v) if type(_a) is int and type(_b) is int else _v"
            )
            return True
        return False

    def _compile_loop(self, frame, start_pc: int, backedge_pc: int) -> CompiledLoop | None:
        ir = self._lower_loop(frame, start_pc, backedge_pc)
        if ir is None:
            return None
        trusted = bool(frame.proto.jit_trust_types)
        body = ir.body.instructions
        cost = len(body) + 1  # body + FORLOOP
        lines = [
            "def _jit_loop(vm, frame, budget):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
            "    cells = frame.cells",
            "    used = 0",
            f"    while budget - used >= {cost}:",
        ]

        for offset, item in enumerate(body):
            ins, pc, op = item.ins, item.pc, item.ins.op
            if op is Op.LOADK:
                lines.append(f"        regs[{ins.a}] = consts[{ins.b}]")
            elif op is Op.MOVE:
                lines.append(f"        regs[{ins.a}] = regs[{ins.b}]")
            elif op is Op.LOCAL:
                lines.append(f"        _v = regs[{ins.b}]")
                lines.append(f"        regs[{ins.a}] = _v")
                lines.append(f"        if {ins.a} in cells:")
                lines.append(f"            cells[{ins.a}] = _Cell(_v)")
            elif op is Op.NEWTABLE:
                lines.append(f"        regs[{ins.a}] = _LuaTable()")
            elif op is Op.GETTABLE:
                self._guard_deopt(
                    lines,
                    f"isinstance(regs[{ins.b}], _LuaTable) and regs[{ins.b}].metatable is None",
                    pc,
                    offset,
                )
                lines.append(f"        regs[{ins.a}] = regs[{ins.b}].rawget(regs[{ins.c}])")
            elif op is Op.SETTABLE:
                self._guard_deopt(
                    lines,
                    f"isinstance(regs[{ins.a}], _LuaTable) and regs[{ins.a}].metatable is None",
                    pc,
                    offset,
                )
                lines.append(f"        _key = regs[{ins.b}]")
                self._guard_deopt(
                    lines,
                    "_key is not None and not (type(_key) is float and _isnan(_key))",
                    pc,
                    offset,
                )
                lines.append(f"        regs[{ins.a}].rawset(_key, regs[{ins.c}])")
            elif op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                symbol = {Op.ADD_I: "+", Op.SUB_I: "-", Op.MUL_I: "*"}[op]
                if not trusted:
                    self._guard_deopt(
                        lines,
                        f"type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int",
                        pc,
                        offset,
                    )
                lines.append(f"        regs[{ins.a}] = _i64(regs[{ins.b}] {symbol} regs[{ins.c}])")
            elif op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                symbol = {Op.ADD_F: "+", Op.SUB_F: "-", Op.MUL_F: "*"}[op]
                if not trusted:
                    self._guard_deopt(
                        lines,
                        f"type(regs[{ins.b}]) in _NUM_TYPES and type(regs[{ins.c}]) in _NUM_TYPES",
                        pc,
                        offset,
                    )
                lines.append(f"        regs[{ins.a}] = float(regs[{ins.b}] {symbol} regs[{ins.c}])")
            elif op in (Op.ADD, Op.SUB, Op.MUL):
                symbol = {Op.ADD: "+", Op.SUB: "-", Op.MUL: "*"}[op]
                if not self._emit_generic_binary(lines, item, symbol, offset):
                    return None
            elif op is Op.MOD:
                if item.specialization != "int":
                    return None
                self._guard_deopt(
                    lines,
                    f"type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int and regs[{ins.c}] != 0",
                    pc,
                    offset,
                )
                lines.append(f"        regs[{ins.a}] = _i64(regs[{ins.b}] % regs[{ins.c}])")
            elif op is Op.EQ:
                if item.specialization not in ("int", "number", "bytes"):
                    return None
                if item.specialization == "int":
                    guard = f"type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int"
                elif item.specialization == "number":
                    guard = f"type(regs[{ins.b}]) in _NUM_TYPES and type(regs[{ins.c}]) in _NUM_TYPES"
                else:
                    guard = f"isinstance(regs[{ins.b}], bytes) and isinstance(regs[{ins.c}], bytes)"
                self._guard_deopt(lines, guard, pc, offset)
                lines.append(f"        regs[{ins.a}] = _lua_equal(regs[{ins.b}], regs[{ins.c}])")
            elif op in (Op.LT, Op.LE):
                if item.specialization not in ("int", "number", "bytes"):
                    return None
                if item.specialization == "int":
                    guard = f"type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int"
                elif item.specialization == "number":
                    guard = f"type(regs[{ins.b}]) in _NUM_TYPES and type(regs[{ins.c}]) in _NUM_TYPES"
                else:
                    guard = f"isinstance(regs[{ins.b}], bytes) and isinstance(regs[{ins.c}], bytes)"
                self._guard_deopt(lines, guard, pc, offset)
                symbol = "<" if op is Op.LT else "<="
                lines.append(f"        regs[{ins.a}] = regs[{ins.b}] {symbol} regs[{ins.c}]")
            elif op is Op.NOT:
                lines.append(f"        _v = regs[{ins.b}]")
                lines.append(f"        regs[{ins.a}] = (_v is None or _v is False)")
            elif op is Op.TOBOOL:
                lines.append(f"        _v = regs[{ins.b}]")
                lines.append(f"        regs[{ins.a}] = not (_v is None or _v is False)")
            else:
                return None

        lines.extend([
            f"        if _forloop(regs, _loop_ins):",
            f"            used += {cost}",
            "            continue",
            f"        used += {cost}",
            f"        frame.pc = {ir.exit_pc}",
            "        return used, True",
            f"    frame.pc = {ir.start_pc}",
            "    return used, used > 0",
        ])
        namespace = {
            "_Cell": Cell,
            "_LuaTable": LuaTable,
            "_NUM_TYPES": (int, float),
            "_i64": i64,
            "_lua_equal": lua_equal,
            "_isnan": __import__("math").isnan,
            "_forloop": None,
            "_loop_ins": ir.loop_ins,
        }
        # Bind the semantic loop helper once as a local global lookup in the
        # generated function.  It remains the interpreter's exact FORLOOP logic.
        namespace["_forloop"] = lambda regs, ins, _vm_holder=[None]: None
        source = "\n".join(lines)
        compiled_ns = dict(namespace)
        exec(compile(source, "<luapyre-jit-loop>", "exec"), compiled_ns)
        raw_runner = compiled_ns["_jit_loop"]

        def runner(vm, frame, budget, _raw=raw_runner, _ins=ir.loop_ins):
            compiled_ns["_forloop"] = vm._forloop
            compiled_ns["_loop_ins"] = _ins
            return _raw(vm, frame, budget)

        return CompiledLoop(ir, cost, runner)

    def try_loop(self, vm, frame, budget: int) -> tuple[int, bool]:
        if not self.enabled or budget <= 0:
            return 0, False
        start_pc = frame.pc
        backedge = self._loop_map(frame.proto).get(start_pc)
        if backedge is None:
            return 0, False
        key = (id(frame.proto), start_pc)
        cached = self._loop_cache.get(key, DEOPT)
        if cached is DEOPT:
            hot = self._loop_hot.get(key, 0) + 1
            self._loop_hot[key] = hot
            if hot < self.threshold:
                return 0, False
            compiled = self._compile_loop(frame, start_pc, backedge)
            self._loop_cache[key] = compiled
            if compiled is None:
                self.stats.compile_failures += 1
                return 0, False
            self.stats.loop_compiles += 1
            cached = compiled
        if cached is None:
            return 0, False
        used, completed_or_progressed = cached.runner(vm, frame, budget)
        if used:
            self.stats.loop_executions += 1
            self.stats.loop_iterations += used // cached.cost_per_iteration
        if not completed_or_progressed and used == 0:
            self.stats.deopts += 1
        elif frame.pc not in (cached.ir.start_pc, cached.ir.exit_pc):
            self.stats.deopts += 1
        return used, completed_or_progressed

    @staticmethod
    def _leaf_profile(proto: Proto, ins: Ins) -> str | None:
        # Leaf specialization is intentionally conservative because no live
        # frame exists when compilation happens. Typed opcodes carry stronger
        # static information; generic arithmetic keeps a cheap numeric guard.
        if ins.op in (Op.ADD, Op.SUB, Op.MUL, Op.MOD, Op.EQ, Op.LT, Op.LE):
            return "dynamic"
        return None

    def _compile_leaf(self, proto: Proto) -> CompiledLeaf | None:
        sequence: list[IRInstruction] = []
        return_ins: IRInstruction | None = None
        for pc, ins in enumerate(proto.code):
            if ins.op not in _LEAF_OPS:
                return None
            item = IRInstruction(pc, ins, self._leaf_profile(proto, ins))
            sequence.append(item)
            if ins.op is Op.RETURN:
                return_ins = item
                break
        if return_ins is None:
            return None
        trusted = bool(proto.jit_trust_types)
        lines = [
            "def _jit_leaf(frame):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
            "    cells = frame.cells",
        ]
        cost = 0
        for item in sequence:
            ins, op = item.ins, item.ins.op
            cost += 1
            if op is Op.LOADK:
                lines.append(f"    regs[{ins.a}] = consts[{ins.b}]")
            elif op is Op.MOVE:
                lines.append(f"    regs[{ins.a}] = regs[{ins.b}]")
            elif op is Op.LOCAL:
                lines.append(f"    _v = regs[{ins.b}]")
                lines.append(f"    regs[{ins.a}] = _v")
                lines.append(f"    if {ins.a} in cells:")
                lines.append(f"        cells[{ins.a}] = _Cell(_v)")
            elif op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                symbol = {Op.ADD_I: "+", Op.SUB_I: "-", Op.MUL_I: "*"}[op]
                if not trusted:
                    lines.append(
                        f"    if not (type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int): return _DEOPT"
                    )
                lines.append(f"    regs[{ins.a}] = _i64(regs[{ins.b}] {symbol} regs[{ins.c}])")
            elif op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                symbol = {Op.ADD_F: "+", Op.SUB_F: "-", Op.MUL_F: "*"}[op]
                if not trusted:
                    lines.append(
                        f"    if not (type(regs[{ins.b}]) in _NUM_TYPES and type(regs[{ins.c}]) in _NUM_TYPES): return _DEOPT"
                    )
                lines.append(f"    regs[{ins.a}] = float(regs[{ins.b}] {symbol} regs[{ins.c}])")
            elif op in (Op.ADD, Op.SUB, Op.MUL):
                symbol = {Op.ADD: "+", Op.SUB: "-", Op.MUL: "*"}[op]
                lines.append(f"    _a = regs[{ins.b}]; _b = regs[{ins.c}]")
                lines.append("    if type(_a) is int and type(_b) is int:")
                lines.append(f"        regs[{ins.a}] = _i64(_a {symbol} _b)")
                lines.append("    elif type(_a) in _NUM_TYPES and type(_b) in _NUM_TYPES:")
                lines.append(f"        regs[{ins.a}] = _a {symbol} _b")
                lines.append("    else:")
                lines.append("        return _DEOPT")
            elif op is Op.MOD:
                lines.append(f"    _a = regs[{ins.b}]; _b = regs[{ins.c}]")
                lines.append("    if type(_a) is not int or type(_b) is not int or _b == 0: return _DEOPT")
                lines.append(f"    regs[{ins.a}] = _i64(_a % _b)")
            elif op is Op.EQ:
                lines.append(f"    _a = regs[{ins.b}]; _b = regs[{ins.c}]")
                lines.append("    if isinstance(_a, _LuaTable) and isinstance(_b, _LuaTable): return _DEOPT")
                lines.append(f"    regs[{ins.a}] = _lua_equal(_a, _b)")
            elif op in (Op.LT, Op.LE):
                symbol = "<" if op is Op.LT else "<="
                lines.append(f"    _a = regs[{ins.b}]; _b = regs[{ins.c}]")
                lines.append(
                    "    if not ((type(_a) in _NUM_TYPES and type(_b) in _NUM_TYPES) or (isinstance(_a, bytes) and isinstance(_b, bytes))): return _DEOPT"
                )
                lines.append(f"    regs[{ins.a}] = _a {symbol} _b")
            elif op is Op.NOT:
                lines.append(f"    _v = regs[{ins.b}]")
                lines.append(f"    regs[{ins.a}] = (_v is None or _v is False)")
            elif op is Op.TOBOOL:
                lines.append(f"    _v = regs[{ins.b}]")
                lines.append(f"    regs[{ins.a}] = not (_v is None or _v is False)")
            elif op is Op.GUARD:
                lines.append(
                    f"    if not _type_matches(consts[{ins.b}], regs[{ins.a}]): return _DEOPT"
                )
            elif op is Op.RETURN:
                if ins.b <= 0:
                    lines.append("    return ()")
                else:
                    values = ", ".join(f"regs[{ins.a + i}]" for i in range(ins.b))
                    if ins.b == 1:
                        values += ","
                    lines.append(f"    return ({values})")
                break
            else:
                return None
        namespace = {
            "_Cell": Cell,
            "_DEOPT": DEOPT,
            "_LuaTable": LuaTable,
            "_NUM_TYPES": (int, float),
            "_i64": i64,
            "_lua_equal": lua_equal,
            "_type_matches": type_matches,
        }
        exec(compile("\n".join(lines), "<luapyre-jit-leaf>", "exec"), namespace)
        return CompiledLeaf(proto, cost, namespace["_jit_leaf"])

    def maybe_leaf(self, proto: Proto) -> CompiledLeaf | None:
        if not self.enabled:
            return None
        ident = id(proto)
        cached = self._leaf_cache.get(ident)
        if cached is not None and cached[0] is proto:
            return cached[1]
        hot = self._leaf_hot.get(ident, 0) + 1
        self._leaf_hot[ident] = hot
        if hot < self.threshold:
            return None
        compiled = self._compile_leaf(proto)
        self._leaf_cache[ident] = (proto, compiled)
        if compiled is None:
            self.stats.compile_failures += 1
            return None
        self.stats.leaf_compiles += 1
        return compiled
