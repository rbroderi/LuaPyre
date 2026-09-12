from __future__ import annotations

import ast

from .bytecode import Cell, Op, Proto
from .jit import CompiledLeaf, CompiledLoop, DEOPT, IRInstruction
from .jit_codegen import (
    LEAF_EMITTERS,
    LOOP_EMITTERS,
    generated_namespace,
    optimize_generated_ast,
)
from .table import LuaTable
from .values import i64, lua_equal, type_matches


_INT_MIN = -(1 << 63)
_INT_MAX = (1 << 63) - 1


class DenseEmitterJITMixin:
    """Dense compiler-side dispatch plus AST helper inlining.

    The opcode lookup here happens only while compiling a hot region.  A dense
    tuple indexed by ``Op.value`` replaces the long emitter ``if/elif`` cascade.
    The selected emitter writes the specialized Python fast path, then an AST
    pass removes tiny runtime helpers such as ``_i64`` before CPython compiles
    the generated function.
    """

    @staticmethod
    def _emitter(table, op: Op):
        value = op.value
        if value >= len(table):
            return None
        return table[value]

    def _compile_loop(self, frame, start_pc: int, backedge_pc: int):
        # Let the dense backend own the same straight-line subset as the legacy
        # PythonJIT implementation.  Branch/super-region shapes deliberately
        # continue through the MRO to AstPythonJIT/RegionPythonJIT.
        ir = self._lower_loop(frame, start_pc, backedge_pc)
        if ir is None:
            return super()._compile_loop(frame, start_pc, backedge_pc)

        trusted = bool(frame.proto.jit_trust_types)
        body = ir.body.instructions
        cost = len(body) + 1
        lines = [
            "def _jit_loop(vm, frame, budget):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
            "    cells = frame.cells",
            "    used = 0",
            f"    while budget - used >= {cost}:",
        ]

        for offset, item in enumerate(body):
            emitter = self._emitter(LOOP_EMITTERS, item.ins.op)
            if emitter is None or not emitter(lines, item, offset, trusted):
                return super()._compile_loop(frame, start_pc, backedge_pc)

        # FORPREP has already normalized numeric loop registers before this hot
        # loop can be compiled. Emit VM._forloop's exact mechanics directly so
        # the backedge no longer pays a Python bound-method call every iteration.
        loop_ins = ir.loop_ins
        idx, limit, step = loop_ins.a, loop_ins.b, loop_ins.c
        lines.extend(
            [
                f"        _idx = regs[{idx}]",
                f"        _limit = regs[{limit}]",
                f"        _step = regs[{step}]",
                "        if type(_idx) is int and type(_limit) is int and type(_step) is int:",
                "            _next = _idx + _step",
                "            if _next >= _INT_MIN and _next <= _INT_MAX and not ((_step > 0 and _next > _limit) or (_step < 0 and _next < _limit)):",
                f"                regs[{idx}] = _next",
                f"                used += {cost}",
                "                continue",
                "        else:",
                "            _next = float(_idx) + float(_step)",
                "            if not ((_step > 0 and _next > _limit) or (_step < 0 and _next < _limit)):",
                f"                regs[{idx}] = _next",
                f"                used += {cost}",
                "                continue",
                f"        used += {cost}",
                f"        frame.pc = {ir.exit_pc}",
                "        return used, True",
                f"    frame.pc = {ir.start_pc}",
                "    return used, used > 0",
            ]
        )
        namespace = generated_namespace(
            {
                "_Cell": Cell,
                "_LuaTable": LuaTable,
                "_NUM_TYPES": (int, float),
                "_INT_MIN": _INT_MIN,
                "_INT_MAX": _INT_MAX,
                # Kept as a safety fallback for helper forms the optimizer does
                # not yet recognize. Direct assignment calls are AST-inlined.
                "_i64": i64,
                "_lua_equal": lua_equal,
                "_isnan": __import__("math").isnan,
            }
        )
        tree = optimize_generated_ast(ast.parse("\n".join(lines)))
        exec(compile(tree, "<luapyre-dense-jit-loop>", "exec"), namespace)
        return CompiledLoop(ir, cost, namespace["_jit_loop"])

    def _compile_leaf(self, proto: Proto):
        sequence: list[IRInstruction] = []
        return_ins: IRInstruction | None = None
        for pc, ins in enumerate(proto.code):
            if ins.op not in self._leaf_ops_for_dense_codegen():
                return super()._compile_leaf(proto)
            item = IRInstruction(pc, ins, self._leaf_profile(proto, ins))
            sequence.append(item)
            if ins.op is Op.RETURN:
                return_ins = item
                break
        if return_ins is None:
            return super()._compile_leaf(proto)

        trusted = bool(proto.jit_trust_types)
        lines = [
            "def _jit_leaf(frame):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
            "    cells = frame.cells",
        ]
        cost = 0
        for offset, item in enumerate(sequence):
            emitter = self._emitter(LEAF_EMITTERS, item.ins.op)
            if emitter is None or not emitter(lines, item, offset, trusted):
                return super()._compile_leaf(proto)
            cost += 1
            if item.ins.op is Op.RETURN:
                break

        namespace = generated_namespace(
            {
                "_Cell": Cell,
                "_DEOPT": DEOPT,
                "_LuaTable": LuaTable,
                "_NUM_TYPES": (int, float),
                "_i64": i64,
                "_lua_equal": lua_equal,
                "_type_matches": type_matches,
            }
        )
        tree = optimize_generated_ast(ast.parse("\n".join(lines)))
        exec(compile(tree, "<luapyre-dense-jit-leaf>", "exec"), namespace)
        return CompiledLeaf(proto, cost, namespace["_jit_leaf"])

    @staticmethod
    def _leaf_ops_for_dense_codegen():
        # Keep this explicit rather than reaching into jit.py's private set. It
        # also makes table coverage testable without coupling modules together.
        return _DENSE_LEAF_OPS


_DENSE_LEAF_OPS = frozenset(
    {
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
    }
)
