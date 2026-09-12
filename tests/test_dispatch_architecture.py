from __future__ import annotations

import ast
import inspect

from luapyre.binary_chunks import _PUC_TRANSLATORS, _Translator


def test_puc_translator_dispatch_table_covers_all_opcodes():
    assert set(_PUC_TRANSLATORS) == set(range(85))


def test_puc_translate_loop_uses_table_dispatch():
    source = inspect.getsource(_Translator.translate)
    assert "_PUC_TRANSLATORS" in source
    assert "elif opcode" not in source



def test_hot_runtime_opcodes_have_specialized_handlers():
    from luapyre.bytecode import Op
    from luapyre.opdispatch import OPCODE_HANDLERS

    groups = [
        (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.IDIV, Op.MOD, Op.POW),
        (Op.BAND, Op.BOR, Op.BXOR, Op.SHL, Op.SHR),
        (Op.ADD_I, Op.SUB_I, Op.MUL_I),
        (Op.ADD_F, Op.SUB_F, Op.MUL_F),
        (Op.EQ, Op.LT, Op.LE),
        (Op.CALL, Op.CALLV, Op.TAILCALL, Op.TAILCALLV),
    ]
    for group in groups:
        handlers = [OPCODE_HANDLERS[opcode] for opcode in group]
        assert len(set(handlers)) == len(handlers)
        for handler in handlers:
            assert "ins.op" not in inspect.getsource(handler)



def test_gc_opcode_analysis_uses_complete_dispatch_tables():
    from luapyre.bytecode import Op
    from luapyre.gc import LuaGC, _RW_HANDLERS, _SUCCESSOR_HANDLERS

    assert set(_RW_HANDLERS) == set(Op)
    assert set(_SUCCESSOR_HANDLERS) == set(Op)
    assert "elif op" not in inspect.getsource(LuaGC._ins_reads_writes)
    assert "elif op" not in inspect.getsource(LuaGC._successors)



def test_jit_codegen_uses_dense_opcode_tables():
    from luapyre.bytecode import Op
    from luapyre.dense_jit import DenseEmitterJITMixin
    from luapyre.jit_codegen import LEAF_EMITTERS, LOOP_EMITTERS

    assert len(LOOP_EMITTERS) > max(op.value for op in Op)
    assert len(LEAF_EMITTERS) > max(op.value for op in Op)
    for op in (
        Op.LOADK,
        Op.MOVE,
        Op.LOCAL,
        Op.ADD_I,
        Op.SUB_I,
        Op.MUL_I,
        Op.ADD_F,
        Op.SUB_F,
        Op.MUL_F,
        Op.ADD,
        Op.SUB,
        Op.MUL,
        Op.MOD,
        Op.EQ,
        Op.LT,
        Op.LE,
        Op.NOT,
        Op.TOBOOL,
    ):
        assert LOOP_EMITTERS[op.value] is not None
        assert LEAF_EMITTERS[op.value] is not None

    source = inspect.getsource(DenseEmitterJITMixin._compile_loop)
    assert "LOOP_EMITTERS" in source
    assert "item.ins.op" in source
    assert "elif op" not in source



def test_jit_ast_pass_inlines_i64_assignment_once():
    from luapyre.jit_codegen import optimize_generated_ast

    tree = ast.parse("result = _i64(left() + right())")
    optimized = optimize_generated_ast(tree)
    text = ast.unparse(optimized)

    assert "_i64(" not in text
    assert text.count("left()") == 1
    assert text.count("right()") == 1
    assert "_MASK64" in text
    assert "_SIGN64" in text
    assert "_TWO64" in text
