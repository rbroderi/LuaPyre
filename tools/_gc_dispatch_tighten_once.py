from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src/luapyre/gc.py"
WORKFLOW = ROOT / ".github/workflows/_gc-dispatch-tighten-once.yml"
SELF = Path(__file__)

source = TARGET.read_text(encoding="utf-8")
source = source.replace(
    '_CALL_OPS = {Op.CALL, Op.CALLV, Op.TAILCALL, Op.TAILCALLV}\n',
    '',
    1,
)
source = source.replace('_rw_gettable', '_rw_read_bc_write_a')
source = source.replace(
    '_RW_HANDLERS = {op: _rw_empty for op in Op}\n',
    '''_RW_HANDLERS = {\n    Op.JMP: _rw_empty,\n    Op.CLOSE: _rw_empty,\n    Op.PCLOSE: _rw_empty,\n    Op.HALT: _rw_empty,\n}\n''',
    1,
)
source = source.replace(
    '_SUCCESSOR_HANDLERS = {op: _succ_fallthrough for op in Op}\n',
    '''_SUCCESSOR_HANDLERS = {\n    op: _succ_fallthrough\n    for op in (\n        Op.LOADK, Op.MOVE, Op.LOCAL,\n        Op.GETGLOBAL, Op.SETGLOBAL, Op.GETUPVAL, Op.SETUPVAL, Op.GETCELL, Op.SETCELL, Op.CLOSURE,\n        Op.NEWTABLE, Op.GETTABLE, Op.SETTABLE, Op.SETLISTV, Op.LEN,\n        Op.ADD, Op.ADD_I, Op.ADD_F, Op.SUB, Op.SUB_I, Op.SUB_F,\n        Op.MUL, Op.MUL_I, Op.MUL_F, Op.DIV, Op.IDIV, Op.MOD, Op.POW,\n        Op.BAND, Op.BOR, Op.BXOR, Op.SHL, Op.SHR, Op.BNOT, Op.CONCAT, Op.NEG, Op.NOT, Op.TOBOOL,\n        Op.EQ, Op.LT, Op.LE,\n        Op.CALL, Op.CALLV, Op.VARARG, Op.UNPACK,\n        Op.TBC, Op.CLOSE, Op.CHECKNIL, Op.GUARD,\n        Op.PTBC, Op.PCLOSE, Op.PVARARG, Op.PGETVARG,\n    )\n}\n''',
    1,
)
ast.parse(source)
TARGET.write_text(source, encoding="utf-8")
SELF.unlink()
if WORKFLOW.exists():
    WORKFLOW.unlink()
