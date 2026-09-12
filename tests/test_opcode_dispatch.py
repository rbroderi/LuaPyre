from luapyre.bytecode import Op
from luapyre.opdispatch import OPCODE_HANDLERS


def test_opcode_handler_table_covers_every_opcode_exactly_once():
    assert set(OPCODE_HANDLERS) == set(Op)
    assert len(OPCODE_HANDLERS) == len(Op)
