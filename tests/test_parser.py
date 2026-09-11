from luapyre.parser import Parser
from luapyre import astnodes as A


def test_named_vararg_parser():
    chunk = Parser('local function f(a: integer, ...rest: string): integer return a end').parse()
    fn = chunk.body[0]
    assert isinstance(fn, A.FunctionDef)
    assert fn.vararg_name == "rest"
    assert fn.vararg_type.name == "string"


def test_table_constructor_forms():
    chunk = Parser('local t = {1, name = "x", [2] = 3}').parse()
    decl = chunk.body[0]
    assert isinstance(decl.values[0], A.TableCtor)
    assert len(decl.values[0].fields) == 3
