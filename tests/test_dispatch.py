from luapyre import astnodes as A
from luapyre.compiler import _EXPR_HANDLERS, _STMT_HANDLERS
from luapyre.semantics import _WALK_HANDLERS


def test_compiler_statement_dispatch_covers_every_concrete_statement():
    assert set(_STMT_HANDLERS) == set(A.Stmt.__subclasses__())


def test_compiler_expression_dispatch_covers_every_concrete_expression():
    assert set(_EXPR_HANDLERS) == set(A.Expr.__subclasses__())


def test_control_flow_dispatch_only_contains_statement_types():
    assert _WALK_HANDLERS
    assert all(issubclass(node_type, A.Stmt) for node_type in _WALK_HANDLERS)
