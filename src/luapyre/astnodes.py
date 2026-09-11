from __future__ import annotations
from dataclasses import dataclass, field
from .typesys import LuaType, ANY


@dataclass(slots=True)
class Node:
    line: int


@dataclass(slots=True)
class Chunk(Node):
    body: list[Stmt]


class Stmt(Node):
    pass


class Expr(Node):
    inferred_type: LuaType = ANY


@dataclass(frozen=True, slots=True)
class DeclaredName:
    name: str
    typ: LuaType = ANY
    attribute: str | None = None


@dataclass(slots=True)
class LocalDecl(Stmt):
    names: list[DeclaredName]
    values: list[Expr]


@dataclass(slots=True)
class GlobalDecl(Stmt):
    names: list[DeclaredName]
    values: list[Expr]
    wildcard: bool = False
    wildcard_attribute: str | None = None


@dataclass(slots=True)
class GlobalFunctionDef(Stmt):
    name: str
    params: list[tuple[str, LuaType]]
    return_types: list[LuaType]
    body: list[Stmt]
    vararg_name: str | None = None
    vararg_type: LuaType = ANY


@dataclass(slots=True)
class Assign(Stmt):
    targets: list[Expr]
    values: list[Expr]


@dataclass(slots=True)
class Return(Stmt):
    values: list[Expr]


@dataclass(slots=True)
class ExprStmt(Stmt):
    expr: Expr


@dataclass(slots=True)
class WhileStmt(Stmt):
    condition: Expr
    body: list[Stmt]


@dataclass(slots=True)
class RepeatStmt(Stmt):
    body: list[Stmt]
    condition: Expr


@dataclass(slots=True)
class DoStmt(Stmt):
    body: list[Stmt]


@dataclass(slots=True)
class NumericForStmt(Stmt):
    name: str
    start: Expr
    limit: Expr
    step: Expr | None
    body: list[Stmt]


@dataclass(slots=True)
class GenericForStmt(Stmt):
    names: list[str]
    values: list[Expr]
    body: list[Stmt]


@dataclass(slots=True)
class BreakStmt(Stmt):
    pass


@dataclass(slots=True)
class GotoStmt(Stmt):
    name: str
    target_id: int = -1
    close_depth: int = 0


@dataclass(slots=True)
class LabelStmt(Stmt):
    name: str
    label_id: int = -1
    close_depth: int = 0


@dataclass(slots=True)
class IfStmt(Stmt):
    clauses: list[tuple[Expr, list[Stmt]]]
    else_body: list[Stmt]


@dataclass(slots=True)
class FunctionDef(Stmt):
    name: str
    params: list[tuple[str, LuaType]]
    return_types: list[LuaType]
    body: list[Stmt]
    local: bool = False
    vararg_name: str | None = None
    vararg_type: LuaType = ANY


@dataclass(slots=True)
class Literal(Expr):
    value: object
    inferred_type: LuaType = field(default=ANY)


@dataclass(slots=True)
class Name(Expr):
    value: str
    inferred_type: LuaType = field(default=ANY)


@dataclass(slots=True)
class VarArg(Expr):
    inferred_type: LuaType = field(default=ANY)


@dataclass(slots=True)
class Unary(Expr):
    op: str
    operand: Expr
    inferred_type: LuaType = field(default=ANY)


@dataclass(slots=True)
class Binary(Expr):
    op: str
    left: Expr
    right: Expr
    inferred_type: LuaType = field(default=ANY)


@dataclass(slots=True)
class Call(Expr):
    func: Expr
    args: list[Expr]
    inferred_type: LuaType = field(default=ANY)


@dataclass(slots=True)
class MethodCall(Expr):
    receiver: Expr
    name: str
    args: list[Expr]
    inferred_type: LuaType = field(default=ANY)


@dataclass(slots=True)
class FunctionExpr(Expr):
    params: list[tuple[str, LuaType]]
    return_types: list[LuaType]
    body: list[Stmt]
    vararg_name: str | None = None
    vararg_type: LuaType = ANY
    inferred_type: LuaType = field(default=ANY)


@dataclass(slots=True)
class Index(Expr):
    table: Expr
    key: Expr
    inferred_type: LuaType = field(default=ANY)


@dataclass(slots=True)
class Field(Expr):
    table: Expr
    name: str
    inferred_type: LuaType = field(default=ANY)


@dataclass(slots=True)
class TableField:
    key: Expr | str | None
    value: Expr


@dataclass(slots=True)
class TableCtor(Expr):
    fields: list[TableField]
    inferred_type: LuaType = field(default=ANY)
