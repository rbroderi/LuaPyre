from __future__ import annotations

from .lexer import Lexer
from .errors import LuaSyntaxError
from . import astnodes as A
from .typesys import ANY, NIL, LuaType, parse_simple_type, union_of


PRECEDENCE = {
    "or": 1, "and": 2,
    "==": 3, "~=": 3, "<": 3, ">": 3, "<=": 3, ">=": 3,
    "|": 4, "~": 5, "&": 6, "<<": 7, ">>": 7,
    "..": 8, "+": 9, "-": 9, "*": 10, "/": 10, "//": 10, "%": 10, "^": 12,
}
RIGHT_ASSOC = {"^", ".."}


class Parser:
    def __init__(self, source: str):
        self.ts = Lexer(source).tokens()
        self.i = 0

    @property
    def t(self):
        return self.ts[self.i]

    def take(self, kind=None):
        t = self.t
        if kind is not None and t.kind != kind:
            raise LuaSyntaxError(f"expected {kind!r}, got {t.kind!r} at line {t.line}")
        self.i += 1
        return t

    def accept(self, kind):
        if self.t.kind == kind:
            return self.take()
        return None

    def parse(self):
        return A.Chunk(1, self.block({"EOF"}))

    def block(self, stops):
        out = []
        while self.t.kind not in stops:
            out.append(self.statement())
            self.accept(";")
        return out

    def statement(self):
        if self.t.kind == "local":
            return self.local_stmt()
        if self.t.kind == "function":
            return self.function_stmt(False)
        if self.t.kind == "return":
            line = self.take().line
            if self.t.kind in ("EOF", "end", "else", ";"):
                return A.Return(line, None)
            return A.Return(line, self.expr())
        if self.t.kind == "while":
            line = self.take().line
            cond = self.expr()
            self.take("do")
            body = self.block({"end"})
            self.take("end")
            return A.WhileStmt(line, cond, body)
        if self.t.kind == "if":
            line = self.take().line
            cond = self.expr()
            self.take("then")
            then_body = self.block({"else", "end"})
            else_body = []
            if self.accept("else"):
                else_body = self.block({"end"})
            self.take("end")
            return A.IfStmt(line, cond, then_body, else_body)
        expr = self.expr()
        if isinstance(expr, A.Name) and self.accept("="):
            return A.Assign(expr.line, expr.value, self.expr())
        return A.ExprStmt(expr.line, expr)

    def local_stmt(self):
        line = self.take("local").line
        if self.t.kind == "function":
            return self.function_stmt(True, line_override=line)
        name = self.take("NAME").value
        annotation = self.type_annotation() if self.accept(":") else ANY
        value = self.expr() if self.accept("=") else None
        return A.LocalDecl(line, name, annotation, value)

    def function_stmt(self, local, line_override=None):
        line = line_override or self.take("function").line
        if local:
            self.take("function")
        name = self.take("NAME").value
        self.take("(")
        params = []
        if self.t.kind != ")":
            while True:
                pname = self.take("NAME").value
                ptype = self.type_annotation() if self.accept(":") else ANY
                params.append((pname, ptype))
                if not self.accept(","):
                    break
        self.take(")")
        rtype = self.type_annotation() if self.accept(":") else ANY
        body = self.block({"end"})
        self.take("end")
        return A.FunctionDef(line, name, params, rtype, body, local)

    def type_annotation(self) -> LuaType:
        parts = [parse_simple_type(self.take("NAME").value)]
        if self.accept("?"):
            parts.append(NIL)
        while self.accept("|"):
            part = parse_simple_type(self.take("NAME").value)
            if self.accept("?"):
                part = union_of(part, NIL)
            parts.append(part)
        return union_of(*parts)

    def expr(self, min_prec=0):
        left = self.prefix()
        while self.t.kind in PRECEDENCE and PRECEDENCE[self.t.kind] >= min_prec:
            op = self.take().kind
            prec = PRECEDENCE[op]
            right = self.expr(prec if op in RIGHT_ASSOC else prec + 1)
            left = A.Binary(left.line, op, left, right)
        return left

    def prefix(self):
        t = self.t
        if t.kind in ("-", "not", "#", "~"):
            self.take()
            return A.Unary(t.line, t.kind, self.expr(11))
        if t.kind == "NUMBER":
            self.take()
            from .typesys import INTEGER, FLOAT
            typ = INTEGER if type(t.value) is int else FLOAT
            return A.Literal(t.line, t.value, typ)
        if t.kind == "STRING":
            self.take()
            from .typesys import STRING
            return A.Literal(t.line, t.value, STRING)
        if t.kind == "true":
            self.take()
            from .typesys import BOOLEAN
            return A.Literal(t.line, True, BOOLEAN)
        if t.kind == "false":
            self.take()
            from .typesys import BOOLEAN
            return A.Literal(t.line, False, BOOLEAN)
        if t.kind == "nil":
            self.take()
            from .typesys import NIL
            return A.Literal(t.line, None, NIL)
        if t.kind == "NAME":
            self.take()
            node = A.Name(t.line, t.value)
            while self.accept("("):
                args = []
                if self.t.kind != ")":
                    while True:
                        args.append(self.expr())
                        if not self.accept(","):
                            break
                self.take(")")
                node = A.Call(t.line, node, args)
            return node
        if self.accept("("):
            e = self.expr()
            self.take(")")
            return e
        raise LuaSyntaxError(f"expected expression at line {t.line}, got {t.kind!r}")
