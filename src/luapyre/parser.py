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
            if self.t.kind in ("EOF", "end", "else", "elseif", ";"):
                return A.Return(line, [])
            return A.Return(line, self.expr_list())
        if self.t.kind == "while":
            line = self.take().line
            cond = self.expr()
            self.take("do")
            body = self.block({"end"})
            self.take("end")
            return A.WhileStmt(line, cond, body)
        if self.t.kind == "if":
            return self.if_stmt()

        first = self.expr()
        if self.t.kind in (",", "="):
            targets = [first]
            while self.accept(","):
                targets.append(self.expr())
            self.take("=")
            self._check_targets(targets)
            return A.Assign(first.line, targets, self.expr_list())
        if not isinstance(first, A.Call):
            raise LuaSyntaxError(f"line {first.line}: statement is neither assignment nor function call")
        return A.ExprStmt(first.line, first)

    def _check_targets(self, targets):
        for target in targets:
            if not isinstance(target, (A.Name, A.Index, A.Field)):
                raise LuaSyntaxError(f"line {target.line}: invalid assignment target")

    def if_stmt(self):
        line = self.take("if").line
        clauses = []
        cond = self.expr()
        self.take("then")
        clauses.append((cond, self.block({"elseif", "else", "end"})))
        while self.accept("elseif"):
            cond = self.expr()
            self.take("then")
            clauses.append((cond, self.block({"elseif", "else", "end"})))
        else_body = []
        if self.accept("else"):
            else_body = self.block({"end"})
        self.take("end")
        return A.IfStmt(line, clauses, else_body)

    def local_stmt(self):
        line = self.take("local").line
        if self.t.kind == "function":
            return self.function_stmt(True, line_override=line)
        names = []
        while True:
            name = self.take("NAME").value
            annotation = self.type_annotation() if self.accept(":") else ANY
            names.append((name, annotation))
            if not self.accept(","):
                break
        values = self.expr_list() if self.accept("=") else []
        return A.LocalDecl(line, names, values)

    def function_stmt(self, local, line_override=None):
        if local:
            line = line_override or self.t.line
            self.take("function")
        else:
            line = self.take("function").line
        name = self.take("NAME").value
        self.take("(")
        params = []
        vararg_name = None
        vararg_type = ANY
        if self.t.kind != ")":
            while True:
                if self.accept("..."):
                    if self.t.kind == "NAME":
                        vararg_name = self.take("NAME").value
                        if self.accept(":"):
                            vararg_type = self.type_annotation()
                    else:
                        vararg_name = ""
                    break
                pname = self.take("NAME").value
                ptype = self.type_annotation() if self.accept(":") else ANY
                params.append((pname, ptype))
                if not self.accept(","):
                    break
        self.take(")")
        return_types = [ANY]
        if self.accept(":"):
            return_types = [self.type_annotation()]
            while self.accept(","):
                return_types.append(self.type_annotation())
        body = self.block({"end"})
        self.take("end")
        return A.FunctionDef(line, name, params, return_types, body, local, vararg_name, vararg_type)

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

    def expr_list(self):
        values = [self.expr()]
        while self.accept(","):
            values.append(self.expr())
        return values

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
        node = self.primary()
        while True:
            if self.accept("["):
                key = self.expr()
                self.take("]")
                node = A.Index(node.line, node, key)
            elif self.accept("."):
                name = self.take("NAME").value
                node = A.Field(node.line, node, name)
            elif self.accept("("):
                args = [] if self.t.kind == ")" else self.expr_list()
                self.take(")")
                node = A.Call(node.line, node, args)
            elif self.t.kind == "STRING":
                s = self.take()
                from .typesys import STRING
                node = A.Call(node.line, node, [A.Literal(s.line, s.value, STRING)])
            elif self.t.kind == "{":
                node = A.Call(node.line, node, [self.table_ctor()])
            else:
                break
        return node

    def primary(self):
        t = self.t
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
        if t.kind == "...":
            self.take()
            return A.VarArg(t.line)
        if t.kind == "NAME":
            self.take()
            return A.Name(t.line, t.value)
        if t.kind == "{":
            return self.table_ctor()
        if self.accept("("):
            e = self.expr()
            self.take(")")
            return e
        raise LuaSyntaxError(f"expected expression at line {t.line}, got {t.kind!r}")

    def table_ctor(self):
        line = self.take("{").line
        fields = []
        while self.t.kind != "}":
            if self.accept("["):
                key = self.expr()
                self.take("]")
                self.take("=")
                fields.append(A.TableField(key, self.expr()))
            elif self.t.kind == "NAME" and self.ts[self.i + 1].kind == "=":
                key = self.take("NAME").value
                self.take("=")
                fields.append(A.TableField(key, self.expr()))
            else:
                fields.append(A.TableField(None, self.expr()))
            if not (self.accept(",") or self.accept(";")):
                break
        self.take("}")
        return A.TableCtor(line, fields)
