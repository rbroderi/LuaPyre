from __future__ import annotations

from .lexer import Lexer
from .errors import LuaSyntaxError
from . import astnodes as A
from .typesys import ANY, NIL, FUNCTION, LuaType, parse_simple_type, union_of


PRECEDENCE = {
    "or": 1, "and": 2,
    "==": 3, "~=": 3, "<": 3, ">": 3, "<=": 3, ">=": 3,
    "|": 4, "~": 5, "&": 6, "<<": 7, ">>": 7,
    "..": 8, "+": 9, "-": 9, "*": 10, "/": 10, "//": 10, "%": 10, "^": 12,
}
RIGHT_ASSOC = {"^", ".."}
_VALID_ATTRIBUTES = {"const", "close"}


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
        return self.take() if self.t.kind == kind else None

    def parse(self):
        return A.Chunk(1, self.block({"EOF"}))

    def block(self, stops):
        out = []
        while self.t.kind not in stops:
            if self.accept(";"):
                continue
            out.append(self.statement())
            self.accept(";")
        return out

    def statement(self):
        kind = self.t.kind
        if kind == "local":
            return self.local_stmt()
        if kind == "global":
            return self.global_stmt()
        if kind == "function":
            return self.function_stmt(False)
        if kind == "goto":
            line = self.take().line
            return A.GotoStmt(line, self.take("NAME").value)
        if kind == "::":
            line = self.take().line
            name = self.take("NAME").value
            self.take("::")
            return A.LabelStmt(line, name)
        if kind == "return":
            line = self.take().line
            if self.t.kind in ("EOF", "end", "else", "elseif", "until", ";"):
                return A.Return(line, [])
            return A.Return(line, self.expr_list())
        if kind == "break":
            return A.BreakStmt(self.take().line)
        if kind == "do":
            line = self.take().line
            body = self.block({"end"})
            self.take("end")
            return A.DoStmt(line, body)
        if kind == "while":
            line = self.take().line
            cond = self.expr()
            self.take("do")
            body = self.block({"end"})
            self.take("end")
            return A.WhileStmt(line, cond, body)
        if kind == "repeat":
            line = self.take().line
            body = self.block({"until"})
            self.take("until")
            return A.RepeatStmt(line, body, self.expr())
        if kind == "for":
            return self.for_stmt()
        if kind == "if":
            return self.if_stmt()

        first = self.expr()
        if self.t.kind in (",", "="):
            targets = [first]
            while self.accept(","):
                targets.append(self.expr())
            self.take("=")
            self._check_targets(targets)
            return A.Assign(first.line, targets, self.expr_list())
        if not isinstance(first, (A.Call, A.MethodCall)):
            raise LuaSyntaxError(f"line {first.line}: statement is neither assignment nor function call")
        return A.ExprStmt(first.line, first)

    def _check_targets(self, targets):
        for target in targets:
            if not isinstance(target, (A.Name, A.Index, A.Field)):
                raise LuaSyntaxError(f"line {target.line}: invalid assignment target")

    def _attribute(self):
        if not self.accept("<"):
            return None
        token = self.take("NAME")
        self.take(">")
        if token.value not in _VALID_ATTRIBUTES:
            raise LuaSyntaxError(f"line {token.line}: unknown variable attribute '{token.value}'")
        return token.value

    def _declared_names(self, *, allow_close: bool):
        prefix = self._attribute()
        if prefix == "close" and not allow_close:
            raise LuaSyntaxError(f"line {self.t.line}: global variables cannot be to-be-closed")
        names = []
        close_count = 0
        while True:
            token = self.take("NAME")
            typ = self.type_annotation() if self.accept(":") else ANY
            postfix = self._attribute()
            if postfix == "close" and not allow_close:
                raise LuaSyntaxError(f"line {token.line}: global variables cannot be to-be-closed")
            if prefix is not None and postfix is not None and prefix != postfix:
                raise LuaSyntaxError(f"line {token.line}: conflicting variable attributes")
            attribute = postfix or prefix
            if attribute == "close":
                close_count += 1
            names.append(A.DeclaredName(token.value, typ, attribute))
            if not self.accept(","):
                break
        if close_count > 1:
            raise LuaSyntaxError("a declaration can contain at most one to-be-closed variable")
        return names

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

    def for_stmt(self):
        line = self.take("for").line
        name = self.take("NAME").value
        if self.accept("="):
            start = self.expr()
            self.take(",")
            limit = self.expr()
            step = self.expr() if self.accept(",") else None
            self.take("do")
            body = self.block({"end"})
            self.take("end")
            return A.NumericForStmt(line, name, start, limit, step, body)
        names = [name]
        while self.accept(","):
            names.append(self.take("NAME").value)
        self.take("in")
        values = self.expr_list()
        self.take("do")
        body = self.block({"end"})
        self.take("end")
        return A.GenericForStmt(line, names, values, body)

    def local_stmt(self):
        line = self.take("local").line
        if self.t.kind == "function":
            return self.function_stmt(True, line_override=line)
        names = self._declared_names(allow_close=True)
        values = self.expr_list() if self.accept("=") else []
        return A.LocalDecl(line, names, values)

    def global_stmt(self):
        line = self.take("global").line
        if self.t.kind == "function":
            self.take("function")
            name = self.take("NAME").value
            params, returns, body, vararg_name, vararg_type = self.function_body()
            return A.GlobalFunctionDef(
                line, name, params, returns, body, vararg_name, vararg_type
            )

        prefix = self._attribute()
        if prefix == "close":
            raise LuaSyntaxError(f"line {line}: global variables cannot be to-be-closed")
        if self.accept("*"):
            return A.GlobalDecl(line, [], [], True, prefix)

        names = []
        while True:
            token = self.take("NAME")
            typ = self.type_annotation() if self.accept(":") else ANY
            postfix = self._attribute()
            if postfix == "close":
                raise LuaSyntaxError(f"line {token.line}: global variables cannot be to-be-closed")
            if prefix is not None and postfix is not None and prefix != postfix:
                raise LuaSyntaxError(f"line {token.line}: conflicting variable attributes")
            names.append(A.DeclaredName(token.value, typ, postfix or prefix))
            if not self.accept(","):
                break
        values = self.expr_list() if self.accept("=") else []
        return A.GlobalDecl(line, names, values)

    def function_stmt(self, local, line_override=None):
        if local:
            line = line_override or self.t.line
            self.take("function")
            name = self.take("NAME").value
            params, returns, body, vararg_name, vararg_type = self.function_body()
            return A.FunctionDef(line, name, params, returns, body, True, vararg_name, vararg_type)

        line = self.take("function").line
        target = A.Name(line, self.take("NAME").value)
        complex_target = False
        method = False
        while self.accept("."):
            target = A.Field(line, target, self.take("NAME").value)
            complex_target = True
        if self.accept(":"):
            target = A.Field(line, target, self.take("NAME").value)
            complex_target = True
            method = True
        params, returns, body, vararg_name, vararg_type = self.function_body(prepend_self=method)
        if not complex_target and isinstance(target, A.Name):
            return A.FunctionDef(line, target.value, params, returns, body, False, vararg_name, vararg_type)
        fn = A.FunctionExpr(line, params, returns, body, vararg_name, vararg_type, FUNCTION)
        return A.Assign(line, [target], [fn])

    def function_body(self, prepend_self=False):
        self.take("(")
        params = []
        vararg_name = None
        vararg_type = ANY
        if prepend_self:
            params.append(("self", ANY))
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
        return params, return_types, body, vararg_name, vararg_type

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
                node = A.Field(node.line, node, self.take("NAME").value)
            elif self.accept(":"):
                name = self.take("NAME").value
                node = A.MethodCall(node.line, node, name, self.call_args())
            elif self.t.kind in ("(", "STRING", "{"):
                node = A.Call(node.line, node, self.call_args())
            else:
                break
        return node

    def call_args(self):
        if self.accept("("):
            args = [] if self.t.kind == ")" else self.expr_list()
            self.take(")")
            return args
        if self.t.kind == "STRING":
            s = self.take()
            from .typesys import STRING
            return [A.Literal(s.line, s.value, STRING)]
        if self.t.kind == "{":
            return [self.table_ctor()]
        raise LuaSyntaxError(f"expected function arguments at line {self.t.line}")

    def primary(self):
        t = self.t
        if t.kind == "NUMBER":
            self.take()
            from .typesys import INTEGER, FLOAT
            return A.Literal(t.line, t.value, INTEGER if type(t.value) is int else FLOAT)
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
            return A.Literal(t.line, None, NIL)
        if t.kind == "...":
            self.take()
            return A.VarArg(t.line)
        if t.kind == "NAME":
            self.take()
            return A.Name(t.line, t.value)
        if t.kind == "function":
            line = self.take().line
            params, returns, body, vararg_name, vararg_type = self.function_body()
            return A.FunctionExpr(line, params, returns, body, vararg_name, vararg_type, FUNCTION)
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
