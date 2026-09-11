from __future__ import annotations

from dataclasses import dataclass
from . import astnodes as A
from .bytecode import Op, Ins, Proto, Closure
from .errors import LuaTypeError
from .typesys import ANY, BOOLEAN, INTEGER, FLOAT, NUMBER, LuaType, accepts


@dataclass(slots=True)
class Symbol:
    reg: int
    typ: LuaType


class Compiler:
    def compile(self, chunk: A.Chunk) -> Proto:
        proto = Proto("<chunk>")
        ctx = _FunctionCompiler(proto)
        ctx.compile_block(chunk.body)
        proto.code.append(Ins(Op.HALT))
        proto.register_count = ctx.max_reg
        return proto


class _FunctionCompiler:
    def __init__(self, proto: Proto, params=None):
        self.proto = proto
        self.symbols: dict[str, Symbol] = {}
        self.next_reg = 0
        self.max_reg = 0
        for name, typ in params or []:
            r = self.alloc()
            self.symbols[name] = Symbol(r, typ)

    def alloc(self):
        r = self.next_reg
        self.next_reg += 1
        self.max_reg = max(self.max_reg, self.next_reg)
        return r

    def emit(self, op, a=0, b=0, c=0):
        self.proto.code.append(Ins(op, a, b, c))
        return len(self.proto.code) - 1

    def patch(self, at, target):
        ins = self.proto.code[at]
        self.proto.code[at] = Ins(ins.op, target, ins.b, ins.c)

    def compile_block(self, body):
        for stmt in body:
            self.stmt(stmt)

    def stmt(self, stmt):
        if isinstance(stmt, A.LocalDecl):
            r = self.alloc()
            actual = ANY
            if stmt.value is not None:
                vr, actual = self.expr(stmt.value)
                self.emit(Op.MOVE, r, vr)
            if stmt.annotation is not ANY and actual is not ANY and not accepts(stmt.annotation, actual):
                raise LuaTypeError(f"line {stmt.line}: cannot assign {actual} to {stmt.annotation}")
            if stmt.annotation is not ANY and actual is ANY and stmt.value is not None:
                self.emit(Op.GUARD, r, self.proto.add_const(stmt.annotation.name))
            self.symbols[stmt.name] = Symbol(r, stmt.annotation)
            return

        if isinstance(stmt, A.Assign):
            vr, actual = self.expr(stmt.value)
            sym = self.symbols.get(stmt.name)
            if sym:
                if sym.typ is not ANY and actual is not ANY and not accepts(sym.typ, actual):
                    raise LuaTypeError(f"line {stmt.line}: cannot assign {actual} to {sym.typ}")
                if sym.typ is not ANY and actual is ANY:
                    self.emit(Op.GUARD, vr, self.proto.add_const(sym.typ.name))
                self.emit(Op.MOVE, sym.reg, vr)
            else:
                self.emit(Op.SETGLOBAL, vr, self.proto.add_const(stmt.name))
            return

        if isinstance(stmt, A.Return):
            if stmt.value is None:
                self.emit(Op.RETURN, 0, 0)
            else:
                r, typ = self.expr(stmt.value)
                if self.proto.return_type is not ANY and typ is not ANY and not accepts(self.proto.return_type, typ):
                    raise LuaTypeError(f"line {stmt.line}: cannot return {typ} as {self.proto.return_type}")
                if self.proto.return_type is not ANY and typ is ANY:
                    self.emit(Op.GUARD, r, self.proto.add_const(self.proto.return_type.name))
                self.emit(Op.RETURN, r, 1)
            return

        if isinstance(stmt, A.ExprStmt):
            self.expr(stmt.expr)
            return

        if isinstance(stmt, A.WhileStmt):
            start = len(self.proto.code)
            cr, _ = self.expr(stmt.condition)
            jump_false = self.emit(Op.JMPIFNOT, 0, cr)
            self.compile_block(stmt.body)
            self.emit(Op.JMP, start)
            self.patch(jump_false, len(self.proto.code))
            return

        if isinstance(stmt, A.IfStmt):
            cr, _ = self.expr(stmt.condition)
            jump_false = self.emit(Op.JMPIFNOT, 0, cr)
            self.compile_block(stmt.then_body)
            if stmt.else_body:
                jump_end = self.emit(Op.JMP, 0)
                self.patch(jump_false, len(self.proto.code))
                self.compile_block(stmt.else_body)
                self.patch(jump_end, len(self.proto.code))
            else:
                self.patch(jump_false, len(self.proto.code))
            return

        if isinstance(stmt, A.FunctionDef):
            child = Proto(
                stmt.name,
                param_count=len(stmt.params),
                param_types=[typ for _, typ in stmt.params],
                return_type=stmt.return_type,
            )
            sub = _FunctionCompiler(child, stmt.params)
            sub.compile_block(stmt.body)
            child.code.append(Ins(Op.RETURN, 0, 0))
            child.register_count = sub.max_reg
            self.proto.children.append(child)
            closure_const = self.proto.add_const(Closure(child))
            r = self.alloc()
            self.emit(Op.LOADK, r, closure_const)
            if stmt.local:
                self.symbols[stmt.name] = Symbol(r, ANY)
            else:
                self.emit(Op.SETGLOBAL, r, self.proto.add_const(stmt.name))
            return

        raise NotImplementedError(type(stmt).__name__)

    def expr(self, expr):
        if isinstance(expr, A.Literal):
            r = self.alloc()
            self.emit(Op.LOADK, r, self.proto.add_const(expr.value))
            return r, expr.inferred_type

        if isinstance(expr, A.Name):
            sym = self.symbols.get(expr.value)
            if sym:
                return sym.reg, sym.typ
            r = self.alloc()
            self.emit(Op.GETGLOBAL, r, self.proto.add_const(expr.value))
            return r, ANY

        if isinstance(expr, A.Unary):
            x, typ = self.expr(expr.operand)
            out = self.alloc()
            op = {"-": Op.NEG, "not": Op.NOT}.get(expr.op)
            if op is None:
                raise NotImplementedError(expr.op)
            self.emit(op, out, x)
            return out, (BOOLEAN if expr.op == "not" else typ)

        if isinstance(expr, A.Binary):
            left, left_type = self.expr(expr.left)
            right, right_type = self.expr(expr.right)
            out = self.alloc()

            if expr.op in ("+", "-", "*"):
                generic = {"+": Op.ADD, "-": Op.SUB, "*": Op.MUL}[expr.op]
                if left_type is INTEGER and right_type is INTEGER:
                    op = {"+": Op.ADD_I, "-": Op.SUB_I, "*": Op.MUL_I}[expr.op]
                    result_type = INTEGER
                elif left_type in (INTEGER, FLOAT) and right_type in (INTEGER, FLOAT) and (
                    left_type is FLOAT or right_type is FLOAT
                ):
                    op = {"+": Op.ADD_F, "-": Op.SUB_F, "*": Op.MUL_F}[expr.op]
                    result_type = FLOAT
                else:
                    op = generic
                    result_type = ANY if ANY in (left_type, right_type) else NUMBER
            elif expr.op == "/":
                op, result_type = Op.DIV, FLOAT
            elif expr.op == "//":
                op, result_type = Op.IDIV, INTEGER
            elif expr.op == "%":
                op = Op.MOD
                result_type = left_type if left_type == right_type else NUMBER
            elif expr.op == "^":
                op, result_type = Op.POW, FLOAT
            elif expr.op in ("==", "~="):
                op, result_type = Op.EQ, BOOLEAN
            elif expr.op in ("<", ">", "<=", ">="):
                op = Op.LT if expr.op in ("<", ">") else Op.LE
                result_type = BOOLEAN
            else:
                raise NotImplementedError(expr.op)

            self.emit(op, out, left, right)
            if expr.op == "~=":
                negated = self.alloc()
                self.emit(Op.NOT, negated, out)
                return negated, BOOLEAN
            if expr.op == ">":
                self.proto.code[-1] = Ins(Op.LT, out, right, left)
            elif expr.op == ">=":
                self.proto.code[-1] = Ins(Op.LE, out, right, left)
            return out, result_type

        if isinstance(expr, A.Call):
            fn_reg, _ = self.expr(expr.func)
            arg_regs = []
            for arg in expr.args:
                arg_reg, _ = self.expr(arg)
                arg_regs.append(arg_reg)

            base = self.alloc()
            for i, arg_reg in enumerate(arg_regs):
                dest = base + i
                while self.next_reg <= dest:
                    self.alloc()
                self.emit(Op.MOVE, dest, arg_reg)

            out = self.alloc()
            self.emit(Op.CALL, out, fn_reg, base)
            self.proto.code.append(Ins(Op.LOADK, self.alloc(), self.proto.add_const(len(arg_regs))))
            return out, ANY

        raise NotImplementedError(type(expr).__name__)
