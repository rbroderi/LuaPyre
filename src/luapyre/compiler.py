from __future__ import annotations

from dataclasses import dataclass

from . import astnodes as A
from .bytecode import Op, Ins, Proto, UpvalueDesc
from .errors import LuaSyntaxError, LuaTypeError
from .typesys import (
    ANY, BOOLEAN, FLOAT, FUNCTION, INTEGER, NUMBER, STRING, TABLE, LuaType, accepts,
)


@dataclass(slots=True)
class Symbol:
    reg: int
    typ: LuaType
    captured: bool = False
    readonly: bool = False
    returns: list[LuaType] | None = None


@dataclass(frozen=True, slots=True)
class Ref:
    kind: str
    index: int
    typ: LuaType
    symbol: Symbol | None = None


class Compiler:
    def compile(self, chunk: A.Chunk) -> Proto:
        proto = Proto("<chunk>")
        ctx = _FunctionCompiler(proto)
        env_reg = ctx.alloc()
        proto.env_reg = env_reg
        ctx.define("_ENV", Symbol(env_reg, TABLE))
        ctx.compile_block(chunk.body, scoped=False)
        proto.code.append(Ins(Op.HALT))
        proto.register_count = ctx.max_reg
        return proto


class _FunctionCompiler:
    def __init__(self, proto: Proto, params=None, parent: _FunctionCompiler | None = None):
        self.proto = proto
        self.parent = parent
        self.scopes: list[dict[str, Symbol]] = [{}]
        self.upvalue_by_name: dict[str, int] = {}
        self.upvalue_types: list[LuaType] = []
        self.next_reg = 0
        self.max_reg = 0
        for name, typ in params or []:
            r = self.alloc()
            self.scopes[0][name] = Symbol(r, typ)

    def alloc(self):
        r = self.next_reg
        self.next_reg += 1
        self.max_reg = max(self.max_reg, self.next_reg)
        return r

    def alloc_n(self, n: int):
        if n <= 0:
            return self.alloc()
        base = self.next_reg
        for _ in range(n):
            self.alloc()
        return base

    def emit(self, op, a=0, b=0, c=0, d=0, e=0):
        self.proto.code.append(Ins(op, a, b, c, d, e))
        return len(self.proto.code) - 1

    def patch_a(self, at, target):
        ins = self.proto.code[at]
        self.proto.code[at] = Ins(ins.op, target, ins.b, ins.c, ins.d, ins.e)

    def push_scope(self):
        self.scopes.append({})

    def pop_scope(self):
        self.scopes.pop()

    def define(self, name: str, sym: Symbol):
        self.scopes[-1][name] = sym

    def find_local(self, name: str) -> Symbol | None:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def ensure_upvalue(self, name: str) -> int | None:
        if name in self.upvalue_by_name:
            return self.upvalue_by_name[name]
        if self.parent is None:
            return None
        source = self.parent.capture_for_child(name)
        if source is None:
            return None
        kind, index, typ = source
        upidx = len(self.proto.upvalues)
        self.proto.upvalues.append(UpvalueDesc(kind, index, name))
        self.upvalue_types.append(typ)
        self.upvalue_by_name[name] = upidx
        return upidx

    def capture_for_child(self, name: str):
        sym = self.find_local(name)
        if sym is not None:
            sym.captured = True
            return "local", sym.reg, sym.typ
        upidx = self.ensure_upvalue(name)
        if upidx is not None:
            return "upvalue", upidx, self.upvalue_types[upidx]
        return None

    def resolve(self, name: str) -> Ref | None:
        sym = self.find_local(name)
        if sym is not None:
            return Ref("local", sym.reg, sym.typ, sym)
        upidx = self.ensure_upvalue(name)
        if upidx is not None:
            return Ref("upvalue", upidx, self.upvalue_types[upidx])
        return None

    def compile_block(self, body, *, scoped=True):
        if scoped:
            self.push_scope()
        try:
            for stmt in body:
                self.stmt(stmt)
        finally:
            if scoped:
                self.pop_scope()

    def nil_reg(self):
        r = self.alloc()
        self.emit(Op.LOADK, r, self.proto.add_const(None))
        return r

    def _load_ref(self, ref: Ref):
        if ref.kind == "local":
            assert ref.symbol is not None
            if ref.symbol.captured:
                out = self.alloc()
                self.emit(Op.GETCELL, out, ref.index)
                return out, ref.typ
            return ref.index, ref.typ
        out = self.alloc()
        self.emit(Op.GETUPVAL, out, ref.index)
        return out, ref.typ

    def _store_ref(self, ref: Ref, value_reg: int, line: int, actual: LuaType = ANY):
        if ref.symbol is not None and ref.symbol.readonly:
            raise LuaTypeError(f"line {line}: cannot assign to read-only local")
        if ref.typ is not ANY and actual is not ANY and not accepts(ref.typ, actual):
            raise LuaTypeError(f"line {line}: cannot assign {actual} to {ref.typ}")
        if ref.typ is not ANY and actual is ANY:
            self.emit(Op.GUARD, value_reg, self.proto.add_const(ref.typ.name))
        if ref.kind == "local":
            assert ref.symbol is not None
            if ref.symbol.captured:
                self.emit(Op.SETCELL, ref.index, value_reg)
            else:
                self.emit(Op.MOVE, ref.index, value_reg)
        else:
            self.emit(Op.SETUPVAL, ref.index, value_reg)

    def _env_reg(self):
        ref = self.resolve("_ENV")
        if ref is None:
            return None
        return self._load_ref(ref)[0]

    def _global_get(self, name: str):
        out = self.alloc()
        env = self._env_reg()
        key = self.proto.add_const(name.encode("utf-8"))
        if env is None:
            self.emit(Op.GETGLOBAL, out, key)
        else:
            kr = self.alloc()
            self.emit(Op.LOADK, kr, key)
            self.emit(Op.GETTABLE, out, env, kr)
        return out, ANY

    def _global_set(self, name: str, value_reg: int):
        env = self._env_reg()
        key = self.proto.add_const(name.encode("utf-8"))
        if env is None:
            self.emit(Op.SETGLOBAL, value_reg, key)
        else:
            kr = self.alloc()
            self.emit(Op.LOADK, kr, key)
            self.emit(Op.SETTABLE, env, kr, value_reg)

    def stmt(self, stmt):
        if isinstance(stmt, A.LocalDecl):
            values = self.adjust_values(stmt.values, len(stmt.names))
            for i, (name, typ) in enumerate(stmt.names):
                r = self.alloc()
                vr, actual = values[i]
                if typ is not ANY and actual is not ANY and not accepts(typ, actual):
                    raise LuaTypeError(f"line {stmt.line}: cannot assign {actual} to {typ}")
                if typ is not ANY and actual is ANY:
                    self.emit(Op.GUARD, vr, self.proto.add_const(typ.name))
                self.emit(Op.LOCAL, r, vr)
                self.define(name, Symbol(r, typ))
            return

        if isinstance(stmt, A.Assign):
            prepared = [self.prepare_target(t) for t in stmt.targets]
            values = self.adjust_values(stmt.values, len(stmt.targets))
            for target, (vr, actual) in zip(prepared, values):
                self.assign_prepared(target, vr, actual, stmt.line)
            return

        if isinstance(stmt, A.Return):
            self.compile_return(stmt)
            return

        if isinstance(stmt, A.ExprStmt):
            if isinstance(stmt.expr, A.Call):
                self.call_expr(stmt.expr, want=0)
            else:
                self.expr(stmt.expr)
            return

        if isinstance(stmt, A.WhileStmt):
            start = len(self.proto.code)
            cr, _ = self.expr(stmt.condition)
            jump_false = self.emit(Op.JMPIFNOT, 0, cr)
            self.compile_block(stmt.body)
            self.emit(Op.JMP, start)
            self.patch_a(jump_false, len(self.proto.code))
            return

        if isinstance(stmt, A.IfStmt):
            end_jumps = []
            for cond, body in stmt.clauses:
                cr, _ = self.expr(cond)
                jf = self.emit(Op.JMPIFNOT, 0, cr)
                self.compile_block(body)
                end_jumps.append(self.emit(Op.JMP, 0))
                self.patch_a(jf, len(self.proto.code))
            if stmt.else_body:
                self.compile_block(stmt.else_body)
            end = len(self.proto.code)
            for j in end_jumps:
                self.patch_a(j, end)
            return

        if isinstance(stmt, A.FunctionDef):
            self.function_def(stmt)
            return

        raise NotImplementedError(type(stmt).__name__)

    def function_def(self, stmt: A.FunctionDef):
        local_sym = None
        if stmt.local:
            r = self.alloc()
            nil = self.nil_reg()
            self.emit(Op.LOCAL, r, nil)
            local_sym = Symbol(r, FUNCTION, returns=stmt.return_types)
            self.define(stmt.name, local_sym)

        child = Proto(
            stmt.name,
            param_count=len(stmt.params),
            param_types=[typ for _, typ in stmt.params],
            return_types=stmt.return_types,
            is_vararg=stmt.vararg_name is not None,
            vararg_type=stmt.vararg_type,
        )
        sub = _FunctionCompiler(child, stmt.params, self)
        if stmt.vararg_name not in (None, ""):
            reg = sub.alloc()
            child.vararg_name_reg = reg
            sub.define(stmt.vararg_name, Symbol(reg, TABLE, readonly=True))
        sub.compile_block(stmt.body, scoped=False)
        child.code.append(Ins(Op.RETURN, 0, 0))
        child.register_count = sub.max_reg
        self.proto.children.append(child)
        out = self.alloc()
        self.emit(Op.CLOSURE, out, len(self.proto.children) - 1)

        if stmt.local:
            ref = Ref("local", local_sym.reg, local_sym.typ, local_sym)
            self._store_ref(ref, out, stmt.line, FUNCTION)
        else:
            self._global_set(stmt.name, out)

    def prepare_target(self, target):
        if isinstance(target, A.Name):
            return ("name", target.value)
        if isinstance(target, A.Field):
            table, _ = self.expr(target.table)
            key = self.alloc()
            self.emit(Op.LOADK, key, self.proto.add_const(target.name.encode("utf-8")))
            return ("table", table, key)
        if isinstance(target, A.Index):
            table, _ = self.expr(target.table)
            key, _ = self.expr(target.key)
            return ("table", table, key)
        raise LuaSyntaxError(f"line {target.line}: invalid assignment target")

    def assign_prepared(self, target, value_reg, actual, line):
        if target[0] == "table":
            self.emit(Op.SETTABLE, target[1], target[2], value_reg)
            return
        name = target[1]
        ref = self.resolve(name)
        if ref is None:
            self._global_set(name, value_reg)
        else:
            self._store_ref(ref, value_reg, line, actual)

    def adjust_values(self, exprs: list[A.Expr], wanted: int):
        values: list[tuple[int, LuaType]] = []
        if not exprs:
            for _ in range(wanted):
                values.append((self.nil_reg(), ANY))
            return values
        for i, expr in enumerate(exprs):
            last = i == len(exprs) - 1
            if last and self.is_multi_expr(expr):
                remaining = max(1, wanted - len(values))
                values.extend(self.multi_expr(expr, remaining))
            else:
                values.append(self.expr(expr))
        while len(values) < wanted:
            values.append((self.nil_reg(), ANY))
        return values[:wanted]

    def compile_return(self, stmt: A.Return):
        if not stmt.values:
            self.emit(Op.RETURN, 0, 0)
            return
        fixed: list[tuple[int, LuaType]] = []
        last_multi = None
        for i, expr in enumerate(stmt.values):
            last = i == len(stmt.values) - 1
            if last and self.is_multi_expr(expr):
                last_multi = self.multi_expr(expr, -1)[0][0]
            else:
                fixed.append(self.expr(expr))

        for i, (_, actual) in enumerate(fixed):
            if i >= len(self.proto.return_types):
                break
            expected = self.proto.return_types[i]
            if expected is not ANY and actual is not ANY and not accepts(expected, actual):
                raise LuaTypeError(f"line {stmt.line}: cannot return {actual} as {expected}")

        base = self.alloc_n(len(fixed)) if fixed else 0
        for i, (reg, actual) in enumerate(fixed):
            if i < len(self.proto.return_types):
                expected = self.proto.return_types[i]
                if expected is not ANY and actual is ANY:
                    self.emit(Op.GUARD, reg, self.proto.add_const(expected.name))
            self.emit(Op.MOVE, base + i, reg)
        if last_multi is not None:
            self.emit(Op.RETURNV, base, len(fixed), last_multi)
        else:
            self.emit(Op.RETURN, base, len(fixed))

    @staticmethod
    def is_multi_expr(expr):
        return isinstance(expr, (A.Call, A.VarArg))

    def multi_expr(self, expr, want: int):
        if isinstance(expr, A.Call):
            return self.call_expr(expr, want)
        if isinstance(expr, A.VarArg):
            if not self.proto.is_vararg:
                raise LuaSyntaxError(f"line {expr.line}: cannot use '...' outside a variadic function")
            if want == -1:
                out = self.alloc()
                self.emit(Op.VARARG, out, -1)
                return [(out, self.proto.vararg_type)]
            base = self.alloc_n(want)
            self.emit(Op.VARARG, base, want)
            return [(base + i, self.proto.vararg_type) for i in range(want)]
        first = self.expr(expr)
        if want == -1:
            return [first]
        out = [first]
        while len(out) < want:
            out.append((self.nil_reg(), ANY))
        return out

    def expr(self, expr):
        if isinstance(expr, A.Literal):
            r = self.alloc()
            self.emit(Op.LOADK, r, self.proto.add_const(expr.value))
            return r, expr.inferred_type

        if isinstance(expr, A.Name):
            ref = self.resolve(expr.value)
            return self._load_ref(ref) if ref is not None else self._global_get(expr.value)

        if isinstance(expr, A.VarArg):
            return self.multi_expr(expr, 1)[0]

        if isinstance(expr, A.TableCtor):
            out = self.alloc()
            self.emit(Op.NEWTABLE, out)
            array_index = 1
            for i, field in enumerate(expr.fields):
                last = i == len(expr.fields) - 1
                if field.key is None:
                    if last and self.is_multi_expr(field.value):
                        mv = self.multi_expr(field.value, -1)[0][0]
                        self.emit(Op.SETLISTV, out, array_index, mv)
                    else:
                        vr, _ = self.expr(field.value)
                        kr = self.alloc()
                        self.emit(Op.LOADK, kr, self.proto.add_const(array_index))
                        self.emit(Op.SETTABLE, out, kr, vr)
                    array_index += 1
                else:
                    if isinstance(field.key, str):
                        kr = self.alloc()
                        self.emit(Op.LOADK, kr, self.proto.add_const(field.key.encode("utf-8")))
                    else:
                        kr, _ = self.expr(field.key)
                    vr, _ = self.expr(field.value)
                    self.emit(Op.SETTABLE, out, kr, vr)
            return out, TABLE

        if isinstance(expr, A.Field):
            table, _ = self.expr(expr.table)
            key = self.alloc()
            self.emit(Op.LOADK, key, self.proto.add_const(expr.name.encode("utf-8")))
            out = self.alloc()
            self.emit(Op.GETTABLE, out, table, key)
            return out, ANY

        if isinstance(expr, A.Index):
            table, _ = self.expr(expr.table)
            key, _ = self.expr(expr.key)
            out = self.alloc()
            self.emit(Op.GETTABLE, out, table, key)
            return out, ANY

        if isinstance(expr, A.Unary):
            x, typ = self.expr(expr.operand)
            out = self.alloc()
            if expr.op == "-":
                self.emit(Op.NEG, out, x)
                return out, typ
            if expr.op == "not":
                self.emit(Op.NOT, out, x)
                return out, BOOLEAN
            if expr.op == "#":
                self.emit(Op.LEN, out, x)
                return out, INTEGER
            if expr.op == "~":
                self.emit(Op.BNOT, out, x)
                return out, INTEGER
            raise NotImplementedError(expr.op)

        if isinstance(expr, A.Binary):
            if expr.op in ("and", "or"):
                left, left_type = self.expr(expr.left)
                out = self.alloc()
                self.emit(Op.MOVE, out, left)
                jump = self.emit(Op.JMPIFNOT if expr.op == "and" else Op.JMPIF, 0, left)
                right, right_type = self.expr(expr.right)
                self.emit(Op.MOVE, out, right)
                self.patch_a(jump, len(self.proto.code))
                return out, left_type if left_type == right_type else ANY

            left, left_type = self.expr(expr.left)
            right, right_type = self.expr(expr.right)
            out = self.alloc()

            if expr.op in ("+", "-", "*"):
                generic = {"+": Op.ADD, "-": Op.SUB, "*": Op.MUL}[expr.op]
                if left_type is INTEGER and right_type is INTEGER:
                    op = {"+": Op.ADD_I, "-": Op.SUB_I, "*": Op.MUL_I}[expr.op]
                    result_type = INTEGER
                elif left_type in (INTEGER, FLOAT) and right_type in (INTEGER, FLOAT) and (left_type is FLOAT or right_type is FLOAT):
                    op = {"+": Op.ADD_F, "-": Op.SUB_F, "*": Op.MUL_F}[expr.op]
                    result_type = FLOAT
                else:
                    op = generic
                    result_type = ANY if ANY in (left_type, right_type) else NUMBER
            elif expr.op == "/":
                op, result_type = Op.DIV, FLOAT
            elif expr.op == "//":
                op = Op.IDIV
                result_type = INTEGER if left_type is INTEGER and right_type is INTEGER else NUMBER
            elif expr.op == "%":
                op = Op.MOD
                result_type = INTEGER if left_type is INTEGER and right_type is INTEGER else NUMBER
            elif expr.op == "^":
                op, result_type = Op.POW, FLOAT
            elif expr.op == "..":
                op, result_type = Op.CONCAT, STRING
            elif expr.op in ("&", "|", "~", "<<", ">>"):
                op = {"&": Op.BAND, "|": Op.BOR, "~": Op.BXOR, "<<": Op.SHL, ">>": Op.SHR}[expr.op]
                result_type = INTEGER
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
            return self.call_expr(expr, 1)[0]

        raise NotImplementedError(type(expr).__name__)

    def call_expr(self, expr: A.Call, want: int):
        fn_reg, _ = self.expr(expr.func)
        fixed_args = []
        multi_last = None
        for i, arg in enumerate(expr.args):
            last = i == len(expr.args) - 1
            if last and self.is_multi_expr(arg):
                multi_last = self.multi_expr(arg, -1)[0][0]
            else:
                fixed_args.append(self.expr(arg)[0])
        arg_base = self.alloc_n(len(fixed_args)) if fixed_args else 0
        for i, reg in enumerate(fixed_args):
            self.emit(Op.MOVE, arg_base + i, reg)

        out_count = 1 if want in (0, -1) else want
        out = self.alloc_n(out_count)
        if multi_last is None:
            self.emit(Op.CALL, out, fn_reg, arg_base, len(fixed_args), want)
        else:
            self.emit(Op.CALLV, out, fn_reg, arg_base, len(fixed_args), multi_last)
            if want != -1:
                mv = out
                material = self.alloc_n(max(1, want))
                self.emit(Op.UNPACK, material, mv, max(0, want))
                out = material
        typ = ANY
        if isinstance(expr.func, A.Name):
            ref = self.resolve(expr.func.value)
            if ref and ref.symbol and ref.symbol.returns:
                typ = ref.symbol.returns[0] if ref.symbol.returns else ANY
        if want == -1:
            return [(out, typ)]
        return [(out + i, typ if i == 0 else ANY) for i in range(max(1, want))]
