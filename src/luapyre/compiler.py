from __future__ import annotations

from dataclasses import dataclass, field

from . import astnodes as A
from .bytecode import Op, Ins, Proto, UpvalueDesc
from .errors import LuaSyntaxError, LuaTypeError
from .semantics import analyze_control_flow
from .typesys import ANY, BOOLEAN, FLOAT, FUNCTION, INTEGER, NUMBER, STRING, TABLE, LuaType, accepts


# Stable compile-time dispatch tables. These are deliberately exact-type maps:
# the AST is a closed set of concrete node classes, and avoiding repeated
# isinstance chains keeps dispatch predictable as the language grows.
_CALL_EXPR_TYPES = frozenset((A.Call, A.MethodCall))
_MULTI_EXPR_TYPES = frozenset((A.Call, A.MethodCall, A.VarArg))

_GENERIC_ARITH_OPS = {"+": Op.ADD, "-": Op.SUB, "*": Op.MUL}
_INTEGER_ARITH_OPS = {"+": Op.ADD_I, "-": Op.SUB_I, "*": Op.MUL_I}
_FLOAT_ARITH_OPS = {"+": Op.ADD_F, "-": Op.SUB_F, "*": Op.MUL_F}
_BITWISE_OPS = {"&": Op.BAND, "|": Op.BOR, "~": Op.BXOR, "<<": Op.SHL, ">>": Op.SHR}
_FIXED_BINARY_OPS = {
    "/": (Op.DIV, FLOAT),
    "^": (Op.POW, FLOAT),
    "..": (Op.CONCAT, STRING),
    "==": (Op.EQ, BOOLEAN),
    "~=": (Op.EQ, BOOLEAN),
    "<": (Op.LT, BOOLEAN),
    ">": (Op.LT, BOOLEAN),
    "<=": (Op.LE, BOOLEAN),
    ">=": (Op.LE, BOOLEAN),
}
_INTEGER_SENSITIVE_OPS = {"//": Op.IDIV, "%": Op.MOD}
_COMPARISON_OPS = frozenset(("==", "~=", "<", ">", "<=", ">="))
_REVERSED_COMPARISON_OPS = frozenset((">", ">="))
_LOGICAL_JUMPS = {"and": Op.JMPIFNOT, "or": Op.JMPIF}
_UNARY_OPS = {"-": Op.NEG, "not": Op.NOT, "#": Op.LEN, "~": Op.BNOT}
_UNARY_RESULT_TYPES = {"not": BOOLEAN, "#": INTEGER, "~": INTEGER}


@dataclass(slots=True)
class Symbol:
    reg: int
    typ: LuaType
    captured: bool = False
    readonly: bool = False
    returns: list[LuaType] | None = None


@dataclass(frozen=True, slots=True)
class GlobalBinding:
    typ: LuaType = ANY
    readonly: bool = False


@dataclass(slots=True)
class Scope:
    bindings: dict[str, Symbol | GlobalBinding] = field(default_factory=dict)
    wildcard: bool | None = None
    implicit_before: bool = True
    close_base: int = 0


@dataclass(frozen=True, slots=True)
class Ref:
    kind: str
    index: int
    typ: LuaType
    symbol: Symbol | None = None
    name: str | None = None
    readonly: bool = False


@dataclass(slots=True)
class LoopContext:
    close_depth: int
    jumps: list[int] = field(default_factory=list)


class Compiler:
    def compile(self, chunk: A.Chunk) -> Proto:
        analyze_control_flow(chunk.body)
        proto = Proto("<chunk>", is_vararg=True)
        ctx = _FunctionCompiler(proto)
        env = ctx.alloc()
        proto.env_reg = env
        ctx.define_local("_ENV", Symbol(env, TABLE))
        ctx.compile_block(chunk.body, scoped=False)
        ctx.emit_close_to(0)
        proto.code.append(Ins(Op.HALT))
        ctx.patch_gotos()
        proto.register_count = ctx.max_reg
        return proto


class _FunctionCompiler:
    def __init__(self, proto, params=None, parent=None):
        self.proto = proto
        self.parent = parent
        inherited_implicit = True if parent is None else parent.implicit_global
        self.implicit_global = inherited_implicit
        self.scopes = [Scope({}, None, inherited_implicit, 0)]
        self.upvalue_by_name: dict[str, int] = {}
        self.upvalue_types: list[LuaType] = []
        self.next_reg = 0
        self.max_reg = 0
        self.close_depth = 0
        self.loop_breaks: list[LoopContext] = []
        self.label_pcs: dict[int, int] = {}
        self.pending_gotos: list[tuple[int, int]] = []
        for name, typ in params or []:
            r = self.alloc()
            self.scopes[0].bindings[name] = Symbol(r, typ)

    def alloc(self):
        r = self.next_reg
        self.next_reg += 1
        self.max_reg = max(self.max_reg, self.next_reg)
        return r

    def alloc_n(self, n):
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

    def patch_d(self, at, target):
        ins = self.proto.code[at]
        self.proto.code[at] = Ins(ins.op, ins.a, ins.b, ins.c, target, ins.e)

    def patch_gotos(self):
        for pc, label_id in self.pending_gotos:
            if label_id not in self.label_pcs:
                raise LuaSyntaxError("internal error: unresolved goto label")
            self.patch_a(pc, self.label_pcs[label_id])

    def push_scope(self):
        self.scopes.append(Scope({}, None, self.implicit_global, self.close_depth))

    def pop_scope(self):
        scope = self.scopes.pop()
        self.implicit_global = scope.implicit_before
        self.close_depth = scope.close_base

    def define_local(self, name, sym):
        self.scopes[-1].bindings[name] = sym

    def declare_global(self, name, typ=ANY, readonly=False):
        if name == "_ENV":
            raise LuaSyntaxError("declaring _ENV as global is not supported")
        self.implicit_global = False
        binding = GlobalBinding(typ, readonly)
        self.scopes[-1].bindings[name] = binding
        return binding

    def declare_wildcard(self, readonly=False):
        self.implicit_global = False
        self.scopes[-1].wildcard = readonly

    def _find_binding(self, name):
        for scope in reversed(self.scopes):
            binding = scope.bindings.get(name)
            if binding is not None:
                return binding
        return None

    def _fallback_global(self, name, *, allow_implicit=True):
        for scope in reversed(self.scopes):
            if scope.wildcard is not None:
                return Ref("global", -1, ANY, name=name, readonly=scope.wildcard)
        if allow_implicit and self.implicit_global:
            return Ref("global", -1, ANY, name=name, readonly=False)
        return None

    def _make_upvalue(self, name, source):
        if name in self.upvalue_by_name:
            idx = self.upvalue_by_name[name]
            return Ref("upvalue", idx, self.upvalue_types[idx], name=name)
        idx = len(self.proto.upvalues)
        self.proto.upvalues.append(UpvalueDesc(source.kind, source.index, name))
        self.upvalue_types.append(source.typ)
        self.upvalue_by_name[name] = idx
        return Ref("upvalue", idx, source.typ, name=name)

    def capture_for_child(self, name, *, allow_implicit=True):
        binding = self._find_binding(name)
        if isinstance(binding, Symbol):
            binding.captured = True
            return Ref("local", binding.reg, binding.typ, binding, name)
        if isinstance(binding, GlobalBinding):
            return Ref("global", -1, binding.typ, name=name, readonly=binding.readonly)

        if name in self.upvalue_by_name:
            idx = self.upvalue_by_name[name]
            return Ref("upvalue", idx, self.upvalue_types[idx], name=name)

        if self.parent is not None:
            source = self.parent.capture_for_child(
                name, allow_implicit=allow_implicit and self.implicit_global
            )
            if source is not None:
                if source.kind in ("local", "upvalue"):
                    return self._make_upvalue(name, source)
                return source

        return self._fallback_global(name, allow_implicit=allow_implicit)

    def resolve(self, name):
        binding = self._find_binding(name)
        if isinstance(binding, Symbol):
            return Ref("local", binding.reg, binding.typ, binding, name)
        if isinstance(binding, GlobalBinding):
            return Ref("global", -1, binding.typ, name=name, readonly=binding.readonly)

        if name in self.upvalue_by_name:
            idx = self.upvalue_by_name[name]
            return Ref("upvalue", idx, self.upvalue_types[idx], name=name)

        if self.parent is not None:
            source = self.parent.capture_for_child(name, allow_implicit=self.implicit_global)
            if source is not None:
                if source.kind in ("local", "upvalue"):
                    return self._make_upvalue(name, source)
                return source

        global_ref = self._fallback_global(name)
        if global_ref is None:
            raise LuaSyntaxError(f"global '{name}' is not declared in this scope")
        return global_ref

    def compile_block(self, body, scoped=True):
        if scoped:
            self.push_scope()
        try:
            for stmt in body:
                self.stmt(stmt)
            if scoped:
                self.emit_close_to(self.scopes[-1].close_base, update=True)
        finally:
            if scoped:
                self.pop_scope()

    def emit_close_to(self, target, *, update=False):
        if self.close_depth > target:
            self.emit(Op.CLOSE, target)
        if update:
            self.close_depth = target

    def nil_reg(self):
        r = self.alloc()
        self.emit(Op.LOADK, r, self.proto.add_const(None))
        return r

    def _env_reg(self):
        ref = self.resolve("_ENV")
        return self._load_ref(ref)[0]

    def _emit_global_get(self, name):
        out = self.alloc()
        env = self._env_reg()
        key = self.alloc()
        self.emit(Op.LOADK, key, self.proto.add_const(name.encode()))
        self.emit(Op.GETTABLE, out, env, key)
        return out

    def _emit_global_set(self, name, value):
        env = self._env_reg()
        key = self.alloc()
        self.emit(Op.LOADK, key, self.proto.add_const(name.encode()))
        self.emit(Op.SETTABLE, env, key, value)

    def _load_ref(self, ref):
        match ref.kind:
            case "global":
                return self._emit_global_get(ref.name), ref.typ
            case "local":
                if ref.symbol.captured:
                    out = self.alloc()
                    self.emit(Op.GETCELL, out, ref.index)
                    return out, ref.typ
                return ref.index, ref.typ
            case "upvalue":
                out = self.alloc()
                self.emit(Op.GETUPVAL, out, ref.index)
                return out, ref.typ
            case _:
                raise RuntimeError(f"unknown reference kind: {ref.kind}")

    def _store_ref(self, ref, value, line, actual=ANY):
        if ref.readonly or (ref.symbol is not None and ref.symbol.readonly):
            kind = "global" if ref.kind == "global" else "local"
            raise LuaTypeError(f"line {line}: cannot assign to read-only {kind} '{ref.name}'")
        if ref.typ is not ANY and actual is not ANY and not accepts(ref.typ, actual):
            raise LuaTypeError(f"line {line}: cannot assign {actual} to {ref.typ}")
        if ref.typ is not ANY and actual is ANY:
            self.emit(Op.GUARD, value, self.proto.add_const(ref.typ.name))

        match ref.kind:
            case "global":
                self._emit_global_set(ref.name, value)
            case "local":
                if ref.symbol.captured:
                    self.emit(Op.SETCELL, ref.index, value)
                else:
                    self.emit(Op.MOVE, ref.index, value)
            case "upvalue":
                self.emit(Op.SETUPVAL, ref.index, value)
            case _:
                raise RuntimeError(f"unknown reference kind: {ref.kind}")

    def _global_initialize(self, declarations, values, line):
        prepared = []
        for declared, (value, actual) in zip(declarations, values):
            if declared.typ is not ANY and actual is not ANY and not accepts(declared.typ, actual):
                raise LuaTypeError(f"line {line}: cannot assign {actual} to {declared.typ}")
            if declared.typ is not ANY and actual is ANY:
                self.emit(Op.GUARD, value, self.proto.add_const(declared.typ.name))
            env = self._env_reg()
            key = self.alloc()
            key_const = self.proto.add_const(declared.name.encode())
            self.emit(Op.LOADK, key, key_const)
            existing = self.alloc()
            self.emit(Op.GETTABLE, existing, env, key)
            self.emit(Op.CHECKNIL, existing, key_const)
            prepared.append((env, key, value))
        for env, key, value in prepared:
            self.emit(Op.SETTABLE, env, key, value)

    def _finish_loop(self, end):
        context = self.loop_breaks.pop()
        for jump in context.jumps:
            self.patch_a(jump, end)

    def stmt(self, stmt):
        handler = _STMT_HANDLERS.get(type(stmt))
        if handler is None:
            raise NotImplementedError(type(stmt).__name__)
        return handler(self, stmt)

    def _stmt_local_decl(self, stmt):
        values = self.adjust_values(stmt.values, len(stmt.names))
        for declared, (vr, actual) in zip(stmt.names, values):
            if declared.typ is not ANY and actual is not ANY and not accepts(declared.typ, actual):
                raise LuaTypeError(f"line {stmt.line}: cannot assign {actual} to {declared.typ}")
            if declared.typ is not ANY and actual is ANY:
                self.emit(Op.GUARD, vr, self.proto.add_const(declared.typ.name))
            r = self.alloc()
            self.emit(Op.LOCAL, r, vr)
            readonly = declared.attribute in ("const", "close")
            self.define_local(declared.name, Symbol(r, declared.typ, readonly=readonly))
            if declared.attribute == "close":
                self.emit(Op.TBC, r)
                self.close_depth += 1

    def _stmt_global_decl(self, stmt):
        if stmt.wildcard:
            self.declare_wildcard(stmt.wildcard_attribute == "const")
            return
        values = self.adjust_values(stmt.values, len(stmt.names)) if stmt.values else []
        for declared in stmt.names:
            self.declare_global(declared.name, declared.typ, declared.attribute == "const")
        if values:
            self._global_initialize(stmt.names, values, stmt.line)

    def _stmt_global_function_def(self, stmt):
        declared = A.DeclaredName(stmt.name, FUNCTION, None)
        self.declare_global(stmt.name, FUNCTION, False)
        out = self._new_child(
            stmt.name, stmt.params, stmt.return_types, stmt.body,
            stmt.vararg_name, stmt.vararg_type,
        )
        self._global_initialize([declared], [(out, FUNCTION)], stmt.line)

    def _stmt_assign(self, stmt):
        targets = [self.prepare_target(t) for t in stmt.targets]
        values = self.adjust_values(stmt.values, len(stmt.targets))
        for target, (vr, actual) in zip(targets, values):
            self.assign_prepared(target, vr, actual, stmt.line)

    def _stmt_return(self, stmt):
        self.compile_return(stmt)

    def _stmt_expr(self, stmt):
        if type(stmt.expr) in _CALL_EXPR_TYPES:
            self.call_expr(stmt.expr, 0)
        else:
            self.expr(stmt.expr)

    def _stmt_label(self, stmt):
        self.label_pcs[stmt.label_id] = len(self.proto.code)

    def _stmt_goto(self, stmt):
        self.emit_close_to(stmt.close_depth)
        jump = self.emit(Op.JMP, 0)
        self.pending_gotos.append((jump, stmt.target_id))

    def _stmt_do(self, stmt):
        self.compile_block(stmt.body)

    def _stmt_while(self, stmt):
        start = len(self.proto.code)
        cr, _ = self.expr(stmt.condition)
        jump_false = self.emit(Op.JMPIFNOT, 0, cr)
        self.loop_breaks.append(LoopContext(self.close_depth))
        self.compile_block(stmt.body)
        self.emit(Op.JMP, start)
        end = len(self.proto.code)
        self.patch_a(jump_false, end)
        self._finish_loop(end)

    def _stmt_repeat(self, stmt):
        base = self.close_depth
        self.push_scope()
        start = len(self.proto.code)
        self.loop_breaks.append(LoopContext(base))
        try:
            self.compile_block(stmt.body, scoped=False)
            cr, _ = self.expr(stmt.condition)
            self.emit_close_to(base, update=True)
            self.emit(Op.JMPIFNOT, start, cr)
            end = len(self.proto.code)
            self._finish_loop(end)
        finally:
            self.pop_scope()

    def _stmt_numeric_for(self, stmt):
        self.numeric_for(stmt)

    def _stmt_generic_for(self, stmt):
        self.generic_for(stmt)

    def _stmt_break(self, stmt):
        if not self.loop_breaks:
            raise LuaSyntaxError(f"line {stmt.line}: 'break' outside a loop")
        context = self.loop_breaks[-1]
        self.emit_close_to(context.close_depth)
        context.jumps.append(self.emit(Op.JMP, 0))

    def _stmt_if(self, stmt):
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
        for jump in end_jumps:
            self.patch_a(jump, end)

    def _stmt_function_def(self, stmt):
        self.function_def(stmt)

    def numeric_for(self, stmt):
        sr, _ = self.expr(stmt.start)
        lr, _ = self.expr(stmt.limit)
        if stmt.step is None:
            tr = self.alloc()
            self.emit(Op.LOADK, tr, self.proto.add_const(1))
        else:
            tr, _ = self.expr(stmt.step)
        idx = self.alloc()
        limit = self.alloc()
        step = self.alloc()
        self.emit(Op.MOVE, idx, sr)
        self.emit(Op.MOVE, limit, lr)
        self.emit(Op.MOVE, step, tr)
        prep = self.emit(Op.FORPREP, idx, limit, step, 0)
        base = self.close_depth
        self.push_scope()
        self.loop_breaks.append(LoopContext(base))
        try:
            visible = self.alloc()
            self.define_local(stmt.name, Symbol(visible, ANY, readonly=True))
            body_start = len(self.proto.code)
            self.emit(Op.LOCAL, visible, idx)
            self.compile_block(stmt.body, scoped=False)
            self.emit_close_to(base, update=True)
            self.emit(Op.FORLOOP, idx, limit, step, body_start)
            end = len(self.proto.code)
            self.patch_d(prep, end)
            self._finish_loop(end)
        finally:
            self.pop_scope()

    def generic_for(self, stmt):
        values = self.adjust_values(stmt.values, 4)
        hidden = self.alloc_n(4)
        for i, (reg, _) in enumerate(values):
            self.emit(Op.MOVE, hidden + i, reg)
        iterator, state, control = hidden, hidden + 1, hidden + 2
        base = self.close_depth
        self.push_scope()
        self.loop_breaks.append(LoopContext(base))
        try:
            variables = []
            for i, name in enumerate(stmt.names):
                reg = self.alloc()
                self.define_local(name, Symbol(reg, ANY, readonly=(i == 0)))
                variables.append(reg)
            loop_start = len(self.proto.code)
            arg_base = self.alloc_n(2)
            self.emit(Op.MOVE, arg_base, state)
            self.emit(Op.MOVE, arg_base + 1, control)
            out = self.alloc_n(max(1, len(variables)))
            self.emit(Op.CALL, out, iterator, arg_base, 2, len(variables))
            done = self.emit(Op.JMPIFNIL, 0, out)
            self.emit(Op.MOVE, control, out)
            for i, reg in enumerate(variables):
                self.emit(Op.LOCAL, reg, out + i)
            self.compile_block(stmt.body, scoped=False)
            self.emit_close_to(base, update=True)
            self.emit(Op.JMP, loop_start)
            end = len(self.proto.code)
            self.patch_a(done, end)
            self._finish_loop(end)
        finally:
            self.pop_scope()

    def _new_child(self, name, params, returns, body, vararg_name, vararg_type):
        analyze_control_flow(body)
        child = Proto(
            name,
            param_count=len(params),
            param_types=[typ for _, typ in params],
            return_types=returns,
            is_vararg=vararg_name is not None,
            vararg_type=vararg_type,
        )
        sub = _FunctionCompiler(child, params, self)
        if vararg_name not in (None, ""):
            reg = sub.alloc()
            child.vararg_name_reg = reg
            sub.define_local(vararg_name, Symbol(reg, TABLE, readonly=True))
        sub.compile_block(body, scoped=False)
        sub.emit_close_to(0)
        child.code.append(Ins(Op.RETURN, 0, 0))
        sub.patch_gotos()
        child.register_count = sub.max_reg
        self.proto.children.append(child)
        out = self.alloc()
        self.emit(Op.CLOSURE, out, len(self.proto.children) - 1)
        return out

    def function_def(self, stmt):
        local_sym = None
        if stmt.local:
            reg = self.alloc()
            nil = self.nil_reg()
            self.emit(Op.LOCAL, reg, nil)
            local_sym = Symbol(reg, FUNCTION, returns=stmt.return_types)
            self.define_local(stmt.name, local_sym)
        out = self._new_child(
            stmt.name, stmt.params, stmt.return_types, stmt.body,
            stmt.vararg_name, stmt.vararg_type,
        )
        if stmt.local:
            self._store_ref(
                Ref("local", local_sym.reg, local_sym.typ, local_sym, stmt.name),
                out, stmt.line, FUNCTION,
            )
        else:
            self._store_ref(self.resolve(stmt.name), out, stmt.line, FUNCTION)

    def function_expr(self, expr):
        return (
            self._new_child(
                "<anonymous>", expr.params, expr.return_types, expr.body,
                expr.vararg_name, expr.vararg_type,
            ),
            FUNCTION,
        )

    def prepare_target(self, target):
        match target:
            case A.Name(value=name):
                return ("name", name)
            case A.Field(table=table_expr, name=name):
                table, _ = self.expr(table_expr)
                key = self.alloc()
                self.emit(Op.LOADK, key, self.proto.add_const(name.encode()))
                return ("table", table, key)
            case A.Index(table=table_expr, key=key_expr):
                table, _ = self.expr(table_expr)
                key, _ = self.expr(key_expr)
                return ("table", table, key)
            case _:
                raise LuaSyntaxError(f"line {target.line}: invalid assignment target")

    def assign_prepared(self, target, value, actual, line):
        match target:
            case ("table", table, key):
                self.emit(Op.SETTABLE, table, key, value)
            case ("name", name):
                self._store_ref(self.resolve(name), value, line, actual)
            case _:
                raise RuntimeError(f"unknown prepared target: {target!r}")

    def adjust_values(self, exprs, wanted):
        values = []
        if not exprs:
            return [(self.nil_reg(), ANY) for _ in range(wanted)]
        for i, expr in enumerate(exprs):
            if i == len(exprs) - 1 and self.is_multi_expr(expr):
                values.extend(self.multi_expr(expr, max(1, wanted - len(values))))
            else:
                values.append(self.expr(expr))
        while len(values) < wanted:
            values.append((self.nil_reg(), ANY))
        return values[:wanted]

    def compile_return(self, stmt):
        if not stmt.values:
            self.emit_close_to(0)
            self.emit(Op.RETURN, 0, 0)
            return
        if (
            len(stmt.values) == 1
            and type(stmt.values[0]) in _CALL_EXPR_TYPES
            and (not self.proto.return_types or self.proto.return_types[0] is ANY)
        ):
            self.tailcall_expr(stmt.values[0])
            return
        fixed = []
        last_multi = None
        for i, expr in enumerate(stmt.values):
            if i == len(stmt.values) - 1 and self.is_multi_expr(expr):
                last_multi = self.multi_expr(expr, -1)[0][0]
            else:
                fixed.append(self.expr(expr))
        for i, (_, actual) in enumerate(fixed):
            if i < len(self.proto.return_types):
                expected = self.proto.return_types[i]
                if expected is not ANY and actual is not ANY and not accepts(expected, actual):
                    raise LuaTypeError(f"line {stmt.line}: cannot return {actual} as {expected}")
        base = self.alloc_n(len(fixed)) if fixed else 0
        for i, (reg, actual) in enumerate(fixed):
            if i < len(self.proto.return_types) and self.proto.return_types[i] is not ANY and actual is ANY:
                self.emit(Op.GUARD, reg, self.proto.add_const(self.proto.return_types[i].name))
            self.emit(Op.MOVE, base + i, reg)
        self.emit_close_to(0)
        if last_multi is not None:
            self.emit(Op.RETURNV, base, len(fixed), last_multi)
        else:
            self.emit(Op.RETURN, base, len(fixed))

    @staticmethod
    def is_multi_expr(expr):
        return type(expr) in _MULTI_EXPR_TYPES

    def multi_expr(self, expr, want):
        expr_type = type(expr)
        if expr_type in _CALL_EXPR_TYPES:
            return self.call_expr(expr, want)
        if expr_type is A.VarArg:
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
        return [first] + [(self.nil_reg(), ANY) for _ in range(want - 1)]

    def expr(self, expr):
        handler = _EXPR_HANDLERS.get(type(expr))
        if handler is None:
            raise NotImplementedError(type(expr).__name__)
        return handler(self, expr)

    def _expr_literal(self, expr):
        r = self.alloc()
        self.emit(Op.LOADK, r, self.proto.add_const(expr.value))
        return r, expr.inferred_type

    def _expr_name(self, expr):
        return self._load_ref(self.resolve(expr.value))

    def _expr_vararg(self, expr):
        return self.multi_expr(expr, 1)[0]

    def _expr_function(self, expr):
        return self.function_expr(expr)

    def _expr_table_ctor(self, expr):
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
                    self.emit(Op.LOADK, kr, self.proto.add_const(field.key.encode()))
                else:
                    kr, _ = self.expr(field.key)
                vr, _ = self.expr(field.value)
                self.emit(Op.SETTABLE, out, kr, vr)
        return out, TABLE

    def _expr_field(self, expr):
        table, _ = self.expr(expr.table)
        key = self.alloc()
        self.emit(Op.LOADK, key, self.proto.add_const(expr.name.encode()))
        out = self.alloc()
        self.emit(Op.GETTABLE, out, table, key)
        return out, ANY

    def _expr_index(self, expr):
        table, _ = self.expr(expr.table)
        key, _ = self.expr(expr.key)
        out = self.alloc()
        self.emit(Op.GETTABLE, out, table, key)
        return out, ANY

    def _expr_unary(self, expr):
        x, typ = self.expr(expr.operand)
        op = _UNARY_OPS.get(expr.op)
        if op is None:
            raise NotImplementedError(expr.op)
        out = self.alloc()
        self.emit(op, out, x)
        return out, _UNARY_RESULT_TYPES.get(expr.op, typ)

    def _expr_binary(self, expr):
        logical_jump = _LOGICAL_JUMPS.get(expr.op)
        if logical_jump is not None:
            left, left_type = self.expr(expr.left)
            out = self.alloc()
            self.emit(Op.MOVE, out, left)
            jump = self.emit(logical_jump, 0, left)
            right, right_type = self.expr(expr.right)
            self.emit(Op.MOVE, out, right)
            self.patch_a(jump, len(self.proto.code))
            return out, left_type if left_type == right_type else ANY

        left, left_type = self.expr(expr.left)
        right, right_type = self.expr(expr.right)
        out = self.alloc()

        if expr.op in _GENERIC_ARITH_OPS:
            if left_type is INTEGER and right_type is INTEGER:
                op = _INTEGER_ARITH_OPS[expr.op]
                result_type = INTEGER
            elif (
                left_type in (INTEGER, FLOAT)
                and right_type in (INTEGER, FLOAT)
                and (left_type is FLOAT or right_type is FLOAT)
            ):
                op = _FLOAT_ARITH_OPS[expr.op]
                result_type = FLOAT
            else:
                op = _GENERIC_ARITH_OPS[expr.op]
                result_type = ANY if ANY in (left_type, right_type) else NUMBER
        elif expr.op in _INTEGER_SENSITIVE_OPS:
            op = _INTEGER_SENSITIVE_OPS[expr.op]
            result_type = INTEGER if left_type is INTEGER and right_type is INTEGER else NUMBER
        elif expr.op in _BITWISE_OPS:
            op = _BITWISE_OPS[expr.op]
            result_type = INTEGER
        else:
            fixed = _FIXED_BINARY_OPS.get(expr.op)
            if fixed is None:
                raise NotImplementedError(expr.op)
            op, result_type = fixed

        emit_left, emit_right = (right, left) if expr.op in _REVERSED_COMPARISON_OPS else (left, right)
        self.emit(op, out, emit_left, emit_right)

        if expr.op in _COMPARISON_OPS:
            self.emit(Op.TOBOOL, out, out)
        if expr.op == "~=":
            negated = self.alloc()
            self.emit(Op.NOT, negated, out)
            return negated, BOOLEAN
        return out, result_type

    def _expr_call(self, expr):
        return self.call_expr(expr, 1)[0]

    def _call_parts(self, expr):
        match expr:
            case A.MethodCall(receiver=receiver_expr, name=name, args=source_args):
                receiver, _ = self.expr(receiver_expr)
                key = self.alloc()
                self.emit(Op.LOADK, key, self.proto.add_const(name.encode()))
                fn = self.alloc()
                self.emit(Op.GETTABLE, fn, receiver, key)
                args = [receiver]
            case A.Call(func=func, args=source_args):
                fn, _ = self.expr(func)
                args = []
            case _:
                raise RuntimeError(f"not a call expression: {type(expr).__name__}")

        multi_last = None
        for i, arg in enumerate(source_args):
            if i == len(source_args) - 1 and self.is_multi_expr(arg):
                multi_last = self.multi_expr(arg, -1)[0][0]
            else:
                args.append(self.expr(arg)[0])
        arg_base = self.alloc_n(len(args)) if args else 0
        for i, reg in enumerate(args):
            self.emit(Op.MOVE, arg_base + i, reg)
        return fn, arg_base, len(args), multi_last

    def call_expr(self, expr, want):
        fn, arg_base, arg_count, multi_last = self._call_parts(expr)
        out_count = 1 if want in (0, -1) else want
        out = self.alloc_n(out_count)
        if multi_last is None:
            self.emit(Op.CALL, out, fn, arg_base, arg_count, want)
        else:
            self.emit(Op.CALLV, out, fn, arg_base, arg_count, multi_last)
            if want != -1:
                mv = out
                material = self.alloc_n(max(1, want))
                self.emit(Op.UNPACK, material, mv, max(0, want))
                out = material
        typ = ANY
        if type(expr) is A.Call and type(expr.func) is A.Name:
            ref = self.resolve(expr.func.value)
            if ref.symbol and ref.symbol.returns:
                typ = ref.symbol.returns[0] if ref.symbol.returns else ANY
            elif ref.kind == "global":
                typ = ref.typ
        if want == -1:
            return [(out, typ)]
        return [(out + i, typ if i == 0 else ANY) for i in range(max(1, want))]

    def tailcall_expr(self, expr):
        fn, arg_base, arg_count, multi_last = self._call_parts(expr)
        self.emit_close_to(0)
        if multi_last is None:
            self.emit(Op.TAILCALL, 0, fn, arg_base, arg_count, 0)
        else:
            self.emit(Op.TAILCALLV, 0, fn, arg_base, arg_count, multi_last)


_STMT_HANDLERS = {
    A.LocalDecl: _FunctionCompiler._stmt_local_decl,
    A.GlobalDecl: _FunctionCompiler._stmt_global_decl,
    A.GlobalFunctionDef: _FunctionCompiler._stmt_global_function_def,
    A.Assign: _FunctionCompiler._stmt_assign,
    A.Return: _FunctionCompiler._stmt_return,
    A.ExprStmt: _FunctionCompiler._stmt_expr,
    A.LabelStmt: _FunctionCompiler._stmt_label,
    A.GotoStmt: _FunctionCompiler._stmt_goto,
    A.DoStmt: _FunctionCompiler._stmt_do,
    A.WhileStmt: _FunctionCompiler._stmt_while,
    A.RepeatStmt: _FunctionCompiler._stmt_repeat,
    A.NumericForStmt: _FunctionCompiler._stmt_numeric_for,
    A.GenericForStmt: _FunctionCompiler._stmt_generic_for,
    A.BreakStmt: _FunctionCompiler._stmt_break,
    A.IfStmt: _FunctionCompiler._stmt_if,
    A.FunctionDef: _FunctionCompiler._stmt_function_def,
}

_EXPR_HANDLERS = {
    A.Literal: _FunctionCompiler._expr_literal,
    A.Name: _FunctionCompiler._expr_name,
    A.VarArg: _FunctionCompiler._expr_vararg,
    A.FunctionExpr: _FunctionCompiler._expr_function,
    A.TableCtor: _FunctionCompiler._expr_table_ctor,
    A.Field: _FunctionCompiler._expr_field,
    A.Index: _FunctionCompiler._expr_index,
    A.Unary: _FunctionCompiler._expr_unary,
    A.Binary: _FunctionCompiler._expr_binary,
    A.Call: _FunctionCompiler._expr_call,
    A.MethodCall: _FunctionCompiler._expr_call,
}
