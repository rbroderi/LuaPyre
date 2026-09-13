from __future__ import annotations

from dataclasses import fields, is_dataclass

from . import astnodes as A
from .bytecode import Ins, Op, Proto
from .compiler import Compiler, LoopContext, Symbol, _FunctionCompiler
from .errors import LuaTypeError
from .jit_policy import can_jit_natural_loop, can_jit_typed_loop
from .semantics import analyze_control_flow
from .source_mode import validate_fully_typed_ast
from .typesys import ANY, FLOAT, INTEGER, TABLE, accepts


def _last_body_line(body, default: int) -> int:
    return max((getattr(stmt, "end_line", 0) or stmt.line for stmt in body), default=default)


def _last_expr_line(expr) -> int:
    line = getattr(expr, "end_line", 0) or expr.line
    if is_dataclass(expr):
        for descriptor in fields(expr):
            value = getattr(expr, descriptor.name)
            if isinstance(value, A.Expr):
                line = max(line, _last_expr_line(value))
            elif isinstance(value, list):
                line = max(
                    (max(line, _last_expr_line(item)) for item in value if isinstance(item, A.Expr)),
                    default=line,
                )
    return line


def _intern_source_strings(root) -> None:
    """Share equal source literals across all nested protos in one chunk."""
    pool: dict[bytes, bytes] = {}

    def visit(value):
        if isinstance(value, A.Literal) and isinstance(value.value, bytes):
            value.value = pool.setdefault(value.value, value.value)
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, tuple):
            for item in value:
                visit(item)
        elif is_dataclass(value):
            for descriptor in fields(value):
                visit(getattr(value, descriptor.name))

    visit(root)


class SourceCompiler(Compiler):
    """Compiler variant that records source lines and source-JIT contracts."""

    def __init__(
        self,
        source: str | bytes | None = "=(luapyre)",
        *,
        fully_typed: bool = False,
    ):
        self.source = source
        self.fully_typed = fully_typed

    def compile(self, chunk: A.Chunk) -> Proto:
        _intern_source_strings(chunk)
        analyze_control_flow(chunk.body)
        if self.fully_typed:
            validate_fully_typed_ast(chunk)
        proto = Proto(
            "<chunk>",
            is_vararg=True,
            source=self.source,
            linedefined=0,
            lastlinedefined=_last_body_line(chunk.body, chunk.line),
            jit_trust_types=True,
            jit_fully_typed=self.fully_typed,
        )
        ctx = _SourceFunctionCompiler(proto, fully_typed=self.fully_typed)
        if self.fully_typed:
            # Fully typed source cannot acquire an untracked Any through an
            # accidental global lookup. Ambient host/stdlib names must be
            # declared, e.g. ``global math: table``.
            ctx.implicit_global = False
            ctx.scopes[0].implicit_before = False
        ctx.current_line = chunk.line
        env = ctx.alloc()
        proto.env_reg = env
        ctx.define_local("_ENV", Symbol(env, TABLE if self.fully_typed else ANY))
        ctx.compile_block(chunk.body, scoped=False)
        ctx.emit_close_to(0)
        ctx.emit(Op.HALT, line=_last_body_line(chunk.body, chunk.line))
        ctx.finish_debug_scope()
        ctx.patch_gotos()
        proto.register_count = ctx.max_reg
        return proto


class _SourceFunctionCompiler(_FunctionCompiler):
    def __init__(self, proto, params=None, parent=None, *, fully_typed=False):
        self.fully_typed = fully_typed
        super().__init__(proto, params, parent)
        self.current_line = proto.linedefined or 1

    def emit(self, op, a=0, b=0, c=0, d=0, e=0, *, line=None):
        self.proto.code.append(Ins(op, a, b, c, d, e))
        self.proto.lineinfo.append(self.current_line if line is None else line)
        self.proto.value_origins.append(dict(self.reg_origins))
        if op in (Op.MOVE, Op.LOCAL) and b in self.reg_origins:
            self.reg_origins[a] = self.reg_origins[b]
        return len(self.proto.code) - 1

    def stmt(self, stmt):
        previous = self.current_line
        previous_definition = getattr(self, "_definition_line", None)
        # PUC associates the closure-creation instruction for a function
        # statement with its closing ``end``. This matters to line hooks: an
        # empty local function reports its end line before the following call.
        self.current_line = (
            stmt.end_line
            if isinstance(stmt, (A.FunctionDef, A.GlobalFunctionDef))
            and stmt.end_line
            else stmt.line
        )
        if isinstance(stmt, (A.FunctionDef, A.GlobalFunctionDef)):
            self._definition_line = stmt.line
        previous_store = getattr(self, "_store_line", None)
        if isinstance(stmt, A.Assign) and stmt.values:
            # A dotted function declaration is parsed as an assignment, but
            # failures while walking/storing its table path belong to the
            # declaration line, not to the closing ``end`` of its function.
            dotted_function = (
                len(stmt.targets) == len(stmt.values) == 1
                and isinstance(stmt.targets[0], (A.Field, A.Index))
                and isinstance(stmt.values[0], A.FunctionExpr)
            )
            self._store_line = (
                stmt.line if dotted_function else _last_expr_line(stmt.values[-1])
            )
        try:
            if self.fully_typed and type(stmt) is A.LocalDecl:
                return self._typed_local_decl(stmt)
            if self.fully_typed and type(stmt) is A.GlobalDecl:
                return self._typed_global_decl(stmt)
            return super().stmt(stmt)
        finally:
            self.current_line = previous
            self._definition_line = previous_definition
            self._store_line = previous_store

    def expr(self, expr):
        previous = self.current_line
        self.current_line = getattr(self, "_line_override", None) or expr.line
        try:
            return super().expr(expr)
        finally:
            self.current_line = previous

    def _binary_left(self, expr):
        # Avoid adding another SourceCompiler.expr frame for every node in a
        # long left-associative chain. Besides matching the operator's line,
        # this lets legal 255-term expressions reach Lua's upvalue/register
        # diagnostics before CPython's own recursion guard.
        previous = self.current_line
        previous_override = getattr(self, "_line_override", None)
        self._line_override = expr.line
        self.current_line = expr.line
        try:
            return super().expr(expr.left)
        finally:
            self._line_override = previous_override
            self.current_line = previous

    def assign_prepared(self, target, value, actual, line):
        previous = self.current_line
        self.current_line = getattr(self, "_store_line", None) or line
        try:
            return super().assign_prepared(target, value, actual, line)
        finally:
            self.current_line = previous

    def _global_initialize(self, declarations, values, line):
        # Global declaration failures (including an invalid replacement
        # _ENV) are reported at the declaration, even when the initializer is
        # a multi-line function whose CLOSURE instruction belongs to ``end``.
        previous = self.current_line
        self.current_line = line
        try:
            return super()._global_initialize(declarations, values, line)
        finally:
            self.current_line = previous

    def _binding_type(self, declared, actual, line: int, kind: str):
        expected = declared.typ
        if expected is ANY:
            if actual is ANY:
                raise LuaTypeError(
                    f"line {line}: fully typed mode cannot infer type of {kind} "
                    f"'{declared.name}'; add a type annotation"
                )
            expected = actual
        if actual is not ANY and not accepts(expected, actual):
            raise LuaTypeError(f"line {line}: cannot assign {actual} to {expected}")
        return expected

    def _typed_local_decl(self, stmt: A.LocalDecl):
        values = self.adjust_values(stmt.values, len(stmt.names))
        for position, (declared, (vr, actual)) in enumerate(zip(stmt.names, values)):
            expected = self._binding_type(declared, actual, stmt.line, "local")
            if actual is ANY:
                self.emit(Op.GUARD, vr, self.proto.add_const(expected.name))
            r = self.alloc()
            value_expr = stmt.values[position] if position < len(stmt.values) else None
            local_line = (
                value_expr.end_line
                if isinstance(value_expr, A.FunctionExpr) and value_expr.end_line
                else stmt.line
            )
            self.emit(Op.LOCAL, r, vr, line=local_line)
            readonly = declared.attribute in ("const", "close")
            self.define_local(declared.name, Symbol(r, expected, readonly=readonly))
            self.reg_origins[r] = ("local", declared.name)
            if declared.attribute == "close":
                self.emit(Op.TBC, r, self.proto.add_const(declared.name.encode()))
                self.close_depth += 1

    def _typed_global_decl(self, stmt: A.GlobalDecl):
        if stmt.wildcard:
            raise LuaTypeError(
                f"line {stmt.line}: fully typed mode does not allow 'global *'"
            )

        values = self.adjust_values(stmt.values, len(stmt.names)) if stmt.values else []
        typed_declarations = []
        if values:
            for declared, (_, actual) in zip(stmt.names, values):
                expected = self._binding_type(declared, actual, stmt.line, "global")
                typed_declarations.append(
                    A.DeclaredName(declared.name, expected, declared.attribute)
                )
        else:
            for declared in stmt.names:
                if declared.typ is ANY:
                    raise LuaTypeError(
                        f"line {stmt.line}: fully typed mode requires a type for "
                        f"ambient global '{declared.name}'"
                    )
                typed_declarations.append(declared)

        for declared in typed_declarations:
            self.declare_global(
                declared.name,
                declared.typ,
                declared.attribute == "const",
            )
        if values:
            self._global_initialize(typed_declarations, values, stmt.line)

    def numeric_for(self, stmt):
        sr, start_type = self.expr(stmt.start)
        lr, limit_type = self.expr(stmt.limit)
        if stmt.step is None:
            tr = self.alloc()
            self.emit(Op.LOADK, tr, self.proto.add_const(1))
            step_type = INTEGER
        else:
            tr, step_type = self.expr(stmt.step)

        numeric_types = (start_type, limit_type, step_type)
        if all(typ is INTEGER for typ in numeric_types):
            loop_type = INTEGER
        elif all(typ in (INTEGER, FLOAT) for typ in numeric_types):
            loop_type = FLOAT
        else:
            loop_type = ANY
        if self.fully_typed and loop_type is ANY:
            raise LuaTypeError(
                f"line {stmt.line}: fully typed numeric for bounds must have "
                "statically numeric types"
            )

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
            self.define_local(stmt.name, Symbol(visible, loop_type, readonly=True))
            body_start = len(self.proto.code)
            body_line = (
                stmt.body[0].line if stmt.body else stmt.end_line or stmt.line
            )
            self.emit(Op.LOCAL, visible, idx, line=body_line)
            self.compile_block(stmt.body, scoped=False)
            self.emit_close_to(base, update=True)
            eligible = (
                can_jit_typed_loop(self.proto.code, body_start, len(self.proto.code))
                if self.fully_typed
                else can_jit_natural_loop(
                    self.proto.code, body_start, len(self.proto.code)
                )
            )
            self.emit(
                Op.JFORLOOP if eligible else Op.FORLOOP,
                idx,
                limit,
                step,
                body_start,
            )
            end = len(self.proto.code)
            self.patch_d(prep, end)
            self._finish_loop(end)
        finally:
            self.pop_scope()

    def generic_for(self, stmt):
        if self.fully_typed:
            raise LuaTypeError(
                f"line {stmt.line}: fully typed generic for variables need an "
                "explicit typed iterator contract; use a numeric loop for now"
            )
        return super().generic_for(stmt)

    def _new_child(self, name, params, returns, body, vararg_name, vararg_type, end_line=0, namewhat=""):
        analyze_control_flow(body)
        defined_line = getattr(self, "_definition_line", None) or self.current_line
        child = Proto(
            name,
            param_count=len(params),
            param_types=[typ for _, typ in params],
            return_types=returns,
            is_vararg=vararg_name is not None,
            vararg_type=vararg_type,
            source=self.proto.source,
            linedefined=defined_line,
            lastlinedefined=end_line or _last_body_line(body, defined_line),
            jit_trust_types=True,
            jit_fully_typed=self.fully_typed,
            debug_namewhat=namewhat,
        )
        sub = _SourceFunctionCompiler(
            child,
            params,
            self,
            fully_typed=self.fully_typed,
        )
        if child.is_vararg:
            child.debug_locals.append(
                ("(vararg table)", -1, 0, (1 << 31) - 1)
            )
        if vararg_name not in (None, ""):
            reg = sub.alloc()
            child.vararg_name_reg = reg
            sub.define_local(vararg_name, Symbol(reg, TABLE, readonly=True))
        sub.compile_block(body, scoped=False)
        sub.emit_close_to(0)
        sub.emit(Op.RETURN, 0, 0, line=end_line or _last_body_line(body, defined_line))
        if child.is_vararg:
            for index, item in enumerate(child.debug_locals):
                if item[0] == "(vararg table)":
                    child.debug_locals[index] = (
                        item[0], item[1], item[2], max(0, len(child.code) - 2)
                    )
                    break
        sub.finish_debug_scope()
        sub.patch_gotos()
        child.register_count = sub.max_reg
        self.proto.children.append(child)
        out = self.alloc()
        self.emit(Op.CLOSURE, out, len(self.proto.children) - 1)
        return out
