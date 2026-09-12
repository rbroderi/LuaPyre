from __future__ import annotations

from . import astnodes as A
from .bytecode import Ins, Op, Proto
from .compiler import Compiler, Symbol, _FunctionCompiler
from .semantics import analyze_control_flow
from .typesys import TABLE


def _last_body_line(body, default: int) -> int:
    return max((stmt.line for stmt in body), default=default)


class SourceCompiler(Compiler):
    """Compiler variant that records Lua source/line information per opcode."""

    def __init__(self, source: str | bytes | None = "=(luapyre)"):
        self.source = source

    def compile(self, chunk: A.Chunk) -> Proto:
        analyze_control_flow(chunk.body)
        proto = Proto(
            "<chunk>",
            is_vararg=True,
            source=self.source,
            linedefined=0,
            lastlinedefined=_last_body_line(chunk.body, chunk.line),
            jit_trust_types=True,
        )
        ctx = _SourceFunctionCompiler(proto)
        ctx.current_line = chunk.line
        env = ctx.alloc()
        proto.env_reg = env
        ctx.define_local("_ENV", Symbol(env, TABLE))
        ctx.compile_block(chunk.body, scoped=False)
        ctx.emit_close_to(0)
        ctx.emit(Op.HALT, line=_last_body_line(chunk.body, chunk.line))
        ctx.patch_gotos()
        proto.register_count = ctx.max_reg
        return proto


class _SourceFunctionCompiler(_FunctionCompiler):
    def __init__(self, proto, params=None, parent=None):
        super().__init__(proto, params, parent)
        self.current_line = proto.linedefined or 1

    def emit(self, op, a=0, b=0, c=0, d=0, e=0, *, line=None):
        self.proto.code.append(Ins(op, a, b, c, d, e))
        self.proto.lineinfo.append(self.current_line if line is None else line)
        return len(self.proto.code) - 1

    def stmt(self, stmt):
        previous = self.current_line
        self.current_line = stmt.line
        try:
            return super().stmt(stmt)
        finally:
            self.current_line = previous

    def expr(self, expr):
        previous = self.current_line
        self.current_line = expr.line
        try:
            return super().expr(expr)
        finally:
            self.current_line = previous

    def _new_child(self, name, params, returns, body, vararg_name, vararg_type):
        analyze_control_flow(body)
        defined_line = self.current_line
        child = Proto(
            name,
            param_count=len(params),
            param_types=[typ for _, typ in params],
            return_types=returns,
            is_vararg=vararg_name is not None,
            vararg_type=vararg_type,
            source=self.proto.source,
            linedefined=defined_line,
            lastlinedefined=_last_body_line(body, defined_line),
            jit_trust_types=True,
        )
        sub = _SourceFunctionCompiler(child, params, self)
        if vararg_name not in (None, ""):
            reg = sub.alloc()
            child.vararg_name_reg = reg
            sub.define_local(vararg_name, Symbol(reg, TABLE, readonly=True))
        sub.compile_block(body, scoped=False)
        sub.emit_close_to(0)
        sub.emit(Op.RETURN, 0, 0, line=_last_body_line(body, defined_line))
        sub.patch_gotos()
        child.register_count = sub.max_reg
        self.proto.children.append(child)
        out = self.alloc()
        self.emit(Op.CLOSURE, out, len(self.proto.children) - 1)
        return out
