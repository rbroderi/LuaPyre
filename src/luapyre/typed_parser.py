from __future__ import annotations

from .errors import LuaSyntaxError
from .parser import Parser
from .typesys import NIL, LuaType, parse_simple_type, union_of


_TYPE_KEYWORDS = frozenset(("function", "nil"))


class TypedParser(Parser):
    """Parser additions that are useful only in LuaPyre's typed dialect.

    Lua reserves ``function`` and ``nil`` as lexical keywords, so the ordinary
    parser cannot consume them with ``take('NAME')`` inside a type annotation.
    Typed source needs both atoms for ambient declarations and unions. Keeping
    this as a small parser specialization leaves plain Lua parsing untouched.
    """

    def __init__(self, source: str):
        super().__init__(source, typed=True)

    def _type_atom(self) -> LuaType:
        token = self.t
        if token.kind == "NAME" or token.kind in _TYPE_KEYWORDS:
            self.take()
            return parse_simple_type(token.value)
        raise LuaSyntaxError(
            f"expected type name, got {token.kind!r} at line {token.line}"
        )

    def type_annotation(self) -> LuaType:
        parts = [self._type_atom()]
        if self.accept("?"):
            parts.append(NIL)
        while self.accept("|"):
            part = self._type_atom()
            if self.accept("?"):
                part = union_of(part, NIL)
            parts.append(part)
        return union_of(*parts)
