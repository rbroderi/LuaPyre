from __future__ import annotations

from dataclasses import fields, is_dataclass
import re

from . import astnodes as A
from .errors import LuaSyntaxError, LuaTypeError
from .typesys import ANY


_COOKIE_RE = re.compile(
    r"^[\t ]*--[\t ]*luapyre[\t ]*:[\t ]*([A-Za-z0-9_-]+)[\t ]*$"
)


PLAIN_MODE = "lua"
FULLY_TYPED_MODE = "typed"


def detect_source_mode(source: str) -> str:
    """Return the LuaPyre source mode requested by the first two lines.

    The cookie intentionally mirrors Python's encoding-cookie placement rule:
    only the first or second physical source line is inspected. The canonical
    spelling is::

        -- luapyre: typed

    A misspelled/unknown LuaPyre mode in those two lines is rejected instead of
    silently compiling under different semantics.
    """

    lines = source.splitlines()[:2]
    for index, line in enumerate(lines):
        if index == 0:
            line = line.lstrip("\ufeff")
        match = _COOKIE_RE.match(line)
        if match is None:
            continue
        mode = match.group(1).lower()
        if mode == FULLY_TYPED_MODE:
            return FULLY_TYPED_MODE
        raise LuaSyntaxError(
            f"line {index + 1}: unknown LuaPyre source mode '{match.group(1)}'"
        )
    return PLAIN_MODE


def _validate_signature(node) -> None:
    for name, typ in node.params:
        if typ is ANY:
            raise LuaTypeError(
                f"line {node.line}: fully typed mode requires a type for parameter '{name}'"
            )
    if not node.return_types or any(typ is ANY for typ in node.return_types):
        raise LuaTypeError(
            f"line {node.line}: fully typed mode requires explicit function return type(s)"
        )
    if node.vararg_name is not None and node.vararg_type is ANY:
        raise LuaTypeError(
            f"line {node.line}: fully typed mode requires a type for varargs"
        )


def validate_fully_typed_ast(chunk: A.Chunk) -> None:
    """Validate signature-level invariants for fully typed source.

    Binding completeness that depends on expression types is enforced while the
    compiler walks declarations. This pass handles function signatures early
    and recursively, including anonymous functions.
    """

    seen: set[int] = set()

    def walk(value) -> None:
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return
        ident = id(value)
        if ident in seen:
            return
        seen.add(ident)

        if isinstance(value, (A.FunctionDef, A.GlobalFunctionDef, A.FunctionExpr)):
            _validate_signature(value)

        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
            return

        if is_dataclass(value):
            for descriptor in fields(value):
                walk(getattr(value, descriptor.name))

    walk(chunk)
