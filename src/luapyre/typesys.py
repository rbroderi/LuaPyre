from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LuaType:
    name: str

    def __str__(self) -> str:
        return self.name


ANY = LuaType("Any")
NIL = LuaType("nil")
BOOLEAN = LuaType("boolean")
INTEGER = LuaType("integer")
FLOAT = LuaType("float")
NUMBER = LuaType("number")
STRING = LuaType("string")

_SIMPLE = {t.name: t for t in (ANY, NIL, BOOLEAN, INTEGER, FLOAT, NUMBER, STRING)}


@dataclass(frozen=True, slots=True)
class UnionType(LuaType):
    members: frozenset[LuaType]

    def __init__(self, members):
        flat: set[LuaType] = set()
        for member in members:
            if isinstance(member, UnionType):
                flat.update(member.members)
            else:
                flat.add(member)
        object.__setattr__(self, "members", frozenset(flat))
        object.__setattr__(self, "name", " | ".join(sorted(t.name for t in flat)))


def parse_simple_type(name: str) -> LuaType:
    return _SIMPLE.get(name, LuaType(name))


def union_of(*types: LuaType) -> LuaType:
    flat: set[LuaType] = set()
    for typ in types:
        if typ is ANY:
            return ANY
        if isinstance(typ, UnionType):
            flat.update(typ.members)
        else:
            flat.add(typ)
    if len(flat) == 1:
        return next(iter(flat))
    return UnionType(flat)


def accepts(expected: LuaType, actual: LuaType) -> bool:
    if expected is ANY or actual is ANY:
        return True
    if expected == actual:
        return True
    if expected is NUMBER and actual in (INTEGER, FLOAT):
        return True
    if isinstance(expected, UnionType):
        return any(accepts(member, actual) for member in expected.members)
    return False


def python_value_type(value) -> LuaType:
    if value is None:
        return NIL
    if type(value) is bool:
        return BOOLEAN
    if type(value) is int:
        return INTEGER
    if type(value) is float:
        return FLOAT
    if isinstance(value, (bytes, str)):
        return STRING
    return ANY
