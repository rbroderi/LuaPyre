from __future__ import annotations

import math
from .errors import LuaRuntimeError

_ABSENT = object()
_BOOL = object()
_NUM = object()
_STR = object()
_OBJ = object()


def _hash_key(key):
    if key is None:
        return None
    if type(key) is bool:
        return (_BOOL, key)
    if type(key) is int:
        return (_NUM, key)
    if type(key) is float:
        # Lua rejects NaN only when assigning a table key. A lookup with NaN
        # cannot match any stored key and therefore behaves like an absent key.
        if math.isnan(key):
            return None
        return (_NUM, int(key) if key.is_integer() else key)
    if isinstance(key, bytes):
        return (_STR, key)
    return (_OBJ, id(key))


class LuaTable:
    """Lua table with dense array storage, tagged hash keys, and a metatable."""

    __slots__ = ("array", "hash", "metatable", "version")

    def __init__(self):
        self.array: list[object | None] = []
        self.hash: dict[object, tuple[object, object]] = {}
        self.metatable: LuaTable | None = None
        self.version = 0

    def rawget(self, key):
        h = _hash_key(key)
        if h is None:
            return None
        if h[0] is _NUM and type(h[1]) is int and h[1] >= 1:
            idx = h[1] - 1
            if idx < len(self.array):
                return self.array[idx]
        item = self.hash.get(h, _ABSENT)
        return None if item is _ABSENT else item[1]

    def rawhas(self, key) -> bool:
        """Whether a non-nil raw key exists, without invoking metamethods."""
        h = _hash_key(key)
        if h is None:
            return False
        if h[0] is _NUM and type(h[1]) is int and h[1] >= 1:
            idx = h[1] - 1
            if idx < len(self.array) and self.array[idx] is not None:
                return True
        return h in self.hash

    def rawset(self, key, value):
        if type(key) is float and math.isnan(key):
            raise LuaRuntimeError("table index is NaN")
        h = _hash_key(key)
        if h is None:
            raise LuaRuntimeError("table index is nil")
        self.version += 1
        if h[0] is _NUM and type(h[1]) is int and h[1] >= 1:
            index = h[1]
            if index <= len(self.array):
                self.array[index - 1] = value
                if value is None:
                    while self.array and self.array[-1] is None:
                        self.array.pop()
                return
            if index == len(self.array) + 1 and value is not None:
                self.array.append(value)
                while True:
                    next_h = (_NUM, len(self.array) + 1)
                    item = self.hash.pop(next_h, _ABSENT)
                    if item is _ABSENT:
                        break
                    self.array.append(item[1])
                return
        if value is None:
            self.hash.pop(h, None)
        else:
            self.hash[h] = (key, value)

    def rawlen(self) -> int:
        return len(self.array)

    def items(self):
        for i, value in enumerate(self.array, 1):
            if value is not None:
                yield i, value
        for original, value in self.hash.values():
            yield original, value

    @classmethod
    def from_sequence(cls, values):
        table = cls()
        for i, value in enumerate(values, 1):
            table.rawset(i, value)
        return table

    def __contains__(self, key):
        if isinstance(key, str):
            key = key.encode("utf-8")
        return self.rawhas(key)

    def __getitem__(self, key):
        if isinstance(key, str):
            key = key.encode("utf-8")
        return self.rawget(key)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = key.encode("utf-8")
        self.rawset(key, value)

    def __repr__(self):
        return f"table: 0x{id(self):x}"
