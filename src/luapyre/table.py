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

    __slots__ = (
        "array", "hash", "metatable", "version", "_gc_owner", "_gc_age",
        "_deleted_successors", "_reserved_bytes",
    )

    def __init__(self):
        self.array: list[object | None] = []
        self.hash: dict[object, tuple[object, object]] = {}
        self.metatable: LuaTable | None = None
        self.version = 0
        self._gc_owner = None
        self._gc_age = 0
        self._deleted_successors: dict[object, object | None] = {}
        self._reserved_bytes = 0

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
        if value is None and self.rawhas(key):
            entries = list(self.items())
            for index, (current, _item) in enumerate(entries):
                if _hash_key(current) == h:
                    successor = entries[index + 1][0] if index + 1 < len(entries) else None
                    self._deleted_successors[h] = successor
                    break
        elif value is not None:
            self._deleted_successors.pop(h, None)
        self.version += 1
        collector = self._gc_owner
        if collector is not None:
            collector.adopt(key)
            collector.adopt(value)
            collector.write_barrier(self, key)
            collector.write_barrier(self, value)
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
                if collector is not None:
                    collector.account_bytes(8)
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
            if collector is not None and h not in self.hash:
                collector.account_bytes(32)
            self.hash[h] = (key, value)

    def successor_after_deleted(self, key):
        """Find a surviving successor for a key deleted during traversal."""
        seen = set()
        successor = self._deleted_successors.get(_hash_key(key), _ABSENT)
        while successor is not _ABSENT and successor is not None:
            successor_hash = _hash_key(successor)
            if successor_hash in seen:
                return True, None
            seen.add(successor_hash)
            if self.rawhas(successor):
                return True, successor
            successor = self._deleted_successors.get(successor_hash, _ABSENT)
        return (False, None) if successor is _ABSENT else (True, None)

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
