from pathlib import Path


def replace(path, old, new):
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"patch context missing in {path}: {old[:140]!r}")
    p.write_text(text.replace(old, new, 1))


# Empty strings are members of every Python string, so the old lookahead
# misclassified a trailing decimal zero as the start of an empty hex literal.
replace(
    "src/luapyre/lexer.py",
    '''        hexadecimal = self._peek() == "0" and self._peek(1) in "xX"\n''',
    '''        lookahead = self._peek(1)\n        hexadecimal = self._peek() == "0" and bool(lookahead) and lookahead in "xX"\n''',
)

# Preserve const/read-only provenance when a local is captured as an upvalue.
replace(
    "src/luapyre/compiler.py",
    '''        self.upvalue_by_name: dict[str, int] = {}\n        self.upvalue_types: list[LuaType] = []\n        self.next_reg = 0\n''',
    '''        self.upvalue_by_name: dict[str, int] = {}\n        self.upvalue_types: list[LuaType] = []\n        self.readonly_upvalues: set[int] = set()\n        self.next_reg = 0\n''',
)
replace(
    "src/luapyre/compiler.py",
    '''    def _make_upvalue(self, name, source):\n        if name in self.upvalue_by_name:\n            idx = self.upvalue_by_name[name]\n            return Ref("upvalue", idx, self.upvalue_types[idx], name=name)\n        idx = len(self.proto.upvalues)\n        self.proto.upvalues.append(UpvalueDesc(source.kind, source.index, name))\n        self.upvalue_types.append(source.typ)\n        self.upvalue_by_name[name] = idx\n        return Ref("upvalue", idx, source.typ, name=name)\n''',
    '''    def _make_upvalue(self, name, source):\n        if name in self.upvalue_by_name:\n            idx = self.upvalue_by_name[name]\n            return Ref(\n                "upvalue", idx, self.upvalue_types[idx], name=name,\n                readonly=idx in self.readonly_upvalues,\n            )\n        idx = len(self.proto.upvalues)\n        self.proto.upvalues.append(UpvalueDesc(source.kind, source.index, name))\n        self.upvalue_types.append(source.typ)\n        readonly = source.readonly or (\n            source.symbol is not None and source.symbol.readonly\n        )\n        if readonly:\n            self.readonly_upvalues.add(idx)\n        self.upvalue_by_name[name] = idx\n        return Ref("upvalue", idx, source.typ, name=name, readonly=readonly)\n''',
)
replace(
    "src/luapyre/compiler.py",
    '''        if name in self.upvalue_by_name:\n            idx = self.upvalue_by_name[name]\n            return Ref("upvalue", idx, self.upvalue_types[idx], name=name)\n\n        if self.parent is not None:\n''',
    '''        if name in self.upvalue_by_name:\n            idx = self.upvalue_by_name[name]\n            return Ref(\n                "upvalue", idx, self.upvalue_types[idx], name=name,\n                readonly=idx in self.readonly_upvalues,\n            )\n\n        if self.parent is not None:\n''',
)
# The same lookup occurs once in resolve(); replace the remaining occurrence.
replace(
    "src/luapyre/compiler.py",
    '''        if name in self.upvalue_by_name:\n            idx = self.upvalue_by_name[name]\n            return Ref("upvalue", idx, self.upvalue_types[idx], name=name)\n\n        if self.parent is not None:\n''',
    '''        if name in self.upvalue_by_name:\n            idx = self.upvalue_by_name[name]\n            return Ref(\n                "upvalue", idx, self.upvalue_types[idx], name=name,\n                readonly=idx in self.readonly_upvalues,\n            )\n\n        if self.parent is not None:\n''',
)

# Lua's X takes alignment from exactly the immediately following option.
# Spaces/configuration/X/z have zero alignment; c is explicitly disallowed.
# A padding x has alignment 1 and is therefore valid.
replace(
    "src/luapyre/stdlib_string.py",
    '''        if op == "X":\n            following = self._raw_option()\n            while following is not None and following[0] == "config":\n                following = self._raw_option()\n            if following is None or following[0] in ("zero", "padding"):\n                raise LuaRuntimeError("invalid next option for option 'X'")\n            return ("align", op, 0, following[3])\n''',
    '''        if op == "X":\n            if self.index >= len(text) or text[self.index] == " ":\n                raise LuaRuntimeError("invalid next option for option 'X'")\n            following = self._raw_option()\n            if (\n                following is None\n                or following[0] in ("config", "fixed", "zero", "align")\n                or following[3] == 0\n            ):\n                raise LuaRuntimeError("invalid next option for option 'X'")\n            return ("align", op, 0, following[3])\n''',
)

print("round-five patch applied")
