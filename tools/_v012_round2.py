from pathlib import Path


def replace(path, old, new):
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"patch context missing in {path}: {old[:80]!r}")
    p.write_text(text.replace(old, new, 1))


replace(
    "src/luapyre/stdlib.py",
    "    def next_fn(table, key=None):\n",
    "    def next_fn(table, key=None, *_ignored):\n",
)

replace(
    "src/luapyre/vm.py",
    '''        if op is Op.IDIV:\n            if not (_is_number(a) and _is_number(b)):\n                return False, None\n            if b == 0:\n                raise LuaRuntimeError("attempt to divide by zero")\n            q = math.floor(a / b)\n            return True, i64(q) if type(a) is int and type(b) is int else float(q)\n''',
    '''        if op is Op.IDIV:\n            if not (_is_number(a) and _is_number(b)):\n                return False, None\n            if b == 0:\n                if type(a) is int and type(b) is int:\n                    raise LuaRuntimeError("attempt to divide by zero")\n                return True, _float_div(a, b)\n            q = math.floor(a / b)\n            return True, i64(q) if type(a) is int and type(b) is int else float(q)\n''',
)

replace(
    "src/luapyre/stdlib_string.py",
    '''        copied = 0\n        count = 0\n        while count < limit and search_pos <= len(s):\n''',
    '''        copied = 0\n        count = 0\n        changed = False\n        while count < limit and search_pos <= len(s):\n''',
)
replace(
    "src/luapyre/stdlib_string.py",
    '''            rendered = _replacement_bytes(replacement)\n            if rendered is None:\n                if replacement is None or replacement is False:\n                    rendered = s[match.start:match.end]\n                else:\n                    raise LuaRuntimeError(\n                        f"invalid replacement value (a {lua_type_name(replacement)})"\n                    )\n            output.extend(rendered)\n''',
    '''            rendered = _replacement_bytes(replacement)\n            if rendered is None:\n                if replacement is None or replacement is False:\n                    rendered = s[match.start:match.end]\n                else:\n                    raise LuaRuntimeError(\n                        f"invalid replacement value (a {lua_type_name(replacement)})"\n                    )\n            else:\n                changed = True\n            output.extend(rendered)\n''',
)
replace(
    "src/luapyre/stdlib_string.py",
    '''        output.extend(s[copied:])\n        return MultiValue((bytes(output), count))\n''',
    '''        output.extend(s[copied:])\n        return MultiValue((bytes(output) if changed else s, count))\n''',
)

print("round-two patch applied")
