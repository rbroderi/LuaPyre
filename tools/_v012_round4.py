from pathlib import Path


def replace(path, old, new):
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"patch context missing in {path}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1))


# Lua calls both <const> locals and named-vararg bindings "const variable" in
# syntax diagnostics. Preserve LuaPyre's existing wording too so API-facing
# diagnostics remain backward compatible while exact upstream checks match.
replace(
    "src/luapyre/compiler.py",
    '''        if ref.readonly or (ref.symbol is not None and ref.symbol.readonly):\n            kind = "global" if ref.kind == "global" else "local"\n            raise LuaTypeError(f"line {line}: cannot assign to read-only {kind} '{ref.name}'")\n''',
    '''        if ref.readonly or (ref.symbol is not None and ref.symbol.readonly):\n            kind = "global" if ref.kind == "global" else "local"\n            raise LuaTypeError(\n                f"line {line}: cannot assign to read-only {kind} '{ref.name}' "\n                f"(const variable '{ref.name}')"\n            )\n''',
)

# Python raises for several libm domain/overflow cases where Lua's C-number
# arithmetic observes the platform IEEE result. Normalize those cases instead.
replace(
    "src/luapyre/vm.py",
    '''def _float_div(a, b):\n    a = float(a)\n    b = float(b)\n    if b != 0.0:\n        return a / b\n    if a == 0.0:\n        return math.nan\n    return math.copysign(math.inf, a * (1.0 if math.copysign(1.0, b) > 0 else -1.0))\n\n\n''',
    '''def _float_div(a, b):\n    a = float(a)\n    b = float(b)\n    if b != 0.0:\n        return a / b\n    if a == 0.0:\n        return math.nan\n    return math.copysign(math.inf, a * (1.0 if math.copysign(1.0, b) > 0 else -1.0))\n\n\ndef _float_pow(a, b):\n    a = float(a)\n    b = float(b)\n    try:\n        return math.pow(a, b)\n    except ValueError:\n        if a == 0.0 and b < 0.0:\n            negative = (\n                math.copysign(1.0, a) < 0.0\n                and math.isfinite(b)\n                and b.is_integer()\n                and int(b) & 1\n            )\n            return -math.inf if negative else math.inf\n        return math.nan\n    except OverflowError:\n        negative = (\n            a < 0.0\n            and math.isfinite(b)\n            and b.is_integer()\n            and int(b) & 1\n        )\n        return -math.inf if negative else math.inf\n\n\n''',
)
replace(
    "src/luapyre/vm.py",
    '''        if op is Op.POW:\n            if not (_is_number(a) and _is_number(b)):\n                return False, None\n            return True, float(a) ** float(b)\n''',
    '''        if op is Op.POW:\n            if not (_is_number(a) and _is_number(b)):\n                return False, None\n            return True, _float_pow(a, b)\n''',
)

# PUC-Lua caps parsed format numerals and all fixed-size pack/packsize arithmetic
# at MAX_SIZE (lua_Integer max on this platform). This prevents Python big ints
# from silently accepting formats that Lua rejects as invalid/too large.
replace(
    "src/luapyre/stdlib_string.py",
    '''    @staticmethod\n    def _number(text: str, index: int):\n        start = index\n        while index < len(text) and text[index].isdigit():\n            index += 1\n        if start == index:\n            return None, index\n        return int(text[start:index]), index\n''',
    '''    @staticmethod\n    def _number(text: str, index: int):\n        start = index\n        value = 0\n        while index < len(text) and text[index].isdigit():\n            digit = ord(text[index]) - ord("0")\n            if value > (INT_MAX - digit) // 10:\n                raise LuaRuntimeError("invalid format")\n            value = value * 10 + digit\n            index += 1\n        if start == index:\n            return None, index\n        return value, index\n''',
)
replace(
    "src/luapyre/stdlib_string.py",
    '''def _padding(offset: int, alignment: int) -> int:\n    return (-offset) % max(1, alignment)\n\n\ndef _pack(fmt: bytes, values):\n''',
    '''def _padding(offset: int, alignment: int) -> int:\n    return (-offset) % max(1, alignment)\n\n\ndef _checked_size(offset: int, amount: int, message: str) -> int:\n    if amount < 0 or offset > INT_MAX - amount:\n        raise LuaRuntimeError(message)\n    return offset + amount\n\n\ndef _pack(fmt: bytes, values):\n''',
)
replace(
    "src/luapyre/stdlib_string.py",
    '''        kind, detail, size, alignment = option\n        pad = _padding(len(out), alignment)\n        out.extend(b"\\0" * pad)\n        if kind == "padding":\n            out.append(0)\n            continue\n        if kind == "align":\n            continue\n        if value_index >= len(values):\n''',
    '''        kind, detail, size, alignment = option\n        pad = _padding(len(out), alignment)\n        projected = _checked_size(len(out), pad, "format result too long")\n        if kind in ("padding", "int", "float", "fixed", "sized"):\n            _checked_size(projected, size, "format result too long")\n        out.extend(b"\\0" * pad)\n        if kind == "padding":\n            out.append(0)\n            continue\n        if kind == "align":\n            continue\n        if value_index >= len(values):\n''',
)
replace(
    "src/luapyre/stdlib_string.py",
    '''        if kind in ("zero", "sized"):\n            raise LuaRuntimeError("variable-length format")\n        offset += _padding(offset, alignment)\n        if kind in ("int", "float", "fixed", "padding"):\n            offset += size\n    return offset\n''',
    '''        if kind in ("zero", "sized"):\n            raise LuaRuntimeError("variable-length format")\n        offset = _checked_size(\n            offset, _padding(offset, alignment), "format result too large"\n        )\n        if kind in ("int", "float", "fixed", "padding"):\n            offset = _checked_size(offset, size, "format result too large")\n    return offset\n''',
)

print("round-four patch applied")
