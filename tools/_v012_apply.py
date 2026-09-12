from pathlib import Path


def replace(path, old, new):
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"expected patch context not found in {path}: {old[:80]!r}")
    p.write_text(text.replace(old, new, 1))


# --- opcode/runtime coercion and named-vararg semantics ---
replace(
    "src/luapyre/opdispatch.py",
    "from .values import MultiValue, i64, static_value_type, truthy, type_matches\n",
    "from .values import (\n"
    "    MultiValue, coerce_lua_integer, i64, parse_lua_number,\n"
    "    static_value_type, truthy, type_matches,\n"
    ")\n",
)
replace(
    "src/luapyre/opdispatch.py",
    '''def _to_int(value):\n    if type(value) is int:\n        return i64(value)\n    if type(value) is float and math.isfinite(value) and value.is_integer():\n        iv = int(value)\n        if _INT_MIN <= iv <= _INT_MAX:\n            return iv\n    raise LuaRuntimeError("number has no integer representation")\n''',
    '''def _to_int(value):\n    integer = coerce_lua_integer(value)\n    if integer is not None:\n        return integer\n    raise LuaRuntimeError("number has no integer representation")\n\n\ndef _shift(value, count, *, left):\n    """Lua logical shift without negating ``math.mininteger``."""\n    value &= _UINT_MASK\n    if count >= 64 or count <= -64:\n        return 0\n    if count < 0:\n        left = not left\n        count = -count\n    result = (value << count) & _UINT_MASK if left else value >> count\n    return i64(result)\n\n\ndef _bitwise_error(vm, frames, frame, op, a, b, dest):\n    name = {\n        Op.BAND: "band", Op.BOR: "bor", Op.BXOR: "bxor",\n        Op.SHL: "shl", Op.SHR: "shr",\n    }[op]\n    tm = vm._first_tm(a, b, vm.ARITH_TM[op])\n    if tm is not None:\n        vm._invoke(frames, frame, tm, [a, b], dest, 1)\n        return\n    bad = a if coerce_lua_integer(a) is None else b\n    raise LuaRuntimeError(\n        f"attempt to perform '{name}' on a {static_value_type(bad).name} value"\n    )\n''',
)
replace(
    "src/luapyre/opdispatch.py",
    '''def _generic_arith(vm, frames, frame, ins, regs, constants):\n    vm._generic_binary(frames, frame, ins.op, regs[ins.b], regs[ins.c], ins.a)\n''',
    '''def _generic_arith(vm, frames, frame, ins, regs, constants):\n    a, b = regs[ins.b], regs[ins.c]\n    if ins.op in (Op.BAND, Op.BOR, Op.BXOR, Op.SHL, Op.SHR):\n        ai, bi = coerce_lua_integer(a), coerce_lua_integer(b)\n        if ai is None or bi is None:\n            _bitwise_error(vm, frames, frame, ins.op, a, b, ins.a)\n            return\n        if ins.op is Op.BAND:\n            value = i64(ai & bi)\n        elif ins.op is Op.BOR:\n            value = i64(ai | bi)\n        elif ins.op is Op.BXOR:\n            value = i64(ai ^ bi)\n        elif ins.op is Op.SHL:\n            value = _shift(ai, bi, left=True)\n        else:\n            value = _shift(ai, bi, left=False)\n        regs[ins.a] = value\n        return\n\n    na, nb = parse_lua_number(a), parse_lua_number(b)\n    if na is not None and nb is not None:\n        vm._generic_binary(frames, frame, ins.op, na, nb, ins.a)\n    else:\n        vm._generic_binary(frames, frame, ins.op, a, b, ins.a)\n''',
)
replace(
    "src/luapyre/opdispatch.py",
    '''def _neg(vm, frames, frame, ins, regs, constants):\n    value = regs[ins.b]\n    if _is_number(value):\n        regs[ins.a] = i64(-value) if type(value) is int else -value\n        return\n''',
    '''def _neg(vm, frames, frame, ins, regs, constants):\n    value = regs[ins.b]\n    number = parse_lua_number(value)\n    if number is not None:\n        regs[ins.a] = i64(-number) if type(number) is int else -number\n        return\n''',
)
replace(
    "src/luapyre/opdispatch.py",
    '''def _bnot(vm, frames, frame, ins, regs, constants):\n    value = regs[ins.b]\n    try:\n        regs[ins.a] = i64(~_to_int(value))\n    except LuaRuntimeError:\n        tm = vm._tm(value, b"__bnot")\n        if tm is None:\n            raise\n        vm._invoke(frames, frame, tm, [value], ins.a, 1)\n''',
    '''def _bnot(vm, frames, frame, ins, regs, constants):\n    value = regs[ins.b]\n    integer = coerce_lua_integer(value)\n    if integer is not None:\n        regs[ins.a] = i64(~integer)\n        return\n    tm = vm._tm(value, b"__bnot")\n    if tm is None:\n        raise LuaRuntimeError(\n            f"attempt to perform 'bnot' on a {static_value_type(value).name} value"\n        )\n    vm._invoke(frames, frame, tm, [value], ins.a, 1)\n''',
)
replace(
    "src/luapyre/opdispatch.py",
    '''def _vararg(vm, frames, frame, ins, regs, constants):\n    if ins.b == -1:\n        regs[ins.a] = MultiValue(frame.varargs)\n        return\n    for i in range(ins.b):\n        regs[ins.a + i] = frame.varargs[i] if i < len(frame.varargs) else None\n''',
    '''def _native_varargs(frame, regs):\n    named = frame.proto.vararg_name_reg\n    if named < 0:\n        return frame.varargs\n    table = regs[named]\n    if not isinstance(table, LuaTable):\n        raise LuaRuntimeError("no proper 'n' field in vararg table")\n    count = table.rawget(b"n")\n    if type(count) is not int or count < 0 or count > 1_000_000:\n        raise LuaRuntimeError("no proper 'n' field in vararg table")\n    return tuple(table.rawget(index) for index in range(1, count + 1))\n\n\ndef _vararg(vm, frames, frame, ins, regs, constants):\n    values = _native_varargs(frame, regs)\n    if ins.b == -1:\n        regs[ins.a] = MultiValue(tuple(values))\n        return\n    for i in range(ins.b):\n        regs[ins.a + i] = values[i] if i < len(values) else None\n''',
)

# Named vararg tables are implicit operands of VARARG and must remain Lua-GC roots.
replace(
    "src/luapyre/gc.py",
    '''        elif op is Op.VARARG:\n            if ins.b == -1:\n                writes.add(ins.a)\n            else:\n                writes.update(range(ins.a, ins.a + max(0, ins.b)))\n''',
    '''        elif op is Op.VARARG:\n            if proto.vararg_name_reg >= 0:\n                reads.add(proto.vararg_name_reg)\n            if ins.b == -1:\n                writes.add(ins.a)\n            else:\n                writes.update(range(ins.a, ins.a + max(0, ins.b)))\n''',
)

# --- pattern diagnostics required by the upstream suite ---
replace(
    "src/luapyre/lua_pattern.py",
    '''                if 49 <= special <= 57:\n                    index = special - 49\n                    if index >= len(captures):\n                        raise LuaRuntimeError("invalid capture index")\n                    capture = captures[index]\n                    if capture.position or capture.end is None:\n                        raise LuaRuntimeError("invalid capture index")\n''',
    '''                if 48 <= special <= 57:\n                    if special == 48:\n                        raise LuaRuntimeError("invalid capture index %0")\n                    index = special - 49\n                    if index >= len(captures):\n                        raise LuaRuntimeError(f"invalid capture index %{chr(special)}")\n                    capture = captures[index]\n                    if capture.position or capture.end is None:\n                        raise LuaRuntimeError(f"invalid capture index %{chr(special)}")\n''',
)
replace(
    "src/luapyre/stdlib_string.py",
    "from .values import MultiValue\n",
    "from .values import MultiValue, lua_type_name\n",
)
replace(
    "src/luapyre/stdlib_string.py",
    '''                            raise LuaRuntimeError("invalid capture index")\n''',
    '''                            raise LuaRuntimeError(\n                                f"invalid capture index %{chr(code)}"\n                            )\n''',
)
replace(
    "src/luapyre/stdlib_string.py",
    '''                    raise LuaRuntimeError("invalid replacement value")\n''',
    '''                    raise LuaRuntimeError(\n                        f"invalid replacement value (a {lua_type_name(replacement)})"\n                    )\n''',
)

# --- pack format limits/alignment diagnostics ---
replace(
    "src/luapyre/stdlib_string.py",
    '''        if op == "!":\n            n, self.index = self._number(text, self.index)\n            n = ctypes.sizeof(ctypes.c_long) if n is None else n\n            if n <= 0 or n > 16 or n & (n - 1):\n                raise LuaRuntimeError("invalid alignment")\n            self.maxalign = n\n            return ("config", op, 0, 1)\n''',
    '''        if op == "!":\n            n, self.index = self._number(text, self.index)\n            n = ctypes.sizeof(ctypes.c_long) if n is None else n\n            if n <= 0 or n > 16:\n                raise LuaRuntimeError(f"integral size ({n}) out of limits [1,16]")\n            if n & (n - 1):\n                raise LuaRuntimeError("format asks for alignment not power of 2")\n            self.maxalign = n\n            return ("config", op, 0, 1)\n''',
)
replace(
    "src/luapyre/stdlib_string.py",
    '''            if not 1 <= size <= 16:\n                raise LuaRuntimeError("integral size out of limits")\n            return ("int", op == "i", size, min(size, self.maxalign))\n''',
    '''            if not 1 <= size <= 16:\n                raise LuaRuntimeError(f"integral size ({size}) out of limits [1,16]")\n            alignment = min(size, self.maxalign)\n            if alignment & (alignment - 1):\n                raise LuaRuntimeError("format asks for alignment not power of 2")\n            return ("int", op == "i", size, alignment)\n''',
)
replace(
    "src/luapyre/stdlib_string.py",
    '''            if not 1 <= size <= 16:\n                raise LuaRuntimeError("integral size out of limits")\n            return ("sized", op, size, min(size, self.maxalign))\n''',
    '''            if not 1 <= size <= 16:\n                raise LuaRuntimeError(f"integral size ({size}) out of limits [1,16]")\n            alignment = min(size, self.maxalign)\n            if alignment & (alignment - 1):\n                raise LuaRuntimeError("format asks for alignment not power of 2")\n            return ("sized", op, size, alignment)\n''',
)

# --- UTF-8: offset walks character boundaries, it does not validate codepoints ---
replace(
    "src/luapyre/stdlib_utf8.py",
    '''    def codepoint(s, i=1, j=None, lax=False):\n        s = need_bytes(s, 1, "codepoint")\n        i = _position(need_integer(i, 2, "codepoint"), len(s), allow_end=True)\n        j = i if j is None else _position(need_integer(j, 3, "codepoint"), len(s), allow_end=True)\n''',
    '''    def codepoint(s, i=1, j=None, lax=False):\n        s = need_bytes(s, 1, "codepoint")\n        i = _position(need_integer(i, 2, "codepoint"), len(s), allow_end=False)\n        j = i if j is None else _position(need_integer(j, 3, "codepoint"), len(s), allow_end=False)\n''',
)
replace(
    "src/luapyre/stdlib_utf8.py",
    '''    def length_fn(s, i=1, j=-1, lax=False):\n        s = need_bytes(s, 1, "len")\n        i = _position(need_integer(i, 2, "len"), len(s), allow_end=True)\n        j = _position(need_integer(j, 3, "len"), len(s), allow_end=True)\n        if i > j:\n            return 0\n''',
    '''    def length_fn(s, i=1, j=-1, lax=False):\n        s = need_bytes(s, 1, "len")\n        i = _position(need_integer(i, 2, "len"), len(s), allow_end=True)\n        raw_j = need_integer(j, 3, "len")\n        j = normalize_index(raw_j, len(s))\n        if len(s) == 0 and raw_j == -1:\n            j = 0\n        if j < 1 or j > len(s):\n            if i == len(s) + 1 and j == len(s):\n                return 0\n            if len(s) == 0 and j == 0:\n                return 0\n            raise LuaRuntimeError("position out of bounds")\n        if i > j:\n            return 0\n''',
)
replace(
    "src/luapyre/stdlib_utf8.py",
    '''        if n == 0:\n            if i == len(s) + 1:\n                return MultiValue((i, i))\n            pos = i - 1\n            while pos > 0 and 0x80 <= s[pos] <= 0xBF:\n                pos -= 1\n            _cp, end = _decode(s, pos, True)\n            if i - 1 >= end:\n                raise LuaRuntimeError("initial position is a continuation byte")\n            return MultiValue((pos + 1, end))\n\n        if i <= len(s) and 0x80 <= s[i - 1] <= 0xBF:\n            raise LuaRuntimeError("initial position is a continuation byte")\n\n        if n > 0:\n            pos = i - 1\n            remaining = n\n            while remaining > 1:\n                if pos >= len(s):\n                    return None\n                _cp, pos = _decode(s, pos, True)\n                remaining -= 1\n            if pos == len(s):\n                return MultiValue((len(s) + 1, len(s) + 1))\n            if pos > len(s):\n                return None\n            _cp, end = _decode(s, pos, True)\n            return MultiValue((pos + 1, end))\n\n        pos = i - 1\n        remaining = -n\n        while remaining > 0:\n            if pos <= 0:\n                return None\n            pos -= 1\n            while pos > 0 and 0x80 <= s[pos] <= 0xBF:\n                pos -= 1\n            remaining -= 1\n        _cp, end = _decode(s, pos, True)\n        return MultiValue((pos + 1, end))\n''',
    '''        if n == 0:\n            if i == len(s) + 1:\n                return MultiValue((i, i))\n            pos = i - 1\n            while pos > 0 and 0x80 <= s[pos] <= 0xBF:\n                pos -= 1\n            end = pos + 1\n            while end < len(s) and 0x80 <= s[end] <= 0xBF:\n                end += 1\n            return MultiValue((pos + 1, end))\n\n        if i <= len(s) and 0x80 <= s[i - 1] <= 0xBF:\n            raise LuaRuntimeError("initial position is a continuation byte")\n\n        if n > 0:\n            pos = i - 1\n            remaining = n\n            while remaining > 1:\n                if pos >= len(s):\n                    return None\n                pos += 1\n                while pos < len(s) and 0x80 <= s[pos] <= 0xBF:\n                    pos += 1\n                remaining -= 1\n            if pos > len(s):\n                return None\n            if pos == len(s):\n                return MultiValue((len(s) + 1, len(s) + 1))\n            end = pos + 1\n            while end < len(s) and 0x80 <= s[end] <= 0xBF:\n                end += 1\n            return MultiValue((pos + 1, end))\n\n        pos = i - 1\n        remaining = -n\n        while remaining > 0:\n            if pos <= 0:\n                return None\n            pos -= 1\n            while pos > 0 and 0x80 <= s[pos] <= 0xBF:\n                pos -= 1\n            remaining -= 1\n        end = pos + 1\n        while end < len(s) and 0x80 <= s[end] <= 0xBF:\n            end += 1\n        return MultiValue((pos + 1, end))\n''',
)

print("0.12 semantic patches applied")
