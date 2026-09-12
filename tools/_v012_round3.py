from pathlib import Path


def replace(path, old, new):
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"patch context missing in {path}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1))


# Lua main chunks are variadic. This matters both for direct source execution
# and for functions produced by load(), which receive their call arguments as
# the chunk's varargs.
replace(
    "src/luapyre/source_compiler.py",
    '''        proto = Proto(\n            "<chunk>",\n            source=self.source,\n''',
    '''        proto = Proto(\n            "<chunk>",\n            is_vararg=True,\n            source=self.source,\n''',
)
replace(
    "src/luapyre/compiler.py",
    '''        proto = Proto("<chunk>")\n''',
    '''        proto = Proto("<chunk>", is_vararg=True)\n''',
)

# Do integer floor division with Python integers, not via float division. The
# latter rounds maxinteger/1 to 2^63 before floor and then wraps incorrectly.
replace(
    "src/luapyre/vm.py",
    '''            if b == 0:\n                if type(a) is int and type(b) is int:\n                    raise LuaRuntimeError("attempt to divide by zero")\n                return True, _float_div(a, b)\n            q = math.floor(a / b)\n            return True, i64(q) if type(a) is int and type(b) is int else float(q)\n''',
    '''            if b == 0:\n                if type(a) is int and type(b) is int:\n                    raise LuaRuntimeError("attempt to divide by zero")\n                return True, _float_div(a, b)\n            if type(a) is int and type(b) is int:\n                return True, i64(a // b)\n            return True, float(math.floor(float(a) / float(b)))\n''',
)

# Walking backward from end-of-string must reject a standalone continuation
# byte instead of treating byte 1 as a valid character boundary.
replace(
    "src/luapyre/stdlib_utf8.py",
    '''        pos = i - 1\n        remaining = -n\n        while remaining > 0:\n            if pos <= 0:\n                return None\n            pos -= 1\n            while pos > 0 and 0x80 <= s[pos] <= 0xBF:\n                pos -= 1\n            remaining -= 1\n        end = pos + 1\n''',
    '''        pos = i - 1\n        remaining = -n\n        while remaining > 0:\n            if pos <= 0:\n                return None\n            pos -= 1\n            while pos > 0 and 0x80 <= s[pos] <= 0xBF:\n                pos -= 1\n            if 0x80 <= s[pos] <= 0xBF:\n                raise LuaRuntimeError("initial position is a continuation byte")\n            remaining -= 1\n        end = pos + 1\n''',
)

print("round-three patch applied")
