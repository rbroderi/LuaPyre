from __future__ import annotations

from dataclasses import dataclass

from .errors import LuaSyntaxError


KEYWORDS = {
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function",
    "goto", "if", "in", "local", "nil", "not", "or", "repeat", "return",
    "then", "true", "until", "while",
}

_INT_MAX = (1 << 63) - 1
_UINT_MASK = (1 << 64) - 1
_HEX = frozenset("0123456789abcdefABCDEF")
_SIMPLE_ESCAPES = {
    "a": 0x07,
    "b": 0x08,
    "f": 0x0C,
    "n": 0x0A,
    "r": 0x0D,
    "t": 0x09,
    "v": 0x0B,
    "\\": 0x5C,
    '"': 0x22,
    "'": 0x27,
}


@dataclass(frozen=True, slots=True)
class Token:
    kind: str
    value: object
    line: int
    column: int


class Lexer:
    def __init__(self, source: str, *, typed: bool = False):
        self.source = source
        self.keywords = KEYWORDS | {"global"} if typed else KEYWORDS
        self.i = 0
        self.line = 1
        self.col = 1
        self._last_newline = ""
        self.n = len(source)
        # Lua's standalone loader treats a Unix shebang as a skipped first
        # line. Preserve its line number while allowing real package scripts
        # to be executed directly by the embedding API.
        if source.startswith("#!"):
            while self._peek() and self._peek() not in "\r\n":
                self._take()

    def _peek(self, offset=0):
        j = self.i + offset
        return self.source[j] if j < self.n else ""

    def _take(self):
        ch = self._peek()
        if not ch:
            return ""
        self.i += 1
        if ch in "\r\n":
            # Lua treats CR, LF, CRLF, and LFCR as one newline sequence.
            if self._last_newline and self._last_newline != ch:
                self._last_newline = ""
            else:
                self.line += 1
                self._last_newline = ch
            self.col = 1
        else:
            self.col += 1
            self._last_newline = ""
        return ch

    def _syntax(self, message: str, line: int | None = None, col: int | None = None):
        line = self.line if line is None else line
        col = self.col if col is None else col
        raise LuaSyntaxError(f"{message} at line {line}, column {col}", line=line, column=col)

    @staticmethod
    def _extended_utf8(value: int) -> bytes:
        if value < 0 or value > 0x7FFFFFFF:
            raise ValueError("UTF-8 value too large")
        if value <= 0x7F:
            return bytes((value,))
        if value <= 0x7FF:
            count, prefix = 2, 0xC0
        elif value <= 0xFFFF:
            count, prefix = 3, 0xE0
        elif value <= 0x1FFFFF:
            count, prefix = 4, 0xF0
        elif value <= 0x3FFFFFF:
            count, prefix = 5, 0xF8
        else:
            count, prefix = 6, 0xFC
        out = bytearray(count)
        current = value
        for index in range(count - 1, 0, -1):
            out[index] = 0x80 | (current & 0x3F)
            current >>= 6
        out[0] = prefix | current
        return bytes(out)

    def _long_level(self, bracket: str = "[") -> int | None:
        if self._peek() != bracket:
            return None
        j = self.i + 1
        count = 0
        while j < self.n and self.source[j] == "=":
            count += 1
            j += 1
        if j < self.n and self.source[j] == bracket:
            return count
        return None

    def _consume_long(self, level: int, *, comment: bool, line: int, col: int) -> bytes | None:
        self._take()
        for _ in range(level):
            self._take()
        self._take()
        if self._peek() == "\r":
            self._take()
            if self._peek() == "\n":
                self._take()
        elif self._peek() == "\n":
            self._take()
            if self._peek() == "\r":
                self._take()

        chars = bytearray()
        while True:
            if not self._peek():
                kind = "comment" if comment else "string"
                self._syntax(f"unfinished long {kind} near <eof>", line, col)
            if self._peek() == "]":
                j = self.i + 1
                count = 0
                while j < self.n and self.source[j] == "=":
                    count += 1
                    j += 1
                if count == level and j < self.n and self.source[j] == "]":
                    self._take()
                    for _ in range(level):
                        self._take()
                    self._take()
                    return None if comment else bytes(chars)
            ch = self._take()
            if not comment:
                if ch == "\r":
                    if self._peek() == "\n":
                        self._take()
                    chars.append(0x0A)
                elif ch == "\n":
                    if self._peek() == "\r":
                        self._take()
                    chars.append(0x0A)
                else:
                    chars.extend(ch.encode("utf-8", "surrogateescape"))

    def _read_short_string(self, quote: str, line: int, col: int) -> bytes:
        self._take()
        literal_start = self.i
        chars = bytearray()
        while True:
            ch = self._peek()
            if not ch:
                self._syntax("unfinished string near <eof>", line, col)
            if ch == quote:
                self._take()
                return bytes(chars)
            if ch in "\r\n":
                self._syntax("unfinished string", line, col)
            if ch != "\\":
                chars.extend(self._take().encode("utf-8", "surrogateescape"))
                continue

            self._take()
            escape_line, escape_col = self.line, self.col
            escape_start = self.i - 1
            esc = self._peek()
            if not esc:
                self._syntax("unfinished string", line, col)
            if esc in _SIMPLE_ESCAPES:
                self._take()
                chars.append(_SIMPLE_ESCAPES[esc])
                continue
            if esc == "x":
                self._take()
                digits = ""
                for _ in range(2):
                    digit = self._take()
                    digits += digit
                    if digit not in _HEX:
                        fragment = self.source[escape_start:self.i]
                        self._syntax(
                            f"hexadecimal digit expected near '{fragment}'",
                            escape_line, escape_col,
                        )
                chars.append(int(digits, 16))
                continue
            if esc == "u":
                self._take()
                if self._peek() != "{":
                    fragment = self.source[literal_start:self.i + 1]
                    self._syntax(
                        f"missing '{{' near '{fragment}'", escape_line, escape_col
                    )
                self._take()
                digits = ""
                while self._peek() in _HEX:
                    digits += self._take()
                if not digits:
                    fragment = self.source[literal_start:self.i + 1]
                    self._syntax(
                        f"hexadecimal digit expected near '{fragment}'",
                        escape_line, escape_col,
                    )
                if self._peek() != "}":
                    fragment = self.source[literal_start:self.i + 1]
                    self._syntax(
                        f"missing '}}' near '{fragment}'", escape_line, escape_col
                    )
                self._take()
                value = int(digits, 16)
                try:
                    chars.extend(self._extended_utf8(value))
                except ValueError:
                    fragment = self.source[literal_start:self.i - 1]
                    self._syntax(
                        f"UTF-8 value too large near '{fragment}'",
                        escape_line, escape_col,
                    )
                continue
            if esc == "z":
                self._take()
                while self._peek() and self._peek().isspace():
                    self._take()
                continue
            if esc == "\n":
                self._take()
                if self._peek() == "\r":
                    self._take()
                chars.append(0x0A)
                continue
            if esc == "\r":
                self._take()
                if self._peek() == "\n":
                    self._take()
                chars.append(0x0A)
                continue
            if esc.isdigit():
                digits = ""
                for _ in range(3):
                    if not self._peek().isdigit():
                        break
                    digits += self._take()
                value = int(digits, 10)
                if value > 255:
                    end = self.i + int(self._peek() in "\"'")
                    fragment = self.source[escape_start:end]
                    self._syntax(
                        f"decimal escape too large near '{fragment}'",
                        escape_line, escape_col,
                    )
                chars.append(value)
                continue
            self._take()
            fragment = self.source[escape_start:self.i]
            self._syntax(
                f"invalid escape sequence near '{fragment}'",
                escape_line, escape_col,
            )

    def _read_number(self, line: int, col: int) -> Token:
        start = self.i
        lookahead = self._peek(1)
        hexadecimal = self._peek() == "0" and bool(lookahead) and lookahead in "xX"
        if hexadecimal:
            self._take(); self._take()
            while self._peek() in _HEX:
                self._take()
            if self._peek() == "." and self._peek(1) != ".":
                self._take()
                while self._peek() in _HEX:
                    self._take()
            if self._peek() and self._peek() in "pP":
                self._take()
                if self._peek() and self._peek() in "+-":
                    self._take()
                while self._peek().isdigit():
                    self._take()
        else:
            if self._peek() == ".":
                self._take()
            while self._peek().isdigit():
                self._take()
            if self._peek() == "." and self._peek(1) != ".":
                self._take()
                while self._peek().isdigit():
                    self._take()
            if self._peek() and self._peek() in "eE":
                self._take()
                if self._peek() and self._peek() in "+-":
                    self._take()
                while self._peek().isdigit():
                    self._take()

        text = self.source[start:self.i]
        if self._peek().isalpha() or self._peek() == "_":
            self._syntax("malformed number", line, col)
        try:
            if hexadecimal:
                body = text[2:]
                floating = "." in body or "p" in body.lower()
                if floating:
                    value = float.fromhex(text)
                else:
                    # Lua accumulates hexadecimal integers in lua_Unsigned;
                    # overflow is modular before the final signed cast.
                    integer = int(body, 16) & _UINT_MASK
                    value = integer if integer <= _INT_MAX else integer - (1 << 64)
            else:
                floating = "." in text or "e" in text.lower()
                if floating:
                    value = float(text)
                else:
                    integer = int(text, 10)
                    value = integer if integer <= _INT_MAX else float(text)
        except (ValueError, OverflowError):
            self._syntax("malformed number", line, col)
        return Token("NUMBER", value, line, col)

    def tokens(self):
        out = []
        while self.i < self.n:
            ch = self._peek()
            if ch.isspace():
                self._take()
                continue
            if ch == "-" and self._peek(1) == "-":
                self._take(); self._take()
                if self._peek() == "[":
                    level = self._long_level("[")
                    if level is not None:
                        self._consume_long(level, comment=True, line=self.line, col=self.col)
                        continue
                while self._peek() not in ("", "\r", "\n"):
                    self._take()
                continue

            line, col = self.line, self.col
            if ch.isalpha() or ch == "_":
                s = self._take()
                while self._peek().isalnum() or self._peek() == "_":
                    s += self._take()
                out.append(Token(s if s in self.keywords else "NAME", s, line, col))
                continue
            if ch.isdigit() or (ch == "." and self._peek(1).isdigit()):
                out.append(self._read_number(line, col))
                continue
            if ch in "'\"":
                out.append(Token("STRING", self._read_short_string(ch, line, col), line, col))
                continue
            if ch == "[":
                level = self._long_level("[")
                if level is not None:
                    value = self._consume_long(level, comment=False, line=line, col=col)
                    out.append(Token("STRING", value, line, col))
                    continue
                if self._peek(1) == "=":
                    self._syntax("invalid long string delimiter", line, col)

            three = self.source[self.i:self.i+3]
            two = self.source[self.i:self.i+2]
            if three == "...":
                self._take(); self._take(); self._take()
                out.append(Token("...", "...", line, col))
                continue
            if two in ("==", "~=", "<=", ">=", "//", "<<", ">>", "..", "::"):
                self._take(); self._take()
                out.append(Token(two, two, line, col))
                continue
            if ch in "+-*/%^#&|~<>=(){}[];,:?.":
                self._take()
                out.append(Token(ch, ch, line, col))
                continue
            self._syntax(f"unexpected character {ch!r}", line, col)
        out.append(Token("EOF", None, self.line, self.col))
        return out
