from __future__ import annotations

from dataclasses import dataclass
from .errors import LuaSyntaxError


KEYWORDS = {
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function",
    "global", "goto", "if", "in", "local", "nil", "not", "or", "repeat", "return",
    "then", "true", "until", "while",
}


@dataclass(frozen=True, slots=True)
class Token:
    kind: str
    value: object
    line: int
    column: int


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.i = 0
        self.line = 1
        self.col = 1
        self.n = len(source)

    def _peek(self, offset=0):
        j = self.i + offset
        return self.source[j] if j < self.n else ""

    def _take(self):
        ch = self._peek()
        if not ch:
            return ""
        self.i += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def tokens(self):
        out = []
        while self.i < self.n:
            ch = self._peek()
            if ch.isspace():
                self._take()
                continue
            if ch == "-" and self._peek(1) == "-":
                self._take(); self._take()
                if self._peek() == "[" and self._peek(1) == "[":
                    self._take(); self._take()
                    while not (self._peek() == "]" and self._peek(1) == "]"):
                        if not self._peek():
                            raise LuaSyntaxError(f"unterminated long comment at line {self.line}")
                        self._take()
                    self._take(); self._take()
                else:
                    while self._peek() not in ("", "\n"):
                        self._take()
                continue
            line, col = self.line, self.col
            if ch.isalpha() or ch == "_":
                s = self._take()
                while self._peek().isalnum() or self._peek() == "_":
                    s += self._take()
                out.append(Token(s if s in KEYWORDS else "NAME", s, line, col))
                continue
            if ch.isdigit():
                s = self._take()
                while self._peek().isdigit():
                    s += self._take()
                if self._peek() == "." and self._peek(1) != ".":
                    s += self._take()
                    while self._peek().isdigit():
                        s += self._take()
                    out.append(Token("NUMBER", float(s), line, col))
                else:
                    out.append(Token("NUMBER", int(s), line, col))
                continue
            if ch in "'\"":
                quote = self._take()
                chars = bytearray()
                while True:
                    c = self._take()
                    if c == "":
                        raise LuaSyntaxError(f"unterminated string at line {line}")
                    if c == quote:
                        break
                    if c == "\\":
                        e = self._take()
                        escaped = {"a":"\a", "b":"\b", "f":"\f", "n":"\n", "r":"\r", "t":"\t", "v":"\v", "\\":"\\", "\"":"\"", "'":"'"}.get(e, e)
                        chars.extend(escaped.encode("utf-8"))
                    else:
                        chars.extend(c.encode("utf-8"))
                out.append(Token("STRING", bytes(chars), line, col))
                continue
            if ch == "[" and self._peek(1) == "[":
                self._take(); self._take()
                chars = bytearray()
                while not (self._peek() == "]" and self._peek(1) == "]"):
                    c = self._take()
                    if not c:
                        raise LuaSyntaxError(f"unterminated long string at line {line}")
                    chars.extend(c.encode("utf-8"))
                self._take(); self._take()
                out.append(Token("STRING", bytes(chars), line, col))
                continue
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
            raise LuaSyntaxError(f"unexpected character {ch!r} at line {line}, column {col}")
        out.append(Token("EOF", None, self.line, self.col))
        return out
