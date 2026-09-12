from __future__ import annotations

from dataclasses import dataclass

from .errors import LuaRuntimeError


@dataclass(frozen=True, slots=True)
class Capture:
    start: int
    end: int | None
    position: bool = False


@dataclass(frozen=True, slots=True)
class PatternMatch:
    start: int
    end: int
    captures: tuple[Capture, ...]

    def capture_values(self, subject: bytes):
        values = []
        for capture in self.captures:
            if capture.position:
                values.append(capture.start + 1)
            else:
                if capture.end is None:
                    raise LuaRuntimeError("unfinished capture")
                values.append(subject[capture.start:capture.end])
        return tuple(values)


class LuaPattern:
    """Backtracking matcher for Lua 5.5 byte patterns.

    Lua patterns are deliberately smaller than regular expressions. Keeping a
    native matcher avoids subtly incorrect regex translations, especially for
    frontier patterns, balanced matches, position captures, and Lua's minimal
    ``-`` repetition.

    A matcher also retains the endpoint of its previous successful search.
    Lua's ``gmatch`` and ``gsub`` reuse one MatchState and deliberately reject a
    second match ending at the same position (the 5.3.3+ empty-match rule).
    ``find`` and ``match`` create a matcher for one search, so this state is not
    observable there.
    """

    MAX_CAPTURES = 32

    def __init__(self, pattern: bytes):
        self.pattern = pattern
        self._last_end: int | None = None
        self._anchored_searched = False

    @staticmethod
    def _class_match(ch: int, cls: int) -> bool:
        lower = cls | 0x20
        if lower == ord("a"):
            result = (65 <= ch <= 90) or (97 <= ch <= 122)
        elif lower == ord("c"):
            result = ch < 32 or ch == 127
        elif lower == ord("d"):
            result = 48 <= ch <= 57
        elif lower == ord("g"):
            result = 33 <= ch <= 126
        elif lower == ord("l"):
            result = 97 <= ch <= 122
        elif lower == ord("p"):
            result = (33 <= ch <= 47) or (58 <= ch <= 64) or (91 <= ch <= 96) or (123 <= ch <= 126)
        elif lower == ord("s"):
            result = ch in b" \f\n\r\t\v"
        elif lower == ord("u"):
            result = 65 <= ch <= 90
        elif lower == ord("w"):
            result = (48 <= ch <= 57) or (65 <= ch <= 90) or (97 <= ch <= 122)
        elif lower == ord("x"):
            result = (48 <= ch <= 57) or (65 <= ch <= 70) or (97 <= ch <= 102)
        elif lower == ord("z"):
            result = ch == 0
        else:
            return ch == cls
        return not result if 65 <= cls <= 90 else result

    def _class_end(self, p: int) -> int:
        pattern = self.pattern
        if p >= len(pattern):
            raise LuaRuntimeError("malformed pattern")
        byte = pattern[p]
        if byte == ord("%"):
            if p + 1 >= len(pattern):
                raise LuaRuntimeError("malformed pattern (ends with '%')")
            return p + 2
        if byte != ord("["):
            return p + 1
        index = p + 1
        if index < len(pattern) and pattern[index] == ord("^"):
            index += 1
        if index < len(pattern) and pattern[index] == ord("]"):
            index += 1
        while index < len(pattern):
            if pattern[index] == ord("]"):
                return index + 1
            if pattern[index] == ord("%") and index + 1 < len(pattern):
                index += 2
            else:
                index += 1
        raise LuaRuntimeError("malformed pattern (missing ']')")

    def _set_match(self, ch: int, start: int, end: int) -> bool:
        pattern = self.pattern
        index = start + 1
        negate = index < end and pattern[index] == ord("^")
        if negate:
            index += 1
        matched = False
        if index < end - 1 and pattern[index] == ord("]"):
            if ch == ord("]"):
                matched = True
            index += 1
        while index < end - 1:
            current = pattern[index]
            if current == ord("%") and index + 1 < end - 1:
                if self._class_match(ch, pattern[index + 1]):
                    matched = True
                index += 2
                continue
            if index + 2 < end - 1 and pattern[index + 1] == ord("-"):
                if current <= ch <= pattern[index + 2]:
                    matched = True
                index += 3
                continue
            if current == ch:
                matched = True
            index += 1
        return not matched if negate else matched

    def _single_match(self, ch: int, p: int, ep: int) -> bool:
        pattern = self.pattern
        byte = pattern[p]
        if byte == ord("."):
            return True
        if byte == ord("%"):
            return self._class_match(ch, pattern[p + 1])
        if byte == ord("["):
            return self._set_match(ch, p, ep)
        return ch == byte

    @staticmethod
    def _last_open_capture(captures: tuple[Capture, ...]) -> int:
        for index in range(len(captures) - 1, -1, -1):
            capture = captures[index]
            if capture.end is None and not capture.position:
                return index
        return -1

    def _balance(self, subject: bytes, s: int, p: int):
        pattern = self.pattern
        if p + 3 >= len(pattern):
            raise LuaRuntimeError("malformed pattern (missing arguments to '%b')")
        opening, closing = pattern[p + 2], pattern[p + 3]
        if s >= len(subject) or subject[s] != opening:
            return None
        depth = 1
        pos = s + 1
        while pos < len(subject):
            byte = subject[pos]
            if byte == closing:
                depth -= 1
                if depth == 0:
                    return pos + 1
            elif byte == opening:
                depth += 1
            pos += 1
        return None

    def _match(self, subject: bytes, s: int, p: int, captures: tuple[Capture, ...]):
        pattern = self.pattern
        while True:
            if p >= len(pattern):
                return s, captures

            if pattern[p] == ord("$") and p + 1 == len(pattern):
                return (s, captures) if s == len(subject) else None

            if pattern[p] == ord("("):
                if len(captures) >= self.MAX_CAPTURES:
                    raise LuaRuntimeError("too many captures")
                if p + 1 < len(pattern) and pattern[p + 1] == ord(")"):
                    captures = captures + (Capture(s, s, True),)
                    p += 2
                    continue
                result = self._match(subject, s, p + 1, captures + (Capture(s, None),))
                return result

            if pattern[p] == ord(")"):
                index = self._last_open_capture(captures)
                if index < 0:
                    raise LuaRuntimeError("invalid pattern capture")
                updated = list(captures)
                current = updated[index]
                updated[index] = Capture(current.start, s, False)
                return self._match(subject, s, p + 1, tuple(updated))

            if pattern[p] == ord("%") and p + 1 < len(pattern):
                special = pattern[p + 1]
                if special == ord("b"):
                    end = self._balance(subject, s, p)
                    if end is None:
                        return None
                    s, p = end, p + 4
                    continue
                if special == ord("f"):
                    set_start = p + 2
                    if set_start >= len(pattern) or pattern[set_start] != ord("["):
                        raise LuaRuntimeError("missing '[' after '%f' in pattern")
                    set_end = self._class_end(set_start)
                    previous = subject[s - 1] if s > 0 else 0
                    current = subject[s] if s < len(subject) else 0
                    if self._set_match(previous, set_start, set_end) or not self._set_match(current, set_start, set_end):
                        return None
                    p = set_end
                    continue
                if 49 <= special <= 57:
                    index = special - 49
                    if index >= len(captures):
                        raise LuaRuntimeError("invalid capture index")
                    capture = captures[index]
                    if capture.position or capture.end is None:
                        raise LuaRuntimeError("invalid capture index")
                    text = subject[capture.start:capture.end]
                    if not subject.startswith(text, s):
                        return None
                    s += len(text)
                    p += 2
                    continue

            ep = self._class_end(p)
            matched = s < len(subject) and self._single_match(subject[s], p, ep)
            suffix = pattern[ep] if ep < len(pattern) else None

            if suffix == ord("?"):
                if matched:
                    result = self._match(subject, s + 1, ep + 1, captures)
                    if result is not None:
                        return result
                p = ep + 1
                continue

            if suffix in (ord("*"), ord("+")):
                minimum = 1 if suffix == ord("+") else 0
                maximum = 0
                pos = s
                while pos < len(subject) and self._single_match(subject[pos], p, ep):
                    pos += 1
                    maximum += 1
                if maximum < minimum:
                    return None
                for count in range(maximum, minimum - 1, -1):
                    result = self._match(subject, s + count, ep + 1, captures)
                    if result is not None:
                        return result
                return None

            if suffix == ord("-"):
                pos = s
                while True:
                    result = self._match(subject, pos, ep + 1, captures)
                    if result is not None:
                        return result
                    if pos >= len(subject) or not self._single_match(subject[pos], p, ep):
                        return None
                    pos += 1

            if not matched:
                return None
            s += 1
            p = ep

    def search(self, subject: bytes, init: int = 0, *, gmatch=False):
        pattern = self.pattern
        anchored = bool(pattern) and pattern[0] == ord("^") and not gmatch
        if anchored and self._anchored_searched:
            return None
        if anchored:
            self._anchored_searched = True
        pattern_start = 1 if anchored else 0
        positions = (init,) if anchored else range(init, len(subject) + 1)
        for start in positions:
            result = self._match(subject, start, pattern_start, ())
            if result is None:
                continue
            end, captures = result
            if self._last_end is not None and end == self._last_end:
                continue
            self._last_end = end
            return PatternMatch(start, end, captures)
        return None
