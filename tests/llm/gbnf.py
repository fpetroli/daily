"""A small GBNF interpreter for the tests: it tells whether a grammar accepts a string.

It covers the subset `daily.llm.grammar` generates (rules, literals, character classes,
groups, alternatives, `?` `*` `+` and `{n}`/`{n,m}`): enough to check the grammar without a
server. Backtracking with generators: fine for short strings.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

_TOKEN = re.compile(
    r"""\s*(?:
    (?P<literal>"(?:[^"\\]|\\.)*")
    |(?P<klass>\[(?:[^\]\\]|\\.)*\])
    |(?P<repeat>\{\d+(?:,\d*)?\})
    |(?P<name>[a-z][a-z0-9-]*)
    |(?P<op>[()|*+?])
    )""",
    re.VERBOSE,
)


def _unescape(text: str) -> list[str]:
    """Characters of a literal or of a class, with escape sequences resolved."""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\\":
            nxt = text[i + 1]
            if nxt == "x":
                out.append(chr(int(text[i + 2 : i + 4], 16)))
                i += 4
                continue
            out.append({"n": "\n", "r": "\r", "t": "\t"}.get(nxt, nxt))
            i += 2
        else:
            out.append(text[i])
            i += 1
    return out


@dataclass
class _Parser:
    tokens: list[tuple[str, str]]
    pos: int = 0

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def alternatives(self) -> Any:
        options = [self.sequence()]
        while self.peek() == ("op", "|"):
            self.pos += 1
            options.append(self.sequence())
        return options[0] if len(options) == 1 else ("alt", options)

    def sequence(self) -> Any:
        items: list[Any] = []
        while (token := self.peek()) is not None and token not in {("op", "|"), ("op", ")")}:
            items.append(self.repeated())
        return ("seq", items)

    def repeated(self) -> Any:
        node = self.atom()
        while (token := self.peek()) is not None:
            kind, text = token
            if kind == "op" and text in "*+?":
                low, high = {"*": (0, None), "+": (1, None), "?": (0, 1)}[text]
            elif kind == "repeat":
                low_s, _, high_s = text[1:-1].partition(",")
                low = int(low_s)
                high = int(high_s) if high_s else (None if "," in text else low)
            else:
                break
            self.pos += 1
            node = ("rep", node, low, high)
        return node

    def atom(self) -> Any:
        kind, text = self.tokens[self.pos]
        self.pos += 1
        if kind == "literal":
            return ("lit", "".join(_unescape(text[1:-1])))
        if kind == "klass":
            body = text[1:-1]
            negate = body.startswith("^")
            chars = _unescape(body[1:] if negate else body)
            ranges: list[tuple[str, str]] = []
            i = 0
            while i < len(chars):
                if i + 2 < len(chars) and chars[i + 1] == "-":
                    ranges.append((chars[i], chars[i + 2]))
                    i += 3
                else:
                    ranges.append((chars[i], chars[i]))
                    i += 1
            return ("class", negate, ranges)
        if kind == "name":
            return ("ref", text)
        assert (kind, text) == ("op", "("), (kind, text)
        node = self.alternatives()
        assert self.peek() == ("op", ")")
        self.pos += 1
        return node


def _tokens(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text.rstrip()):
        match = _TOKEN.match(text, pos)
        if match is None:
            raise ValueError(f"invalid GBNF at {text[pos : pos + 30]!r}")
        kind = match.lastgroup or ""
        out.append((kind, match.group(kind)))
        pos = match.end()
    return out


class Grammar:
    def __init__(self, text: str) -> None:
        self.rules: dict[str, Any] = {}
        for line in text.strip().splitlines():
            name, _, body = line.partition("::=")
            parser = _Parser(_tokens(body))
            self.rules[name.strip()] = parser.alternatives()
            assert parser.pos == len(parser.tokens), f"unread rest in {name}"
        used = {n for r in self.rules.values() for n in _refs(r)}
        assert used <= set(self.rules), f"undefined rules: {used - set(self.rules)}"

    def accepts(self, text: str) -> bool:
        return any(end == len(text) for end in self._match(("ref", "root"), text, 0))

    def _match(self, node: Any, text: str, i: int) -> Iterator[int]:
        kind = node[0]
        if kind == "lit":
            if text.startswith(node[1], i):
                yield i + len(node[1])
        elif kind == "class":
            if i < len(text):
                inside = any(lo <= text[i] <= hi for lo, hi in node[2])
                if inside != node[1]:
                    yield i + 1
        elif kind == "ref":
            yield from self._match(self.rules[node[1]], text, i)
        elif kind == "alt":
            for option in node[1]:
                yield from self._match(option, text, i)
        elif kind == "seq":
            yield from self._sequence(node[1], text, i)
        elif kind == "rep":
            yield from self._repeat(node[1], node[2], node[3], text, i, 0)

    def _sequence(self, items: list[Any], text: str, i: int) -> Iterator[int]:
        if not items:
            yield i
            return
        for end in self._match(items[0], text, i):
            yield from self._sequence(items[1:], text, end)

    def _repeat(
        self, node: Any, low: int, high: int | None, text: str, i: int, count: int
    ) -> Iterator[int]:
        if count >= low:
            yield i
        if high is not None and count >= high:
            return
        for end in self._match(node, text, i):
            if end > i:  # a repetition that consumes nothing leads nowhere
                yield from self._repeat(node, low, high, text, end, count + 1)


def _refs(node: Any) -> Iterator[str]:
    if node[0] == "ref":
        yield node[1]
    elif node[0] in {"alt", "seq"}:
        for child in node[1]:
            yield from _refs(child)
    elif node[0] == "rep":
        yield from _refs(node[1])
