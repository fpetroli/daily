"""Compact GBNF grammar for `llama-server`, derived from the tool call schema.

The server's converter (`response_format`) allows spaces and newlines between tokens, and the
model takes advantage of it: many outputs are indented JSON, up to 70% more tokens, and
generation (~9 tokens per second) is the biggest cost of a turn. This grammar says the same
things as the schema (allowed tools and fields, value shapes, integer bounds) but on one line,
with no spaces.

Only the subset the schema uses is translated; any other construct raises `ValueError`,
because a constraint silently ignored would let invalid values through.
"""

from __future__ import annotations

import json
import re
from typing import Any

# A JSON string character with its escape sequences, as in the llama.cpp converter. Unlike
# that one, a string is never empty and never starts with a space: a `""` title (which the
# model writes to "omit" a title) is always an error, and validation would reject it only
# after the turn is lost.
_STRING_RULES = (
    r'''string ::= "\"" [^"\\ \x7F\x00-\x1F] char* "\""''',
    r"""char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})""",
)

_LITERAL_CHARACTER = re.compile(r"[A-Za-z0-9:\- ]")
_CHARACTER_CLASS = re.compile(r"\[\^?[A-Za-z0-9-]+\]")
_REPETITION = re.compile(r"\{\d+(,\d*)?\}")


def _literal(text: str) -> str:
    """A GBNF literal."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def regex_to_gbnf(pattern: str) -> str:
    """Translates an anchored regular expression (`^…$`) of the subset we use: character
    classes, groups, alternatives, `{n}` repetitions and literal characters."""
    if not (pattern.startswith("^") and pattern.endswith("$")):
        raise ValueError(f"pattern not anchored: {pattern!r}")
    body = pattern[1:-1]
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "[":
            end = body.find("]", i)
            if end < 0 or not _CHARACTER_CLASS.fullmatch(body[i : end + 1]):
                raise ValueError(f"unsupported character class in {pattern!r}")
            out.append(body[i : end + 1])
            i = end + 1
        elif ch == "{":
            match = _REPETITION.match(body, i)
            if match is None or not out:
                raise ValueError(f"unsupported repetition in {pattern!r}")
            out[-1] += match.group()  # attached to the element it repeats, with no space
            i = match.end()
        elif ch in "()|":
            out.append(ch)
            i += 1
        elif _LITERAL_CHARACTER.fullmatch(ch):
            out.append(_literal(ch))
            i += 1
        else:
            raise ValueError(f"unsupported character {ch!r} in {pattern!r}")
    return " ".join(out)


def _digits(low: str, high: str) -> str:
    return low if low == high else f"[{low}-{high}]"


def _same_length(low: str, high: str) -> list[str]:
    """Alternatives (digit sequences and classes) for the numbers of equal length in [low, high]."""
    if len(low) == 1:
        return [_digits(low, high)]
    if low[0] == high[0]:
        return [low[0] + rest for rest in _same_length(low[1:], high[1:])]
    tail = len(low) - 1
    parts = [low[0] + rest for rest in _same_length(low[1:], "9" * tail)]
    if int(high[0]) - int(low[0]) > 1:
        middle = _digits(str(int(low[0]) + 1), str(int(high[0]) - 1))
        parts.append(f"{middle}[0-9]{{{tail}}}")
    parts += [high[0] + rest for rest in _same_length("0" * tail, high[1:])]
    return parts


def _unsigned_range(low: int, high: int) -> str:
    alternatives: list[str] = []
    for length in range(len(str(low)), len(str(high)) + 1):
        start = max(low, 0 if length == 1 else 10 ** (length - 1))
        end = min(high, 10**length - 1)
        alternatives += _same_length(str(start), str(end))
    return "(" + "|".join(alternatives) + ")"


def integer_regex(low: int, high: int) -> str:
    """Regular expression (without anchors) of the decimal integers in [low, high], with no
    leading zeros."""
    if low > high:
        raise ValueError(f"empty range: {low}..{high}")
    parts: list[str] = []
    if high >= 0:
        parts.append(_unsigned_range(max(low, 0), high))
    if low < 0:
        parts.append("-" + _unsigned_range(max(-high, 1), -low))
    return "(" + "|".join(parts) + ")"


def _integer_bounds(spec: dict[str, Any]) -> tuple[int, int]:
    low = spec.get("minimum", spec["exclusiveMinimum"] + 1 if "exclusiveMinimum" in spec else None)
    high = spec.get("maximum", spec["exclusiveMaximum"] - 1 if "exclusiveMaximum" in spec else None)
    if low is None or high is None:
        raise ValueError(f"integer without bounds: {spec}")
    return low, high


class _Builder:
    def __init__(self) -> None:
        self.rules: dict[str, str] = {}

    def _shared(self, prefix: str, expression: str) -> str:
        for name, existing in self.rules.items():
            if existing == expression and name.startswith(prefix + "-"):
                return name
        count = sum(1 for name in self.rules if name.startswith(prefix + "-"))
        name = f"{prefix}-{count + 1}"
        self.rules[name] = expression
        return name

    def value(self, spec: dict[str, Any]) -> str:
        if "const" in spec:
            return _literal(json.dumps(spec["const"], ensure_ascii=False))
        if "enum" in spec:
            options = (_literal(json.dumps(v, ensure_ascii=False)) for v in spec["enum"])
            return "(" + " | ".join(options) + ")"
        kind = spec.get("type")
        if kind == "string":
            if "pattern" not in spec:
                return "string"
            body = regex_to_gbnf(spec["pattern"])
            return self._shared("pattern", f'"\\"" ({body}) "\\""')
        if kind == "integer":
            return self._shared(
                "integer", regex_to_gbnf(f"^{integer_regex(*_integer_bounds(spec))}$")
            )
        if kind == "boolean":
            return '("true" | "false")'
        raise ValueError(f"unsupported type: {spec}")

    def variant(self, schema: dict[str, Any]) -> str:
        properties = schema["properties"]
        tool = properties["tool"]["const"]
        parts = [_literal('{"tool":' + json.dumps(tool))]
        for name, spec in properties.items():
            if name != "tool":
                parts.append(f"({_literal(',' + json.dumps(name) + ':')} {self.value(spec)})?")
        parts.append(_literal("}"))
        rule = "call-" + tool.replace("_", "-")
        self.rules[rule] = " ".join(parts)
        return rule


def compact_grammar(schema: dict[str, Any]) -> str:
    """The GBNF grammar of the tool call union of `tool_call_json_schema`, with no spaces.

    Keys follow the schema order (as in the server's converter), so the prompt examples must
    follow it too: a test checks it.
    """
    builder = _Builder()
    calls = [builder.variant(variant) for variant in schema["oneOf"]]
    lines = ["root ::= " + " | ".join(calls)]
    lines += [f"{name} ::= {expression}" for name, expression in builder.rules.items()]
    lines.extend(_STRING_RULES)
    return "\n".join(lines) + "\n"
