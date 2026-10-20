"""The report of a run: percentages per section, temporal expressions, latency and failures."""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from daily.evals.runner import CaseResult


def percentile(values: list[float], fraction: float) -> float:
    """Nearest rank: 0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


@dataclass(frozen=True, slots=True)
class Tally:
    total: int = 0
    passed: int = 0

    @property
    def percent(self) -> float:
        return 100.0 * self.passed / self.total if self.total else 0.0

    def add(self, passed: bool) -> "Tally":
        return Tally(self.total + 1, self.passed + int(passed))


@dataclass(frozen=True, slots=True)
class Summary:
    parser: str
    overall: Tally
    turns: int
    by_section: dict[str, Tally]
    by_split: dict[str, Tally]
    by_temporal: dict[str, Tally]
    tool_calls: Tally  # same tool as the expected tool call
    exact_calls: Tally  # tool call identical to the expected one
    latency_p50_ms: float
    latency_p95_ms: float
    failures: tuple[CaseResult, ...]
    unexpected_writes: int = 0


def summarize(results: list[CaseResult], parser: str) -> Summary:
    overall = Tally()
    sections: dict[str, Tally] = defaultdict(Tally)
    splits: dict[str, Tally] = defaultdict(Tally)
    temporal: dict[str, Tally] = defaultdict(Tally)
    tools, exact = Tally(), Tally()
    latencies: list[float] = []
    for result in results:
        overall = overall.add(result.passed)
        sections[result.case.section] = sections[result.case.section].add(result.passed)
        splits[result.case.split] = splits[result.case.split].add(result.passed)
        if result.case.temporal:
            temporal[result.case.temporal] = temporal[result.case.temporal].add(result.passed)
        for turn in result.turns:
            latencies.append(turn.latency_ms)
            if turn.tool_match is not None:
                tools = tools.add(turn.tool_match)
                exact = exact.add(bool(turn.exact_match))
    return Summary(
        parser=parser,
        overall=overall,
        turns=len(latencies),
        by_section=dict(sorted(sections.items())),
        by_split=dict(sorted(splits.items())),
        by_temporal=dict(sorted(temporal.items())),
        tool_calls=tools,
        exact_calls=exact,
        latency_p50_ms=percentile(latencies, 0.5),
        latency_p95_ms=percentile(latencies, 0.95),
        failures=tuple(r for r in results if not r.passed),
        unexpected_writes=sum(1 for r in results if r.wrote_unexpected),
    )


def _table(title: str, rows: dict[str, Tally]) -> list[str]:
    if not rows:
        return []
    lines = [f"| {title} | Cases | Correct | % |", "|---|---:|---:|---:|"]
    lines += [f"| {name} | {t.total} | {t.passed} | {t.percent:.1f} |" for name, t in rows.items()]
    return lines + [""]


def render(summary: Summary, *, notes: tuple[str, ...] = (), max_failures: int = 40) -> str:
    total = summary.overall
    lines = [
        f"# Eval — parser: {summary.parser}",
        "",
        f"Cases: {total.total} · correct: {total.passed} ({total.percent:.1f}%) · "
        f"turns: {summary.turns}",
        "",
    ]
    lines += _table("Section", summary.by_section)
    lines += _table("Split", summary.by_split)
    lines += _table("Temporal, by kind of expression", summary.by_temporal)
    calls = summary.tool_calls
    if calls.total:
        lines += [
            "Tool calls (diagnostic, on the turns with an expected tool call): same tool "
            f"{calls.passed}/{calls.total} ({calls.percent:.1f}%), identical "
            f"{summary.exact_calls.passed}/{summary.exact_calls.total} "
            f"({summary.exact_calls.percent:.1f}%).",
            "",
        ]
    lines += [
        f"Unwanted writes (something extra in the schedule or among the rules): "
        f"{summary.unexpected_writes} of {total.total} cases.",
        "",
        f"Latency per turn: p50 {summary.latency_p50_ms:.1f} ms · "
        f"p95 {summary.latency_p95_ms:.1f} ms",
        "",
    ]
    if notes:
        lines += [*notes, ""]
    if summary.failures:
        lines.append(f"## Failed cases ({len(summary.failures)})")
        for result in summary.failures[:max_failures]:
            lines.append(f"- **{result.case.id}** ({result.case.section}, {result.case.split})")
            lines += [f"  - {problem}" for problem in result.failures]
        if len(summary.failures) > max_failures:
            lines.append(f"- … {len(summary.failures) - max_failures} more")
    return "\n".join(lines).rstrip() + "\n"


def to_json(summary: Summary, results: list[CaseResult]) -> dict[str, Any]:
    def tally(t: Tally) -> dict[str, Any]:
        return {"total": t.total, "passed": t.passed, "percent": round(t.percent, 2)}

    return {
        "parser": summary.parser,
        "overall": tally(summary.overall),
        "by_section": {k: tally(v) for k, v in summary.by_section.items()},
        "by_split": {k: tally(v) for k, v in summary.by_split.items()},
        "by_temporal": {k: tally(v) for k, v in summary.by_temporal.items()},
        "tool_calls": tally(summary.tool_calls),
        "exact_calls": tally(summary.exact_calls),
        "unexpected_writes": summary.unexpected_writes,
        "latency_ms": {"p50": summary.latency_p50_ms, "p95": summary.latency_p95_ms},
        "cases": [
            {
                "id": r.case.id,
                "section": r.case.section,
                "split": r.case.split,
                "passed": r.passed,
                "failures": list(r.failures),
                "turns": [
                    {
                        "say": t.message,
                        "kind": t.kind,
                        "reply": t.reply,
                        "latency_ms": round(t.latency_ms, 2),
                        "expected_call": t.expected_call,
                        "actual_call": t.actual_call,
                        **({"parse_detail": t.parse_detail} if t.parse_detail else {}),
                    }
                    for t in r.turns
                ],
            }
            for r in results
        ],
    }
