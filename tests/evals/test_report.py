import json

from daily.evals.cases import Case, Expect, Seed, Turn
from daily.evals.report import percentile, render, summarize, to_json
from daily.evals.runner import CaseResult, TurnRecord


def result(
    case_id, section, *, passed=True, split="dev", temporal=None, latencies=(1.0,), calls=True
):
    from datetime import datetime

    case = Case(
        case_id,
        section,
        split,
        datetime(2026, 9, 21, 15, 30),
        Seed(),
        (Turn("x"),),
        Expect(),
        temporal,
    )
    expected = {"tool": "add_event"} if calls else None
    turns = tuple(
        TurnRecord(
            "x", "event_created", "ok", (), ms, expected, {"tool": "add_event"} if passed else None
        )
        for ms in latencies
    )
    return CaseResult(case, () if passed else ("schedule: missing [x]",), turns)


def test_nearest_rank_percentile():
    assert percentile([], 0.5) == 0.0
    assert percentile([5.0], 0.95) == 5.0
    values = [float(n) for n in range(1, 101)]
    assert percentile(values, 0.5) == 50.0
    assert percentile(values, 0.95) == 95.0
    assert percentile(list(reversed(values)), 0.95) == 95.0


def test_summary_counts_by_section_split_and_temporal_kind():
    results = [
        result("a", "creation"),
        result("b", "creation", passed=False),
        result("c", "temporal", temporal="duration", split="holdout"),
        result("d", "temporal", temporal="duration", passed=False),
        result("e", "temporal", temporal="time-band"),
    ]

    summary = summarize(results, "oracle")

    assert (summary.overall.total, summary.overall.passed) == (5, 3)
    assert summary.by_section["creation"].percent == 50.0
    assert (summary.by_split["dev"].total, summary.by_split["holdout"].total) == (4, 1)
    assert summary.by_temporal["duration"].percent == 50.0
    assert summary.by_temporal["time-band"].percent == 100.0
    assert [r.case.id for r in summary.failures] == ["b", "d"]


def test_tool_call_diagnostic_counts_only_turns_with_an_expected_call():
    results = [result("a", "s"), result("b", "s", passed=False), result("c", "s", calls=False)]

    summary = summarize(results, "oracle")

    assert (summary.tool_calls.total, summary.tool_calls.passed) == (2, 1)


def _with_failures(base, *failures):
    return CaseResult(base.case, failures, base.turns)


def test_unwanted_writes_are_only_the_extra_ones():
    results = [
        result("ok", "s"),
        _with_failures(result("missing", "s"), "schedule: missing [('x', 'a', 'b')]"),
        _with_failures(result("moved", "s"), "schedule: missing [x]", "schedule: extra [y]"),
        _with_failures(result("rule", "s"), "rules: extra [('a', 'b')]"),
        _with_failures(result("risposta", "s"), "turn 1: the reply does not contain 'Ok'"),
    ]

    summary = summarize(results, "llama")

    assert [r.wrote_unexpected for r in results] == [False, False, True, True, False]
    assert summary.unexpected_writes == 2
    assert "Unwanted writes (something extra in the schedule or among the rules): 2 of 5" in render(
        summary
    )
    assert to_json(summary, results)["unexpected_writes"] == 2


def test_latency_is_per_turn():
    summary = summarize([result("a", "s", latencies=(1.0, 2.0, 3.0, 4.0, 100.0))], "p")

    assert summary.turns == 5
    assert (summary.latency_p50_ms, summary.latency_p95_ms) == (3.0, 100.0)


def test_text_shows_percentages_and_failures():
    results = [result("a", "creation"), result("b", "creation", passed=False)]

    text = render(summarize(results, "llama"))

    assert "# Eval — parser: llama" in text
    assert "Cases: 2 · correct: 1 (50.0%)" in text
    assert "| creation | 2 | 1 | 50.0 |" in text
    assert "## Failed cases (1)" in text and "**b** (creation, dev)" in text
    assert "schedule: missing [x]" in text


def test_without_failures_the_text_has_no_failed_cases_section():
    assert "Failed cases" not in render(summarize([result("a", "s")], "oracle"))


def test_json_is_serializable_and_reports_every_case():
    results = [result("a", "creation"), result("b", "creation", passed=False)]

    data = to_json(summarize(results, "oracle"), results)
    json.dumps(data)

    assert data["overall"] == {"total": 2, "passed": 1, "percent": 50.0}
    assert [c["id"] for c in data["cases"]] == ["a", "b"]
    assert data["cases"][1]["failures"] == ["schedule: missing [x]"]
