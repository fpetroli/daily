"""The real cases in `evals/cases`: the gate the eval set must pass.

The "oracle" parser returns the expected tool call of every turn: if it scores 100%, the
runner, the adapters, the templates, the history and the cases themselves are consistent. It
is the condition for a percentage measured with the real model to mean anything.
"""

from collections import Counter
from pathlib import Path

import pytest

from daily.agent.schemas import TOOL_NAMES
from daily.evals.cases import load_cases
from daily.evals.parsers import PARSERS
from daily.evals.report import summarize
from daily.evals.runner import run_cases

CASES_DIR = Path(__file__).resolve().parents[2] / "evals" / "cases"
SECTIONS = {
    "creation", "deletion", "multiturn", "no_action", "reads", "registers", "rules", "temporal",
    "update",
}  # fmt: skip
TEMPORAL_TYPES = {
    "relative-date", "weekday", "absolute-date", "compound-date", "numeric-time",
    "colloquial-time", "relative-time", "duration", "time-band",
}  # fmt: skip


@pytest.fixture(scope="module")
def cases():
    return load_cases(CASES_DIR)


def test_all_sections_and_enough_cases_are_present(cases):
    assert {c.section for c in cases} == SECTIONS
    assert len(cases) >= 150
    assert sum(len(c.turns) for c in cases) >= 180


def test_multi_turn_is_first_class(cases):
    multi = [c for c in cases if len(c.turns) > 1]

    assert len(multi) >= 20
    assert {c.section for c in multi} == {"multiturn"}


def test_every_schema_tool_is_exercised_by_at_least_one_case(cases):
    used = {t.call["tool"] for c in cases for t in c.turns if t.call}

    assert used == set(TOOL_NAMES)


def test_temporal_expressions_are_labelled_by_kind(cases):
    temporal = [c for c in cases if c.section == "temporal"]

    assert len(temporal) >= 50
    assert {c.temporal for c in temporal} == TEMPORAL_TYPES
    assert all(c.temporal is None for c in cases if c.section != "temporal")


def test_holdout_set_is_a_reasonable_share_of_each_section(cases):
    holdout = Counter(c.section for c in cases if c.split == "holdout")
    total = len(cases)

    assert 0.12 <= sum(holdout.values()) / total <= 0.30
    assert set(holdout) == SECTIONS  # every section has a few held-out cases


def test_cases_start_on_monday_september_21_at_15_30_unless_stated(cases):
    assert {c.now.strftime("%Y-%m-%dT%H:%M") for c in cases} == {"2026-09-21T15:30"}


async def test_oracle_scores_100_percent_on_all_cases(cases):
    results = await run_cases(cases, PARSERS["oracle"])

    summary = summarize(results, "oracle")

    problems = [(r.case.id, r.failures) for r in summary.failures]
    assert problems == []
    assert summary.overall.percent == 100.0
    assert summary.tool_calls.percent == summary.exact_calls.percent == 100.0


async def test_a_parser_that_never_acts_fails_cases_requiring_an_action(cases):
    """The opposite check: the bench notices a useless parser."""
    from daily.agent.schemas import NoActionCall

    class Silent:
        async def parse(self, request):
            return NoActionCall(tool="none")

    results = await run_cases(cases, lambda case: Silent())
    summary = summarize(results, "silent")

    assert summary.by_section["creation"].percent < 25
    assert summary.by_section["update"].percent < 25
    assert summary.by_section["temporal"].percent == 0
    assert summary.by_section["no_action"].percent == 100  # here "no action" is right
    assert summary.overall.percent < 20
