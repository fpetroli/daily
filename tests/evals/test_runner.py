import textwrap

from daily.agent.parser import ParseFailure
from daily.agent.schemas import parse_tool_call
from daily.evals.cases import load_cases
from daily.evals.parsers import PARSERS
from daily.evals.runner import run_case, run_cases


def cases_from(tmp_path, text):
    path = tmp_path / "sample.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return load_cases(path)


class Fixed:
    """A parser with fixed answers, like a model that always makes the same mistake."""

    def __init__(self, *answers):
        self._answers = list(answers)

    async def parse(self, request):
        answer = self._answers.pop(0)
        return answer if isinstance(answer, ParseFailure) else parse_tool_call(answer)


CREATION = """
    - id: c
      say: "add gym tomorrow at 6 pm for an hour"
      call:
        {tool: add_event, title: gym, date: "{tomorrow}", time: "18:00", duration_minutes: 60}
      added: [[gym, "{tomorrow}T18:00", "{tomorrow}T19:00"]]
"""


async def run_one(tmp_path, text, factory=PARSERS["oracle"]):
    [case] = cases_from(tmp_path, text)
    return await run_case(case, factory)


async def test_oracle_passes_a_right_case(tmp_path):
    result = await run_one(tmp_path, CREATION)

    assert result.passed and result.failures == ()
    [turn] = result.turns
    assert turn.kind == "event_created" and turn.latency_ms > 0
    assert turn.tool_match and turn.exact_match


async def test_a_wrong_expected_state_is_a_failed_case(tmp_path):
    wrong = CREATION.replace('"{tomorrow}T19:00"]]', '"{tomorrow}T19:30"]]')

    result = await run_one(tmp_path, wrong)

    assert not result.passed
    assert any("schedule: missing" in f for f in result.failures)
    assert any("schedule: extra" in f for f in result.failures)


async def test_an_extra_creation_is_an_error_even_if_not_expected(tmp_path):
    text = """
    - id: c
      say: "add yoga"
      call: {tool: add_event, title: yoga}
      kind: insufficient_info
    """
    creator = Fixed({"tool": "add_event", "title": "yoga", "time": "19:00", "duration_minutes": 45})

    result = await run_one(tmp_path, text, lambda case: creator)

    assert not result.passed
    assert any(
        "reply kind 'event_created' instead of 'insufficient_info'" in f for f in result.failures
    )
    assert any("schedule: extra" in f for f in result.failures)


async def test_a_parser_that_does_nothing_fails_cases_requiring_an_action(tmp_path):
    silent = Fixed({"tool": "none"})

    result = await run_one(tmp_path, CREATION, lambda case: silent)

    assert not result.passed
    assert any("missing" in f for f in result.failures)
    assert result.turns[0].tool_match is False


async def test_an_equivalent_but_different_action_is_right_and_the_diagnostic_tells_it(tmp_path):
    with_end = Fixed(
        {
            "tool": "add_event",
            "title": "Gym",
            "date": "2026-09-22",
            "time": "18:00",
            "end_time": "19:00",
        }
    )

    result = await run_one(tmp_path, CREATION, lambda case: with_end)

    assert result.passed  # same state: case-insensitive title, explicit end
    [turn] = result.turns
    assert turn.tool_match is True and turn.exact_match is False


async def test_title_variants_are_written_with_a_bar(tmp_path):
    text = """
    - id: c
      say: "x"
      call:
        {tool: add_event, title: the accountant, time: "16:00", duration_minutes: 30}
      added: [["accountant|the accountant", "{today}T16:00", "{today}T16:30"]]
    """

    assert (await run_one(tmp_path, text)).passed


async def test_reply_kind_phrases_and_quick_replies(tmp_path):
    text = """
    - id: c
      say: "add yoga"
      call: {tool: add_event, title: yoga}
      kind: needs_anchor_confirmation
      reply_has: ["which time", "What time do you want"]
      quick_replies: ["Yes", "No"]
    """

    result = await run_one(tmp_path, text)

    assert len(result.failures) == 3
    assert "reply kind 'insufficient_info'" in result.failures[0]
    assert "does not contain 'which time'" in result.failures[1]
    assert "quick replies () instead of ('Yes', 'No')" in result.failures[2]


async def test_rules_and_usual_times_are_compared(tmp_path):
    text = """
    - id: c
      say: "x"
      call: {tool: add_event_rule, trigger_title: shower, related_title: gym, position: after}
    """
    unexpected = await run_one(tmp_path, text)
    assert any("rules: extra" in f for f in unexpected.failures)

    declared = text.rstrip() + "\n      rules_added: [[shower, gym]]\n"
    assert (await run_one(tmp_path, declared)).passed

    defaults = """
    - id: c
      say: "x"
      call:
        {tool: add_event_rule, trigger_title: yoga, typical_start: "19:00",
         typical_duration_minutes: 45}
      defaults: {yoga: ["19:00", 30]}
    """
    result = await run_one(tmp_path, defaults)
    assert any("usual times" in f for f in result.failures)


async def test_confirmed_and_rejected_habits_are_counted(tmp_path):
    text = """
    - id: c
      say: "x"
      call: {tool: add_event, title: yoga, date: "{tomorrow}", time: "19:00", duration_minutes: 45}
      added: [[yoga, "{tomorrow}T19:00", "{tomorrow}T19:45"]]
      confirmed_patterns: 2
    """

    result = await run_one(tmp_path, text)

    assert any("habits confirmed: 0 instead of 2" in f for f in result.failures)


async def test_a_failing_parser_gives_a_failed_case_with_the_reason(tmp_path):
    down = Fixed(ParseFailure("timeout"))

    result = await run_one(tmp_path, CREATION, lambda case: down)

    assert not result.passed
    assert result.turns[0].kind == "parse_failed:timeout"
    assert result.turns[0].actual_call is None


async def test_several_turns_and_cases_have_separate_databases(tmp_path):
    text = """
    - id: one
      say: "x"
      call: {tool: add_event, title: yoga, time: "19:00", duration_minutes: 45}
      added: [[yoga, "{today}T19:00", "{today}T19:45"]]
    - id: two
      say: "y"
      call: {tool: none}
    """

    results = await run_cases(cases_from(tmp_path, text), PARSERS["oracle"])

    assert [r.passed for r in results] == [True, True]  # "two" does not see the yoga of "one"


async def test_a_seed_with_an_undefined_parent_is_a_case_error(tmp_path):
    import pytest

    from daily.evals.cases import CaseError

    text = """
    - id: c
      say: "x"
      seed:
        events:
          - {title: shower, start: "{today}T19:00", end: "{today}T19:15", parent: gym}
    """

    with pytest.raises(CaseError, match="parent"):
        await run_one(tmp_path, text)


async def test_an_event_to_remove_not_in_the_seed_is_a_case_error(tmp_path):
    import pytest

    from daily.evals.cases import CaseError

    text = """
    - id: c
      say: "x"
      removed: [[ghost, "{today}T19:00", "{today}T19:15"]]
    """

    with pytest.raises(CaseError, match="not in the seed"):
        await run_one(tmp_path, text)
