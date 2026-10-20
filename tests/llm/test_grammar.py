import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from daily.agent.prompts import _example_calls, _example_conversations, _example_days
from daily.agent.schemas import parse_tool_call, tool_call_json_schema
from daily.evals.cases import load_cases
from daily.llm.grammar import compact_grammar, integer_regex, regex_to_gbnf

from .gbnf import Grammar

NOW = datetime(2026, 9, 21, 15, 30)
CASES = Path(__file__).resolve().parents[2] / "evals" / "cases"


def compact(call: dict) -> str:
    """As the model writes it: keys in schema order, with no spaces."""
    dumped = parse_tool_call(call).model_dump(exclude_none=True)
    return json.dumps(dumped, ensure_ascii=False, separators=(",", ":"))


@pytest.fixture(scope="module")
def follow_up() -> Grammar:
    return Grammar(compact_grammar(tool_call_json_schema(follow_up=True)))


@pytest.fixture(scope="module")
def single() -> Grammar:
    return Grammar(compact_grammar(tool_call_json_schema(follow_up=False)))


# --- from regular expressions to grammar ----------------------------------------------


def test_an_anchored_pattern_becomes_a_gbnf_sequence():
    assert regex_to_gbnf("^[0-9]{2}-(0[1-9]|1[0-2])$") == ('[0-9]{2} "-" ( "0" [1-9] | "1" [0-2] )')


@pytest.mark.parametrize("pattern", ["[0-9]+", "^\\d$", "^a.b$", "^a*$", "^{2}$", "^[é]$"])
def test_what_cannot_be_translated_raises_instead_of_not_constraining(pattern):
    with pytest.raises(ValueError):
        regex_to_gbnf(pattern)


@pytest.mark.parametrize(
    ("low", "high"),
    [(1, 1440), (0, 1440), (-10080, 10080), (3, 7), (0, 0), (-5, -1), (10, 99), (95, 105)],
)
def test_integers_accept_exactly_the_range(low, high):
    pattern = re.compile(integer_regex(low, high))

    for n in range(-20000, 20001):
        assert bool(pattern.fullmatch(str(n))) == (low <= n <= high), n
    for text in ["", "007", "+5", "1.0", "01", "-0", " 5"]:
        assert not pattern.fullmatch(text) or text == "0" and low <= 0 <= high


# --- the whole grammar ---------------------------------------------------------------------


def test_every_rule_is_defined_and_on_one_line():
    text = compact_grammar(tool_call_json_schema())

    Grammar(text)  # raises if a rule is undefined or unreadable
    assert text.startswith("root ::= call-add-event | ")
    assert "space" not in text  # this is the point: no spaces or newlines between tokens


def test_accepts_every_tool_call_expected_by_eval_cases(follow_up):
    calls = [t.call for case in load_cases(CASES) for t in case.turns if t.call is not None]

    assert len(calls) > 200
    for call in calls:
        assert follow_up.accepts(compact(call)), compact(call)


def test_accepts_prompt_examples_in_key_order(single, follow_up):
    """If an example had its keys in another order, it would teach the model a shape the grammar
    then rejects."""
    day = _example_days(NOW)
    for _, call, *_ in _example_calls(day):
        text = json.dumps(call, ensure_ascii=False, separators=(",", ":"))
        assert single.accepts(text), text  # single-message examples use no answer fields
    for _, _, call in _example_conversations(day):
        text = json.dumps(call, ensure_ascii=False, separators=(",", ":"))
        assert follow_up.accepts(text), text


@pytest.mark.parametrize(
    "text",
    [
        '{\n  "tool": "none"\n}',  # indented
        '{"tool": "none"}',  # with a space
        '{"tool":"invented"}',
        '{"tool":"none","title":"x"}',  # field of another tool
        '{"tool":"add_event","date":"2026-13-01"}',  # month 13
        '{"tool":"add_event","time":"24:00"}',  # hour 24
        '{"tool":"add_event","date":"2026-09-22T10:00"}',  # a date with the time attached
        '{"tool":"add_event","time":"2026-09-22"}',  # a date where only the time is needed
        '{"tool":"add_event","duration_minutes":0}',
        '{"tool":"add_event","duration_minutes":1441}',
        '{"tool":"add_event","duration_minutes":-30}',
        '{"tool":"add_event","duration_minutes":"30"}',
        '{"tool":"add_event","date":"2026-09-22","title":"yoga"}',  # key order
        '{"tool":"update_event","shift_minutes":10081}',
        '{"tool":"get_schedule","time_band":"dawn"}',
        '{"tool":"add_event","title":"yoga"} ',  # text after the JSON
        '{"tool":"add_event","title":"yoga',
    ],
)
def test_rejects_what_the_schema_does_not_allow(follow_up, text):
    assert not follow_up.accepts(text)


@pytest.mark.parametrize(
    "text",
    [
        '{"tool":"none"}',
        '{"tool":"add_event","title":"Café \\"old\\" in Zürich"}',
        '{"tool":"add_event","title":"yoga","date":"2026-09-22","time":"19:00",'
        '"duration_minutes":45}',
        '{"tool":"update_event","event_description":"call","shift_minutes":-30}',
        '{"tool":"get_free_slots","date":"2026-09-26","min_duration_minutes":60,"time_band":"evening"}',
        '{"tool":"add_event_rule","trigger_title":"mass","typical_start":"10:00"}',
        '{"tool":"dismiss_habit"}',
    ],
)
def test_accepts_valid_shapes(follow_up, text):
    assert follow_up.accepts(text)


def test_without_a_pending_question_there_are_no_answer_fields(single, follow_up):
    answer = '{"tool":"add_event","title":"dinner","with_gap":false}'
    anchor = '{"tool":"add_event","title":"shower","anchor_time":"18:00"}'

    assert follow_up.accepts(answer) and follow_up.accepts(anchor)
    assert not single.accepts(answer) and not single.accepts(anchor)
    # the anchor time of a relative event is needed by a standalone message too
    relative = '{"tool":"add_event_relative","title":"coffee","anchor_time":"17:00"}'
    assert single.accepts(relative)
