from datetime import datetime

import pytest

from daily.agent.parser import Exchange, ParseRequest
from daily.agent.prompts import (
    CALENDAR_DAYS,
    DEFAULT_STYLE,
    INSTRUCTIONS,
    PROMPT_STYLES,
    TOOL_HELP,
    build_messages,
    calendar_block,
    examples_block,
    instructions_block,
    system_message,
    user_message,
)
from daily.agent.schemas import TOOL_CALL_MODELS, TOOL_NAMES, tool_of

from .conftest import NOW


def test_calendar_starts_today_with_the_weekday():
    lines = calendar_block(NOW).splitlines()

    assert lines[0] == "Today is Monday 2026-09-21."
    assert lines[2] == "2026-09-21 Monday (today)"
    assert lines[3] == "2026-09-22 Tuesday (tomorrow)"
    assert lines[4] == "2026-09-23 Wednesday (the day after tomorrow)"
    assert lines[5] == "2026-09-24 Thursday (in 3 days)"
    assert lines[9] == "2026-09-28 Monday (in a week)"
    assert lines[16] == "2026-10-05 Monday (in two weeks)"
    assert len(lines) == 2 + CALENDAR_DAYS + 1  # header, days, line of the bare weekdays


def test_calendar_says_which_date_a_bare_weekday_is():
    """The model looks up instead of computing: "Saturday" is the first Saturday after today."""
    last = calendar_block(NOW).splitlines()[-1]

    assert "Saturday 2026-09-26" in last
    assert "Tuesday 2026-09-22" in last
    assert "Monday 2026-09-28" in last  # said on a Monday it is the one of the next week


def test_calendar_crosses_the_month_change():
    text = calendar_block(datetime(2026, 9, 29, 8, 0))

    assert "2026-09-30 Wednesday (tomorrow)" in text
    assert "2026-10-01 Thursday (the day after tomorrow)" in text
    assert "2026-10-12 Monday (in 13 days)" in text
    assert "2026-10-13 Tuesday (in two weeks)" in text


@pytest.mark.parametrize("style", PROMPT_STYLES)
def test_prefix_is_identical_for_the_whole_day(style):
    """It is the prefix the prompt cache reuses: the time must not get into it."""
    morning = build_messages(ParseRequest("a", datetime(2026, 9, 21, 7, 5)), style)[:-1]
    evening = build_messages(ParseRequest("b", datetime(2026, 9, 21, 23, 59)), style)[:-1]

    assert morning == evening
    assert build_messages(ParseRequest("a", datetime(2026, 9, 22, 7, 5)), style)[:-1] != morning


@pytest.mark.parametrize("style", PROMPT_STYLES)
def test_time_is_in_the_turn_not_in_the_prefix(style):
    request = ParseRequest("add yoga at 7 pm", NOW, timezone="Europe/Rome")

    assert "15:30" not in "".join(m["content"] for m in build_messages(request, style)[:-1])
    assert user_message(request, style).startswith("Current time: 15:30, Europe/Rome")


def test_default_style_is_the_e4b_one():
    request = ParseRequest("hi", NOW)

    assert DEFAULT_STYLE == "classic"
    assert build_messages(request) == build_messages(request, "classic")


# --- 'classic' style (E4B): examples in the system message ------------------------------------


def test_classic_is_a_system_plus_a_turn():
    messages = build_messages(ParseRequest("hi", NOW), "classic")

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == system_message(NOW, "classic")


def test_classic_instructions_calendar_and_examples_in_this_order():
    text = system_message(NOW, "classic")

    assert text.startswith(instructions_block("classic"))
    assert text.index("Available actions") < text.index("Calendar of the next")
    assert text.index("Calendar of the next") < text.index("Examples (the dates are")


def test_classic_has_the_examples_e4b_was_measured_with():
    """22 standalone messages and 5 answers; the ones added for E2B are absent (E4B gets worse)."""
    text = examples_block(NOW)

    assert sum(1 for line in text.splitlines() if line.startswith('"')) == 22
    assert text.count("\n→ ") == 5
    assert "dismiss_habit" not in text and "remove_event_rule" not in text


def test_classic_header_has_the_note_on_the_time():
    """After the time the note costs about 1 s per turn, but without it E4B misses some cases."""
    text = user_message(ParseRequest("what do I have tomorrow?", NOW), "classic")

    assert text.splitlines()[0].startswith('Current time: 15:30 (only for "in an hour"')
    assert text.splitlines()[1] == "User message: what do I have tomorrow?"
    assert "Recent conversation" not in text


def test_classic_with_history_the_turn_shows_last_exchange_before_message():
    history = (Exchange("add yoga", 'What time do you want "yoga"?'),)

    text = user_message(ParseRequest("at 7 pm", NOW, history=history), "classic")

    assert text.splitlines()[1:] == [
        "Recent conversation:",
        "User: add yoga",
        'Assistant: What time do you want "yoga"?',
        "",
        "User message: at 7 pm",
    ]


def test_classic_with_several_exchanges_shows_them_all_in_order():
    """History always up to 3 exchanges, no longer only the last one."""
    history = (
        Exchange("add yoga", 'What time do you want "yoga"?'),
        Exchange("at 7 pm", 'How long does "yoga" usually last?'),
        Exchange("45 minutes", "Done: yoga today from 19:00 to 19:45."),
    )

    text = user_message(ParseRequest("what do I have today?", NOW, history=history), "classic")

    assert text.splitlines()[1:] == [
        "Recent conversation:",
        "User: add yoga",
        'Assistant: What time do you want "yoga"?',
        "User: at 7 pm",
        'Assistant: How long does "yoga" usually last?',
        "User: 45 minutes",
        "Assistant: Done: yoga today from 19:00 to 19:45.",
        "",
        "User message: what do I have today?",
    ]


# --- 'turns' style (E2B): examples as a conversation -------------------------------------------


def test_turns_is_system_examples_calendar_and_real_turn():
    messages = build_messages(ParseRequest("hi", NOW), "turns")

    roles = [m["role"] for m in messages]
    assert roles[0] == "system" and roles[-1] == "user"
    assert roles[1:-1] == ["user", "assistant"] * ((len(messages) - 2) // 2)
    assert messages[0]["content"] == system_message(NOW, "turns")
    assert messages[-1] not in messages[1:-1]  # the real turn is not an example


def test_turns_system_message_never_changes():
    """Instructions and tools only: examples and calendar change with the days, and come later."""
    assert system_message(NOW, "turns") == instructions_block("turns")
    assert system_message(datetime(2026, 12, 1), "turns") == system_message(NOW, "turns")
    assert "Calendar of the next" not in system_message(NOW, "turns")


def test_turns_calendar_is_the_last_element_of_the_prefix():
    """After ~3,000 tokens of examples a small model got dates wrong: it stays near the question."""
    messages = build_messages(ParseRequest("hi", NOW), "turns")

    assert messages[-3]["role"] == "user" and messages[-3]["content"] == calendar_block(NOW)
    assert messages[-2] == {"role": "assistant", "content": "Received."}
    assert "User message: hi" in messages[-1]["content"]


def test_turns_header_is_the_time_alone():
    """What the time is for is said by the instructions and the examples: after the time the
    cache no longer holds, and every fixed word is paid on every message."""
    text = user_message(ParseRequest("what do I have tomorrow?", NOW), "turns")

    assert text.splitlines() == ["Current time: 15:30", "User message: what do I have tomorrow?"]
    assert "The current time is only for" in instructions_block("turns")


def test_turns_with_history_the_turn_shows_last_exchange_before_message():
    history = (Exchange("add yoga", 'What time do you want "yoga"?'),)

    text = user_message(ParseRequest("at 7 pm", NOW, history=history), "turns")

    assert text.splitlines()[1:] == [
        "Recent conversation:",
        "User: add yoga",
        'Assistant: What time do you want "yoga"?',
        "",
        "User message: at 7 pm",
    ]


def test_turns_with_several_exchanges_shows_them_all_in_order():
    """History always up to 3 exchanges, no longer only the last one."""
    history = (
        Exchange("add yoga", 'What time do you want "yoga"?'),
        Exchange("at 7 pm", 'How long does "yoga" usually last?'),
        Exchange("45 minutes", "Done: yoga today from 19:00 to 19:45."),
    )

    text = user_message(ParseRequest("what do I have today?", NOW, history=history), "turns")

    assert text.splitlines()[1:] == [
        "Recent conversation:",
        "User: add yoga",
        'Assistant: What time do you want "yoga"?',
        "User: at 7 pm",
        'Assistant: How long does "yoga" usually last?',
        "User: 45 minutes",
        "Assistant: Done: yoga today from 19:00 to 19:45.",
        "",
        "User message: what do I have today?",
    ]


def test_both_instruction_forms_say_the_same_about_tools_and_fields():
    for style in PROMPT_STYLES:
        text = instructions_block(style)
        for name in TOOL_NAMES:
            assert name in text, (style, name)


def test_every_tool_is_described_and_only_schema_tools():
    assert set(TOOL_HELP) == set(TOOL_NAMES)
    for name in TOOL_NAMES:
        assert f"- {name}: " in instructions_block()


def test_every_schema_field_is_explained_in_the_prompt():
    for model in TOOL_CALL_MODELS:
        help_text = TOOL_HELP[tool_of(model)]
        for field in model.model_fields:
            if field != "tool":
                assert field in help_text, f"{tool_of(model)}.{field} is missing from the prompt"


def test_tools_mentioned_in_the_rules_exist():
    for name in ("add_event", "update_event", "remove_event", "add_event_rule", "dismiss_habit"):
        assert name in INSTRUCTIONS and name in TOOL_NAMES


def test_instructions_say_how_to_answer_every_system_question():
    for key in (
        "Shall I also add",
        "Shall I also move",
        "Shall I also delete",
        "Shall I keep the gap",
        "what time is it",
        "How long does",
        "Which one do you mean?",
        "dismiss_habit",
        "anchor_time",
        "with_gap",
    ):
        assert key in INSTRUCTIONS, key


def test_every_prompt_example_is_a_valid_tool_call():
    from daily.agent.prompts import _example_calls, _example_conversations, _example_days
    from daily.agent.schemas import parse_tool_call

    day = _example_days(NOW)
    calls = [call for _, call, *_ in _example_calls(day)]
    calls += [call for _, _, call in _example_conversations(day)]

    assert len(calls) >= 18
    assert {parse_tool_call(call).tool for call in calls} == set(TOOL_NAMES)


def test_examples_use_the_dates_of_todays_calendar():
    from daily.agent.prompts import example_turns

    text = "\n".join(turn["content"] for turn in example_turns(NOW))

    assert '"date":"2026-09-22"' in text  # tomorrow
    assert '"date":"2026-09-26","time":"10:30"' in text  # the first Saturday after today
    assert example_turns(datetime(2026, 9, 22, 8, 0)) != example_turns(NOW)  # from the calendar


def test_examples_have_the_real_turn_shape_and_do_not_use_current_time():
    """A small model copies the header's time into `start` or `event_time`: the examples show a
    different time on every turn and an answer that does not contain it."""
    from daily.agent.prompts import EXAMPLE_CLOCKS, example_turns

    turns = example_turns(NOW)
    users, answers = turns[0::2], turns[1::2]

    assert all(u["content"].startswith("Current time: ") for u in users)
    assert len({u["content"].splitlines()[0] for u in users}) > 1
    assert len(EXAMPLE_CLOCKS) > 1 and "15:30" not in EXAMPLE_CLOCKS
    for user, answer in zip(users, answers, strict=True):
        clock = user["content"].split()[2]  # "Current time: HH:MM"
        assert clock not in answer["content"], (clock, answer["content"])


def _example_messages() -> list[str]:
    """The sentences the examples put in the user's mouth: standalone messages, requests of the
    previous exchange and answers. One- or two-word answers ("yes", "yes thanks", "the first")
    are common to every conversation and cannot be avoided: the exchanges are what counts."""
    from daily.agent.prompts import _example_calls, _example_conversations, _example_days

    day = _example_days(NOW)
    messages = [m for m, *_ in _example_calls(day)]
    for exchange, answer, _ in _example_conversations(day):
        messages += [user for user, _ in exchange]
        if len(answer.split()) > 2:
            messages.append(answer)
    return messages


def test_examples_do_not_repeat_eval_case_messages():
    """Copying the cases' sentences would skew the measurement: the examples are other sentences."""
    from pathlib import Path

    from daily.evals.cases import load_cases

    cases = load_cases(Path(__file__).resolve().parents[2] / "evals" / "cases")
    said = {t.say.strip().casefold() for c in cases for t in c.turns}

    assert {m.strip().casefold() for m in _example_messages()} & said == set()


def test_examples_are_not_near_copies_of_case_messages():
    """A sentence with one word changed rewards copying more than an identical one. The limit
    catches near-copies, not sentences with the same structure ("add X after Y"): those
    stay, and that is why the `dev` and `holdout` cases do not prove generalization to new
    sentences."""
    import re
    from difflib import SequenceMatcher
    from pathlib import Path

    from daily.evals.cases import load_cases

    def words(text: str) -> list[str]:
        return re.findall(r"\w+", text.casefold())

    cases = load_cases(Path(__file__).resolve().parents[2] / "evals" / "cases")
    said = [(c.id, t.say) for c in cases for t in c.turns]

    for example in _example_messages():
        for case_id, message in said:
            ratio = SequenceMatcher(None, words(example), words(message)).ratio()
            assert ratio < 0.7, f"'{example}' is almost the same as '{message}' ({case_id})"
