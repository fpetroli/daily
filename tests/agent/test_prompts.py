from datetime import datetime

from daily.agent.parser import Exchange, ParseRequest
from daily.agent.prompts import (
    CALENDAR_DAYS,
    INSTRUCTIONS,
    TOOL_HELP,
    build_messages,
    calendar_block,
    instructions_block,
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


def test_prefix_is_identical_for_the_whole_day():
    """It is the prefix the prompt cache reuses: the time must not get into it."""
    morning = build_messages(ParseRequest("a", datetime(2026, 9, 21, 7, 5)))[:-1]
    evening = build_messages(ParseRequest("b", datetime(2026, 9, 21, 23, 59)))[:-1]

    assert morning == evening
    assert build_messages(ParseRequest("a", datetime(2026, 9, 22, 7, 5)))[:-1] != morning


def test_time_is_in_the_turn_not_in_the_prefix():
    request = ParseRequest("add yoga at 7 pm", NOW, timezone="Europe/Rome")

    assert "15:30" not in "".join(m["content"] for m in build_messages(request)[:-1])
    assert user_message(request).startswith("Current time: 15:30, Europe/Rome")


def test_messages_are_instructions_calendar_and_real_turn():
    """The shape the model is fine-tuned on: no examples in the prompt."""
    messages = build_messages(ParseRequest("hi", NOW))

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == instructions_block()
    assert messages[1]["content"] == calendar_block(NOW)
    assert messages[2] == {"role": "assistant", "content": "Received."}
    assert "User message: hi" in messages[3]["content"]


def test_system_message_never_changes():
    """Instructions and tools only: the calendar changes with the days, and comes later."""
    later = build_messages(ParseRequest("hi", datetime(2026, 12, 1)))

    assert later[0] == build_messages(ParseRequest("hi", NOW))[0]
    assert "Calendar of the next" not in instructions_block()


def test_header_is_the_time_alone():
    """What the time is for is said by the instructions: after the time the cache no longer
    holds, and every fixed word is paid on every message."""
    text = user_message(ParseRequest("what do I have tomorrow?", NOW))

    assert text.splitlines() == ["Current time: 15:30", "User message: what do I have tomorrow?"]
    assert "The current time is only for" in instructions_block()


def test_with_history_the_turn_shows_last_exchange_before_message():
    history = (Exchange("add yoga", 'What time do you want "yoga"?'),)

    text = user_message(ParseRequest("at 7 pm", NOW, history=history))

    assert text.splitlines()[1:] == [
        "Recent conversation:",
        "User: add yoga",
        'Assistant: What time do you want "yoga"?',
        "",
        "User message: at 7 pm",
    ]


def test_with_several_exchanges_shows_them_all_in_order():
    """History always up to 3 exchanges, no longer only the last one."""
    history = (
        Exchange("add yoga", 'What time do you want "yoga"?'),
        Exchange("at 7 pm", 'How long does "yoga" usually last?'),
        Exchange("45 minutes", "Done: yoga today from 19:00 to 19:45."),
    )

    text = user_message(ParseRequest("what do I have today?", NOW, history=history))

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
