import asyncio
import json
import logging
import random
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from daily.agent.prompts import WEEKDAYS
from daily.agent.schemas import parse_tool_call
from daily.datagen.__main__ import eval_titles, main, normalize
from daily.datagen.dialogues import DIALOGUES, missing_event, run
from daily.datagen.single import GENERATORS, Context
from daily.datagen.values import (
    MONTHS,
    TITLES,
    hhmm,
    say_absolute_date,
    say_day,
    say_range,
    say_time,
    titles,
)
from daily.evals.cases import load_cases

NOW = datetime(2026, 9, 21, 15, 30)  # a Monday
CASES = Path(__file__).parents[2] / "evals" / "cases"


def read_time(text: str) -> int:
    """An independent reading of the time phrases, in minutes: the generator must agree."""
    text = text.removeprefix("at ").strip()
    if text in ("noon", "midday"):
        return 12 * 60
    if m := re.fullmatch(r"half past (\d+)( in the evening)?", text):
        return (int(m[1]) + (12 if m[2] else 0)) * 60 + 30
    if m := re.fullmatch(r"a quarter past (\d+)", text):
        return int(m[1]) * 60 + 15
    if m := re.fullmatch(r"a quarter to (\d+) in the evening", text):
        return (int(m[1]) + 12) * 60 - 15
    if m := re.fullmatch(r"(\d+) in the (morning|afternoon|evening)", text):
        return (int(m[1]) + (0 if m[2] == "morning" else 12)) * 60
    m = re.fullmatch(r"(\d+)(?::(\d\d))? ?(am|pm)?", text)
    assert m, text
    hour, minute = int(m[1]), int(m[2] or 0)
    if m[3] == "pm" and hour != 12:
        hour += 12
    if m[3] is None and hour < 13:
        assert 7 <= hour <= 12 or m[2], f"ambiguous bare hour: {text}"
    return hour * 60 + minute


@pytest.mark.parametrize("minutes", range(7 * 60, 23 * 60, 5))
def test_every_way_of_saying_a_time_means_that_time(minutes):
    rng = random.Random(minutes)
    for _ in range(40):
        assert read_time(say_time(rng, minutes)) == minutes


def read_range(text: str) -> tuple[int, int] | None:
    """Tries every "to"/"and" as the split point: "a quarter to 9" has a "to" of its own."""
    lead, _, rest = text.partition(" ")
    assert lead in ("from", "between"), text
    joint = " to " if lead == "from" else " and "
    for m in re.finditer(re.escape(joint), rest):
        first, second = rest[: m.start()], rest[m.end() :]
        # "from 4 to 5 pm" is 16:00, "from 11:55 to 12:55 pm" is 11:55: the pm of the end
        # goes to the start only if it keeps the start before the end
        bare = re.match(r"(\d+)", first)
        try:
            end = read_time(second)
            if second.endswith(" pm") and bare and not first.endswith(("am", "pm")):
                if int(bare[1]) + 12 <= end // 60:
                    first += " pm"
            return read_time(first), end
        except AssertionError:
            continue
    return None


def test_a_range_says_its_two_ends():
    rng = random.Random(1)
    for _ in range(500):
        start = rng.randrange(7 * 60, 21 * 60, 5)
        end = start + rng.choice([30, 60, 90])
        text = say_range(rng, start, end)
        assert read_range(text) == (start, end), text


def test_a_weekday_is_its_first_occurrence_from_tomorrow():
    rng = random.Random(2)
    seen = set()
    for _ in range(400):
        text, value = say_day(rng, NOW)
        name = text.split()[-1].capitalize() if text else ""
        if name in WEEKDAYS:
            day = date.fromisoformat(value)
            assert WEEKDAYS[day.weekday()] == name
            assert 1 <= (day - NOW.date()).days <= 7
            seen.add(name)
    assert len(seen) == 7


def test_today_is_an_omitted_date():
    rng = random.Random(3)
    for _ in range(200):
        text, value = say_day(rng, NOW, today=1)
        assert text in ("", "today") and value is None


def test_an_absolute_date_has_the_current_year_and_its_month_and_day():
    rng = random.Random(4)
    for _ in range(200):
        text, value = say_absolute_date(rng, NOW)
        day = date.fromisoformat(value)
        assert day.year == NOW.year and day > NOW.date() + timedelta(days=1)
        assert MONTHS[day.month - 1] in text and str(day.day) in text


def test_relative_days_match_the_calendar():
    rng = random.Random(5)
    words = {"tomorrow": 1, "the day after tomorrow": 2, "in a week": 7, "in two weeks": 14}
    for _ in range(300):
        text, value = say_day(rng, NOW, today=0)
        if text in words:
            assert value == f"{NOW + timedelta(days=words[text]):%Y-%m-%d}"


def test_the_title_pool_shares_nothing_with_the_eval_cases():
    used = {title.lower() for title in eval_titles(load_cases(CASES))}
    pool = titles(eval_titles(load_cases(CASES)))

    assert len(pool) >= 80
    assert not {title.lower() for title in pool} & used
    assert set(pool) <= set(TITLES)


@pytest.mark.parametrize("tool", GENERATORS)
def test_every_generator_writes_a_valid_call_of_its_tool(tool):
    rng = random.Random(tool)
    pool = titles(set())
    for _ in range(300):
        now = NOW + timedelta(days=rng.randrange(400), minutes=rng.randrange(-500, 400, 5))
        message, call = GENERATORS[tool][1](Context(rng, now, pool))
        assert call["tool"] == tool
        assert message.strip() == message and "  " not in message
        assert None not in call.values()


def test_the_command_writes_tuned_chats_and_leaves_out_the_eval_messages(tmp_path, capsys):
    out = tmp_path / "train.jsonl"

    args = ["--count", "300", "--dialogues", "0", "--seed", "7", "--cases", str(CASES)]
    assert main([*args, "--out", str(out)]) == 0

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 300
    held_out = {normalize(turn.say) for case in load_cases(CASES) for turn in case.turns}
    for row in rows:
        *prompt, answer = row["messages"]
        assert answer["role"] == "assistant"
        assert json.loads(answer["content"])["tool"] == row["tool"]
        assert " " not in answer["content"].split('"')[0]  # compact, as the model must write it
        message = prompt[-1]["content"].split("User message: ", 1)[1]
        assert normalize(message) not in held_out
    assert "300 examples" in capsys.readouterr().err


def test_the_same_seed_gives_the_same_file(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    args = ["--count", "50", "--dialogues", "20", "--seed", "3", "--cases", str(CASES)]
    main([*args, "--out", str(first)])
    main([*args, "--out", str(second)])

    assert first.read_text() == second.read_text()


@pytest.fixture
def quiet_agent():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


@pytest.mark.parametrize("name", DIALOGUES)
def test_every_dialogue_gets_the_replies_it_expects_from_the_agent(name, quiet_agent):
    """`run` raises if a reply misses the expected question: the labels rely on it."""
    rng = random.Random(name)
    pool = titles(set())

    async def go():
        for _ in range(40):
            now = NOW + timedelta(days=rng.randrange(400), minutes=rng.randrange(-500, 400, 5))
            dialogue = DIALOGUES[name][1](Context(rng, now, pool))
            turns = await run(dialogue, now, [], rng)
            assert len(turns) == len(dialogue.turns) >= 2
            for request, call in turns:
                assert parse_tool_call(call).tool == call["tool"]
                assert request.now == now

    asyncio.run(go())


def test_the_model_sees_the_real_question_of_the_agent(quiet_agent):
    rng = random.Random(11)
    dialogue = DIALOGUES["ask_duration"][1](Context(rng, NOW, ["haircut"]))

    turns = asyncio.run(run(dialogue, NOW, [], rng))

    (first, _), (answer, call) = turns
    assert first.history == ()
    assert answer.history[-1].reply == 'How long does "haircut" last?'
    assert answer.history[-1].user == turns[0][0].message
    assert call["title"] == "haircut"
    assert "duration_minutes" in call or "end_time" in call


def test_unrelated_requests_come_first_and_stay_in_the_history(quiet_agent):
    rng = random.Random(12)
    prelude = [("what do I have tomorrow?", {"tool": "get_schedule", "date": "2026-09-22"})]
    dialogue = DIALOGUES["rule_position"][1](Context(rng, NOW, titles(set())))

    (first, _), _ = asyncio.run(run(dialogue, NOW, prelude, rng))

    assert first.history[0].user.lower() == "what do i have tomorrow?"


def test_move_it_never_follows_an_unrelated_request(quiet_agent):
    """After "Done: volleyball …", "move it by an hour" means the volleyball."""
    rng = random.Random(13)
    prelude = [
        (
            "add a swim at 9 for an hour",
            {"tool": "add_event", "title": "swim", "time": "09:00", "duration_minutes": 60},
        )
    ]
    dialogue = missing_event(Context(rng, NOW, ["haircut"]))

    (first, _), _ = asyncio.run(run(dialogue, NOW, prelude, rng))

    assert first.history == ()


def test_the_command_runs_the_conversations_after_the_single_messages(tmp_path, quiet_agent):
    out = tmp_path / "train.jsonl"

    args = ["--count", "10", "--dialogues", "30", "--seed", "5", "--cases", str(CASES)]
    assert main([*args, "--out", str(out)]) == 0

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["source"] for row in rows[:10]] == ["single"] * 10
    dialogue_rows = rows[10:]
    assert len(dialogue_rows) >= 60 and {row["source"] for row in dialogue_rows} <= set(DIALOGUES)
    with_history = [
        row for row in dialogue_rows if "Recent conversation:" in row["messages"][-2]["content"]
    ]
    assert len(with_history) > len(dialogue_rows) / 2


def test_hhmm():
    assert hhmm(7 * 60 + 5) == "07:05"
