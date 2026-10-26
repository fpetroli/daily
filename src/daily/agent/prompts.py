"""The parser's prompt: fixed instructions, calendar and current turn.

The order is the one the `llama-server` prompt cache rewards: first the instructions and
tools, which never change; then the calendar, which changes once a day; last the current
time, the recent conversation and the message, which change on every turn. The time stays
out of the prefix: inside it, it would invalidate the cache every minute.

The model **looks up** the calendar instead of computing dates; the code only checks that the
result is a valid date.
"""

import json
from datetime import datetime, timedelta
from typing import Literal

from daily.agent.parser import Exchange, ParseRequest

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CALENDAR_DAYS = 15

PromptStyle = Literal["classic", "turns", "tuned"]
PROMPT_STYLES: tuple[PromptStyle, ...] = ("classic", "turns", "tuned")
DEFAULT_STYLE: PromptStyle = "classic"

# The instructions come in two forms. 'classic' is the one tuned on E4B: the tool is chosen
# "from the verb", in a single prose passage. 'turns' is the one tuned on E2B: a "message shape
# → tool" list at the top, which a small model follows better. 'tuned' is 'turns' without the
# examples, for a model fine-tuned on generated examples (`daily.datagen`): it no longer needs
# them, and the prompt goes from ~7,000 tokens to ~2,000.
_INTRO = """\
You are the component that interprets the messages of a personal schedule app.
You receive a user message (and, when available, the recent conversation) and you answer \
ONLY with a JSON object describing the action to execute. Do not write any text outside the \
JSON. Write it on a single line, compact, with no spaces or newlines.

"""

_ROUTING_LIST = """\
Which action: look at the shape of the message.
- A command to add ("add", "put", "schedule", "block", "book", or just the event name with \
the day or the time: "dinner on Friday at 8 pm") → add_event.
- A command to add something BEFORE or AFTER another event already in the schedule ("add X \
after Y") → add_event_relative. It is not a rule.
- A QUESTION whether a time is free or taken ("can I put…?", "is … free?", "do I already have \
something at…?", "is there a conflict…?") → check_overlap: it only checks, it adds nothing.
- A question about what is in the schedule ("what do I have…", "show me the day", "how does my \
Thursday look") → get_schedule.
- A question about when the user is free in general ("am I free…", "find me a slot") → \
get_free_slots.
- Move, bring forward, lengthen, shorten, rename → update_event.
- Delete ("delete", "remove", "cancel", "drop", "take off") → remove_event.
- A rule that ALWAYS or USUALLY holds, with no specific day ("I always…", "it always goes \
after…", "it always lasts…", "remember that X is at 10") → add_event_rule; forgetting a rule → \
remove_event_rule.
- Greetings, thanks, questions that are not about the schedule → none.

"""

_FIELDS_BEFORE = """\
- Fill ONLY the fields the user said. Omit everything else: do not invent titles, times, \
durations, time bands or other fields. If the user does not say the time, omit "time"; if they \
say neither the end nor the duration, omit "end_time" and "duration_minutes".
- Formats: date "YYYY-MM-DD", date and time "YYYY-MM-DDTHH:MM" (24h), time only "HH:MM", \
durations and shifts in whole minutes.
- Convert "tomorrow", "the day after tomorrow", "Friday", "at 6 pm", "an hour", "half an \
hour", "in two weeks" with the calendar below: look it up, do not compute. A weekday \
("Friday") is its first occurrence in the calendar starting from tomorrow. "tonight", \
"tomorrow morning" give the day; the time is the one the user says ("tonight at 8" is 20:00).
- The current time is only for "in an hour" and "in half an hour": it is never the time of an \
event nor a time band, unless the user says so.
"""

_ROUTING_CLASSIC = """\
- Choose the tool from the verb: add, put, schedule, block, set, book → add_event; move, \
bring forward, lengthen, rename → update_event; delete, remove, cancel → remove_event; what do I \
have, show me the day → get_schedule; am I free, find me a slot → get_free_slots (slots, free \
hours in general); "can I put…?", "is … free?" (even with a precise range), "do I already have \
something at…?", "is there a conflict…?" → check_overlap (it only checks, it adds nothing); "I \
always", "it always lasts", "remember that X is at H" (a usual time, with no specific day) → \
add_event_rule.
"""

_FIELDS_AFTER = """\
- "move it by half an hour" or "postpone" → positive shift_minutes (later); only "bring \
forward" or "move earlier" → negative. If the user says the new time ("move it to 7 pm", "make \
it start at 16:30") use new_time (and new_date if the day changes too), not shift_minutes. \
"lengthen by N minutes" → positive extend_minutes, "shorten" → negative.
- "an event", "a commitment", "an appointment" are not titles: omit "title". Day and time are \
two independent fields, never combined into one value: "date" for the day (read from the \
calendar; omitted is today) and "time" for the time (read from the message, unchanged). \
"Thursday at 7 pm" is date plus time, not a single "start". An absolute date ("October 12") \
is written directly with the current year: it is not in the calendar.
- If the user does not say before or after ("near", "next to"), omit "position". For a \
recurring free slot with no name, omit "trigger_title".
- Use "none" ONLY if the message is not about the schedule, or if it is an answer with no \
pending question. If you understand which action the user wants but some data is missing, \
still choose the tool and omit the missing fields: the system asks for them.

"""

_ANSWERS = """\
Answers to a question or a proposal. If there is a recent conversation, the last assistant \
message contained a question or a proposal and the user message is the answer. Complete the \
action with ALL the values (titles, times, minutes) the assistant wrote:
- "yes", "ok", "sure" to "Shall I also add X from H1 to H2?" → add_event with the proposed \
title, time and end_time.
- "yes" to "Shall I also move X ...?" → update_event with the proposed new_time and \
new_end_time. "yes" to "Shall I also delete X ...?" → remove_event with the title and the time.
- to "Shall I keep the gap?": yes → add_event with with_gap true, no → with_gap false.
- to "what time is it, so I can place <event>?": the answer is the time of that event → \
add_event of the requested event with anchor_time. To "How long does ... last?" → add_event \
with duration_minutes.
- to "Which one do you mean?" with several events: the answer picks one of the candidates \
("the second one", its title) → repeat the action of the previous request with its \
event_description and its event_time.
- to a habit suggestion: yes → add_event_rule with the proposed \
values; no → dismiss_habit.
- to a question about missing data ("At what time?", "What is the event called?"): redo the \
same action adding the value.
- If there is no pending question, a plain "yes" or "no" → none.
"""


def _paragraphs(*parts: str) -> str:
    """Paragraphs separated by an empty line, with the final newline."""
    return "\n\n".join(part.strip("\n") for part in parts) + "\n"


INSTRUCTIONS_CLASSIC = _paragraphs(
    _INTRO,
    "Rules:\n" + _FIELDS_BEFORE + _ROUTING_CLASSIC + _FIELDS_AFTER,
    _ANSWERS,
)
INSTRUCTIONS_TURNS = _paragraphs(
    _INTRO,
    _ROUTING_LIST,
    "How to fill the fields:\n" + _FIELDS_BEFORE + _FIELDS_AFTER,
    _ANSWERS,
)
INSTRUCTIONS = INSTRUCTIONS_TURNS  # for tests looking for tools and fields: same words in both

# One line per tool, with its allowed fields. Every schema field appears here: a test checks it.
TOOL_HELP: dict[str, str] = {
    "add_event": (
        "adds an event at a precise time, or with no time if the user does not say it (the app "
        "places it with the rules). Fields: title, date (the day; omitted is today), time (the "
        "start time, if said), end_time (the end time, if said), duration_minutes (duration in "
        "minutes, if said; it wins over end_time), anchor_time (time of the reference event, if "
        "the user says it while answering a question), with_gap (true or false, only as an "
        "answer to a question about the usual gap)."
    ),
    "add_event_relative": (
        'adds an event NOW, right before or after another one already in the schedule ("add '
        'the shower after the gym"). Fields: title, anchor_description (the reference event: '
        '"after the gym" → gym), anchor_time (time of the reference event, to tell apart events '
        "with the same name), date (day of the reference event), "
        'position ("before" or "after"), offset_minutes (only if the user says how many minutes '
        'before or after: "10 minutes after the run"), duration_minutes (how long the new '
        "event lasts)."
    ),
    "update_event": (
        "moves or changes an event ALREADY in the schedule. To find it: event_description (it "
        'may be missing if the user only says the time: "the 6 pm event"), event_date (the day '
        "it is on now, only if the user says it) and event_time (the time it is at now, to tell "
        "apart events with the same name). New values: new_date (the new day, if it changes; "
        "omitted is the same day), new_time (the new start), new_end_time (the new end), "
        "new_duration_minutes, shift_minutes (shift in minutes, negative to bring forward), "
        'extend_minutes (minutes to add to the duration, negative to shorten: "lengthen by '
        'half an hour"), new_title.'
    ),
    "remove_event": (
        "deletes an event already in the schedule. Fields: event_description (it may be missing "
        'if the user only says the time: "the 6 pm event"), event_date, event_time.'
    ),
    "get_schedule": "shows the schedule of one day. Fields: date, time_band.",
    "get_free_slots": "looks for free slots. Fields: date, min_duration_minutes, time_band.",
    "check_overlap": "checks whether a time is taken. Fields: date (the day; omitted is "
    "today), time, end_time.",
    "add_event_rule": (
        'stores a rule or a habit, ONLY if the user says it always or usually holds ("the '
        'shower always goes after the gym", "the meeting lasts an hour") or accepts a '
        "suggestion. Do not use it to add an event. Fields: trigger_title (the event that "
        "follows the rule), related_title (the reference event), position, offset_minutes, "
        "gap_minutes (recurring free minutes), typical_start (HH:MM), typical_duration_minutes."
    ),
    "remove_event_rule": "deletes a rule. Fields: trigger_title, related_title, field.",
    "dismiss_habit": "the user rejects a habit suggestion. No fields.",
    "none": "no action on the schedule. No fields.",
}

TIME_BANDS = "Time bands (time_band): morning, afternoon, evening, night."


def instructions_block(style: PromptStyle = DEFAULT_STYLE) -> str:
    """The part that never changes: instructions and tools."""
    instructions = INSTRUCTIONS_CLASSIC if style == "classic" else INSTRUCTIONS_TURNS
    tools = "\n".join(f"- {name}: {help_}" for name, help_ in TOOL_HELP.items())
    return f'{instructions}\nAvailable actions ("tool" field):\n{tools}\n\n{TIME_BANDS}\n'


def calendar_block(now: datetime, days: int = CALENDAR_DAYS) -> str:
    """Today and the next days, with the labels the user uses ("tomorrow", "in a week") and the
    first date of each weekday: it changes once a day.

    The model looks up this table instead of computing: a 4B model gets dates wrong when it
    has to count, for example "Sunday" became the Sunday after.
    """
    labels = {
        0: "today",
        1: "tomorrow",
        2: "the day after tomorrow",
        7: "in a week",
        14: "in two weeks",
    }
    rows = []
    for offset in range(days):
        day = now + timedelta(days=offset)
        label = labels.get(offset, f"in {offset} days")
        rows.append(f"{day:%Y-%m-%d} {WEEKDAYS[day.weekday()]} ({label})")
    upcoming = []
    for offset in range(1, 8):
        day = now + timedelta(days=offset)
        upcoming.append(f"{WEEKDAYS[day.weekday()]} {day:%Y-%m-%d}")
    header = f"Today is {WEEKDAYS[now.weekday()]} {now:%Y-%m-%d}."
    return (
        f"{header}\nCalendar of the next {days} days:\n"
        + "\n".join(rows)
        + '\nA weekday on its own ("Saturday", "next Saturday") is the first of these: '
        + "; ".join(upcoming)
        + ".\n"
    )


def _example_days(now: datetime) -> dict[str, str]:
    """The dates the examples use, taken from today's calendar."""

    def next_weekday(weekday: int) -> str:
        for offset in range(1, 8):
            day = now + timedelta(days=offset)
            if day.weekday() == weekday:
                return f"{day:%Y-%m-%d}"
        raise AssertionError("a week has every weekday")

    return {
        "today": f"{now:%Y-%m-%d}",
        "tomorrow": f"{now + timedelta(days=1):%Y-%m-%d}",
        "day_after_tomorrow": f"{now + timedelta(days=2):%Y-%m-%d}",
        "saturday": next_weekday(5),
        "tuesday": next_weekday(1),
        "thursday": next_weekday(3),
        "wednesday": next_weekday(2),
        "monday": next_weekday(0),
    }


# The examples must NOT repeat sentences, titles or values of the eval cases: otherwise the
# measurement would reward copying, not understanding (a test checks it on the cases' text).
# Their titles must not be common names either ("meeting", "call"): E2B copied "meeting with
# Marta" or "call with the notary" when the user said only "the meeting" or "the call".
def _example_calls(day: dict[str, str]) -> list[tuple[str, dict[str, object]]]:
    tomorrow = day["tomorrow"]
    add = "add_event"
    return [
        (
            "put english class tomorrow at 5 pm for 90 minutes",
            {
                "tool": add,
                "title": "english class",
                "date": tomorrow,
                "time": "17:00",
                "duration_minutes": 90,
            },
        ),
        (
            "schedule the hairdresser on saturday at 10:30",
            {"tool": add, "title": "hairdresser", "date": day["saturday"], "time": "10:30"},
        ),
        (
            "meeting with Anna today from 11 to 12",
            {
                "tool": add,
                "title": "meeting with Anna",
                "time": "11:00",
                "end_time": "12:00",
            },
        ),
        ("add reading", {"tool": add, "title": "reading"}),
        ("add tennis on saturday", {"tool": add, "title": "tennis", "date": day["saturday"]}),
        (
            "schedule a commitment on thursday at 3 pm",
            {"tool": add, "date": day["thursday"], "time": "15:00"},
        ),
        (
            "put an appointment the day after tomorrow at 9 for half an hour",
            {
                "tool": add,
                "date": day["day_after_tomorrow"],
                "time": "09:00",
                "duration_minutes": 30,
            },
        ),
        (
            "move english class by an hour",
            {"tool": "update_event", "event_description": "english class", "shift_minutes": 60},
        ),
        (
            "bring the 1 pm lunch forward to 12:30",
            {
                "tool": "update_event",
                "event_description": "lunch",
                "event_time": "13:00",
                "new_time": "12:30",
            },
        ),
        (
            "move saturday's tennis to 11",
            {
                "tool": "update_event",
                "event_description": "tennis",
                "event_date": day["saturday"],
                "new_time": "11:00",
            },
        ),
        (
            "lengthen tennis by 20 minutes",
            {"tool": "update_event", "event_description": "tennis", "extend_minutes": 20},
        ),
        (
            "delete the guitar lesson",
            {"tool": "remove_event", "event_description": "guitar lesson"},
        ),
        ("remove the 9 pm event", {"tool": "remove_event", "event_time": "21:00"}),
        (
            "what am I doing on saturday morning?",
            {"tool": "get_schedule", "date": day["saturday"], "time_band": "morning"},
        ),
        (
            "do I have a free hour on tuesday afternoon?",
            {
                "tool": "get_free_slots",
                "date": day["tuesday"],
                "min_duration_minutes": 60,
                "time_band": "afternoon",
            },
        ),
        (
            "tomorrow at 8 do I already have something?",
            {"tool": "check_overlap", "date": tomorrow, "time": "08:00"},
        ),
        (
            "would a checkup fit today from 12 to 1 pm?",
            {"tool": "check_overlap", "time": "12:00", "end_time": "13:00"},
        ),
        (
            "add stretching after saturday's run for 10 minutes",
            {
                "tool": "add_event_relative",
                "title": "stretching",
                "anchor_description": "run",
                "date": day["saturday"],
                "position": "after",
                "duration_minutes": 10,
            },
        ),
        (
            "after swimming I always do the sauna",
            {
                "tool": "add_event_rule",
                "trigger_title": "sauna",
                "related_title": "swimming",
                "position": "after",
            },
        ),
        (
            "remember that mass is at 10",
            {"tool": "add_event_rule", "trigger_title": "mass", "typical_start": "10:00"},
        ),
        (
            "I always do the sauna near swimming",
            {"tool": "add_event_rule", "trigger_title": "sauna", "related_title": "swimming"},
        ),
        (
            "cooking class on wednesday at 7 pm for two hours",
            {
                "tool": add,
                "title": "cooking class",
                "date": day["wednesday"],
                "time": "19:00",
                "duration_minutes": 120,
            },
        ),
        (
            "add coffee after the meeting, 10 minutes",
            {
                "tool": "add_event_relative",
                "title": "coffee",
                "anchor_description": "meeting",
                "position": "after",
                "duration_minutes": 10,
            },
        ),
        (
            "put the warm-up before the match for a quarter of an hour",
            {
                "tool": "add_event_relative",
                "title": "warm-up",
                "anchor_description": "match",
                "position": "before",
                "duration_minutes": 15,
            },
        ),
        (
            "could I have a meeting today from 10 to 11?",
            {"tool": "check_overlap", "time": "10:00", "end_time": "11:00"},
        ),
        (
            "how does my thursday look?",
            {"tool": "get_schedule", "date": day["thursday"]},
        ),
        (
            "is there a 40-minute slot on thursday?",
            {"tool": "get_free_slots", "date": day["thursday"], "min_duration_minutes": 40},
        ),
        (
            "postpone the piano lesson by 45 minutes",
            {"tool": "update_event", "event_description": "piano lesson", "shift_minutes": 45},
        ),
        (
            "bring the piano lesson forward by a quarter of an hour",
            {"tool": "update_event", "event_description": "piano lesson", "shift_minutes": -15},
        ),
        (
            "the piano lesson ends at 19:30",
            {
                "tool": "update_event",
                "event_description": "piano lesson",
                "new_end_time": "19:30",
            },
        ),
        (
            "cancel the five-a-side match",
            {"tool": "remove_event", "event_description": "five-a-side match"},
        ),
        (
            "remove thursday's cooking class",
            {
                "tool": "remove_event",
                "event_description": "cooking class",
                "event_date": day["thursday"],
            },
        ),
        (
            "swimming class always lasts 50 minutes",
            {
                "tool": "add_event_rule",
                "trigger_title": "swimming class",
                "typical_duration_minutes": 50,
            },
        ),
        (
            "I no longer want the sauna to follow swimming",
            {"tool": "remove_event_rule", "trigger_title": "sauna", "related_title": "swimming"},
        ),
        (
            "move the aperitif by 20 minutes",
            {"tool": "update_event", "event_description": "aperitif", "shift_minutes": 20},
        ),
        (
            "cancel monday's lunch",
            {"tool": "remove_event", "event_description": "lunch", "event_date": day["monday"]},
        ),
        (
            "my pottery class always starts at 5 pm and ends after 50 minutes",
            {
                "tool": "add_event_rule",
                "trigger_title": "pottery class",
                "typical_start": "17:00",
                "typical_duration_minutes": 50,
            },
        ),
        (
            "put the visit to the notary in half an hour",
            {"tool": add, "title": "visit to the notary", "time": "11:05"},
            "10:35",
        ),
        (
            "in an hour I have the checkup, it lasts 40 minutes",
            {"tool": add, "title": "checkup", "time": "17:20", "duration_minutes": 40},
            "16:20",
        ),
        (
            "block the manicure in two hours",
            {"tool": add, "title": "manicure", "time": "11:10"},
            "09:10",
        ),
        ("do you know what time it is in Tokyo?", {"tool": "none"}),
        ("thanks a lot", {"tool": "none"}),
    ]


def _example_conversations(
    day: dict[str, str],
) -> list[tuple[list[tuple[str, str]], str, dict[str, object]]]:
    """(previous exchange, user message, tool call)."""
    return [
        (
            [("add meditation", 'What time do you want "meditation"?')],
            "at 7 for 20 minutes",
            {
                "tool": "add_event",
                "title": "meditation",
                "time": "07:00",
                "duration_minutes": 20,
            },
        ),
        (
            [
                (
                    "put swimming today at 5 pm for an hour",
                    "Done: swimming today from 17:00 to 18:00. "
                    'Shall I also add "sauna" from 18:00 to 18:30?',
                )
            ],
            "sure",
            {
                "tool": "add_event",
                "title": "sauna",
                "time": "18:00",
                "end_time": "18:30",
            },
        ),
        (
            [
                (
                    "delete the lesson",
                    "I found several events today: 1) lesson with Anna (09:00-10:00), "
                    "2) lesson with Paolo (17:00-18:00). Which one do you mean?",
                )
            ],
            "the second one",
            {
                "tool": "remove_event",
                "event_description": "lesson with Paolo",
                "event_time": "17:00",
            },
        ),
        (
            [
                (
                    "add lunch",
                    'After "lesson" there is usually a 15-minute gap: "lunch" would be today at '
                    "13:15 with the gap, at 13:00 without it. Shall I keep the gap?",
                )
            ],
            "better not, put it right away",
            {"tool": "add_event", "title": "lunch", "with_gap": False},
        ),
        (
            [
                (
                    "add the stretching",
                    'I can\'t find "run" in the schedule today: what time is it, so I can place '
                    '"stretching" accordingly?',
                )
            ],
            "at 7",
            {"tool": "add_event", "title": "stretching", "anchor_time": "07:00"},
        ),
        (
            [("put the coffee after the meeting", 'How long does "coffee" last?')],
            "10 minutes",
            {
                "tool": "add_event_relative",
                "title": "coffee",
                "anchor_description": "meeting",
                "position": "after",
                "duration_minutes": 10,
            },
        ),
        (
            [
                (
                    "lunch always goes near the walk",
                    'Does "lunch" go before or after "walk"?',
                )
            ],
            "before",
            {
                "tool": "add_event_rule",
                "trigger_title": "lunch",
                "related_title": "walk",
                "position": "before",
            },
        ),
        (
            [
                (
                    "add the run today at 7 for half an hour",
                    "Done: run today from 07:00 to 07:30. "
                    'Shall I also add "stretching" from 07:30 to 07:40?',
                )
            ],
            "no, never mind",
            {"tool": "none"},
        ),
        (
            [
                (
                    "remove the rehearsal",
                    "I found several events today: 1) rehearsal with the choir (10:00-11:00), "
                    "2) rehearsal with Marta (15:00-16:00). Which one do you mean?",
                )
            ],
            "the first one",
            {
                "tool": "remove_event",
                "event_description": "rehearsal with the choir",
                "event_time": "10:00",
            },
        ),
        (
            [
                (
                    "add the cooking class at 6 pm for two hours",
                    "Done: cooking class today from 18:00 to 20:00. "
                    'Shall I also add "dinner" from 20:00 to 21:00?',
                )
            ],
            "yes",
            {
                "tool": "add_event",
                "title": "dinner",
                "time": "20:00",
                "end_time": "21:00",
            },
        ),
        (
            [
                (
                    "move the 5 pm lesson to 6 pm",
                    "Done: lesson today from 18:00 to 19:00. "
                    'Shall I also move "dinner" from 19:00 to 20:00?',
                )
            ],
            "yes thanks",
            {
                "tool": "update_event",
                "event_description": "dinner",
                "new_time": "19:00",
                "new_end_time": "20:00",
            },
        ),
        (
            [
                (
                    "delete the cooking class",
                    'Done: I deleted "cooking class" (today 18:00-20:00). '
                    'Shall I also delete "dinner" (20:00-21:00)?',
                )
            ],
            "yes",
            {"tool": "remove_event", "event_description": "dinner", "event_time": "20:00"},
        ),
        (
            [
                (
                    "add the warm-up",
                    'I can\'t find "match" in the schedule today: what time is it, so I can place '
                    '"warm-up" accordingly?',
                )
            ],
            "at 3 pm",
            {"tool": "add_event", "title": "warm-up", "anchor_time": "15:00"},
        ),
        (
            [
                (
                    "put the aperitif",
                    'After "meeting" there is usually a 30-minute gap: "aperitif" would be today '
                    "at 18:30 with the gap, at 18:00 without it. Shall I keep the gap?",
                )
            ],
            "yes",
            {"tool": "add_event", "title": "aperitif", "with_gap": True},
        ),
        (
            [
                (
                    "herbal tea always pairs with dinner",
                    'Does "herbal tea" go before or after "dinner"?',
                )
            ],
            "after",
            {
                "tool": "add_event_rule",
                "trigger_title": "herbal tea",
                "related_title": "dinner",
                "position": "after",
            },
        ),
        (
            [
                (
                    "add meditation today at 7 for 20 minutes",
                    "Done: meditation today from 07:00 to 07:20. I noticed you often do "
                    '"meditation" around 07:00 (20 minutes): shall I save it as a usual time?',
                )
            ],
            "yes",
            {
                "tool": "add_event_rule",
                "trigger_title": "meditation",
                "typical_start": "07:00",
                "typical_duration_minutes": 20,
            },
        ),
        (
            [
                (
                    "put tai chi today at 8 for 30 minutes",
                    "Done: tai chi today from 08:00 to 08:30. I noticed you often do "
                    '"tai chi" around 08:00 (30 minutes): shall I save it as a usual time?',
                )
            ],
            "no",
            {"tool": "dismiss_habit"},
        ),
        # The recent conversation is about something else: the message is a request of its
        # own, and the previous action, title and minutes are not copied (real use, 2026-10).
        (
            [
                (
                    "after swimming I always do the sauna",
                    'Done: "sauna" always goes after "swimming", right after.',
                )
            ],
            "put swimming on thursday at 6 pm for an hour",
            {
                "tool": "add_event",
                "title": "swimming",
                "date": day["thursday"],
                "time": "18:00",
                "duration_minutes": 60,
            },
        ),
        (
            [
                (
                    "the pilates class is always at 8 am and lasts an hour",
                    'Done: "pilates class" usually is at 08:00 and lasts 60 minutes.',
                )
            ],
            "schedule the pilates class on saturday",
            {"tool": "add_event", "title": "pilates class", "date": day["saturday"]},
        ),
        (
            [("delete it", "Which event do you mean?")],
            "the piano lesson always lasts 45 minutes",
            {
                "tool": "add_event_rule",
                "trigger_title": "piano lesson",
                "typical_duration_minutes": 45,
            },
        ),
        (
            [("put tai chi today at 8 for 30 minutes", "Done: tai chi today from 08:00 to 08:30.")],
            "nap at 2 pm for 20 minutes",
            {"tool": "add_event", "title": "nap", "time": "14:00", "duration_minutes": 20},
        ),
        (
            [
                (
                    "the sauna always goes after swimming",
                    'Done: "sauna" always goes after "swimming", right after.',
                ),
                (
                    "stretching comes 10 minutes after every run",
                    'Done: "stretching" always goes after "run", 10 minutes after.',
                ),
            ],
            "the run usually lasts 40 minutes",
            {"tool": "add_event_rule", "trigger_title": "run", "typical_duration_minutes": 40},
        ),
        (
            [
                (
                    "stretching comes 10 minutes after every run",
                    'Done: "stretching" always goes after "run", 10 minutes after.',
                )
            ],
            "after the match add a 20-minute cool-down",
            {
                "tool": "add_event_relative",
                "title": "cool-down",
                "anchor_description": "match",
                "position": "after",
                "duration_minutes": 20,
            },
        ),
        (
            [("add the warm-up before the match", 'How long does "warm-up" last?')],
            "it takes a quarter of an hour",
            {
                "tool": "add_event_relative",
                "title": "warm-up",
                "anchor_description": "match",
                "position": "before",
                "duration_minutes": 15,
            },
        ),
    ]


# The examples E4B was tuned and measured with ('classic' style): a subset of the 'turns' ones,
# in the same order. The others were added for E2B and make E4B worse.
_CLASSIC_MESSAGES = frozenset(
    {
        "put english class tomorrow at 5 pm for 90 minutes",
        "schedule the hairdresser on saturday at 10:30",
        "meeting with Anna today from 11 to 12",
        "add reading",
        "add tennis on saturday",
        "schedule a commitment on thursday at 3 pm",
        "put an appointment the day after tomorrow at 9 for half an hour",
        "move english class by an hour",
        "bring the 1 pm lunch forward to 12:30",
        "move saturday's tennis to 11",
        "lengthen tennis by 20 minutes",
        "delete the guitar lesson",
        "remove the 9 pm event",
        "what am I doing on saturday morning?",
        "do I have a free hour on tuesday afternoon?",
        "tomorrow at 8 do I already have something?",
        "would a checkup fit today from 12 to 1 pm?",
        "add stretching after saturday's run for 10 minutes",
        "after swimming I always do the sauna",
        "remember that mass is at 10",
        "I always do the sauna near swimming",
        "thanks a lot",
    }
)
_CLASSIC_ANSWERS = frozenset(
    {"at 7 for 20 minutes", "sure", "the second one", "better not, put it right away", "at 7"}
)


def examples_block(now: datetime) -> str:
    """'classic' style: the examples in the system message, as text, with today's dates.

    They show the format (one line, compact) and, above all, what to **omit**: without
    examples the model fills every field, in particular with the current time.
    """

    def dump(call: dict[str, object]) -> str:
        return json.dumps(call, ensure_ascii=False, separators=(",", ":"))

    day = _example_days(now)
    lines = ["Examples (the dates are those of today's calendar):"]
    lines += [
        f'"{message}" → {dump(call)}'
        for message, call, *_ in _example_calls(day)
        if message in _CLASSIC_MESSAGES
    ]
    lines.append("")
    lines.append("Examples of answers to a question of the assistant:")
    for exchange, answer, call in _example_conversations(day):
        if answer not in _CLASSIC_ANSWERS:
            continue
        for user, reply in exchange:
            lines += [f"User: {user}", f"Assistant: {reply}"]
        lines += [f"User: {answer}", f"→ {dump(call)}", ""]
    return "\n".join(lines).rstrip() + "\n"


# Fixed times for the examples: every turn's header carries the current time, and the example
# must show that the output does not use it (a small model copies it into `event_time` or
# `start`).
EXAMPLE_CLOCKS = ("08:20", "12:40", "17:05", "21:15", "10:35", "14:50", "19:25", "07:10", "13:05")


def example_turns(now: datetime) -> list[dict[str, str]]:
    """The examples as a conversation: every turn has a shape **identical** to the real one
    (header with the time, last exchange, message) and the answer is the tool call.

    They change once a day with the calendar, so they stay in the cached prefix. A small model
    imitates the shape of earlier turns far more than it follows a written rule: they show the
    format (one line, compact) and what to **omit**.
    """
    day = _example_days(now)
    turns: list[dict[str, str]] = []

    def add(request: ParseRequest, call: dict[str, object]) -> None:
        answer = json.dumps(call, ensure_ascii=False, separators=(",", ":"))
        turns.append({"role": "user", "content": user_message(request, "turns")})
        turns.append({"role": "assistant", "content": answer})

    def clock() -> datetime:
        hour, minute = EXAMPLE_CLOCKS[(len(turns) // 2) % len(EXAMPLE_CLOCKS)].split(":")
        return now.replace(hour=int(hour), minute=int(minute))

    for message, call, *own_clock in _example_calls(day):
        if own_clock:  # examples with "in an hour" use the header's time: it is fixed here
            hour, minute = own_clock[0].split(":")
            when = now.replace(hour=int(hour), minute=int(minute))
        else:
            when = clock()
        add(ParseRequest(message, when), call)
    for exchange, answer, call in _example_conversations(day):
        history = tuple(Exchange(user, reply) for user, reply in exchange)
        add(ParseRequest(answer, clock(), history=history), call)
    return turns


def system_message(now: datetime, style: PromptStyle = DEFAULT_STYLE) -> str:
    """The system message. 'classic': instructions, calendar and examples, identical for the
    whole day (hence cached). 'turns': instructions and tools only, the same forever."""
    if style == "turns":
        return instructions_block("turns")
    return instructions_block("classic") + "\n" + calendar_block(now) + "\n" + examples_block(now)


def calendar_turn(now: datetime) -> list[dict[str, str]]:
    """The calendar as the last element of the prefix, right before the real turn.

    Placed far from the question (after ~3,000 tokens of examples) a small model got dates
    wrong: "the day after tomorrow" became the 25th instead of the 23rd. Close to the question
    it does not.
    """
    return [
        {"role": "user", "content": calendar_block(now)},
        {"role": "assistant", "content": "Received."},
    ]


def user_message(request: ParseRequest, style: PromptStyle = DEFAULT_STYLE) -> str:
    """The current turn: time, recent conversation (if any) and message.

    The time changes every minute: from there on the prompt cache no longer holds and every
    token is processed again (~50 ms each). The 'classic' style keeps a note on what the time
    is for: it costs about 1 s per turn, but removing or moving it makes E4B miss a few more
    cases (for the answer "15 minutes" it wrote the current time as the event's start). The
    'turns' style has the time alone: the examples have the same header and show that it is
    not used, except for "in an hour".
    """
    zone = f", {request.timezone}" if request.timezone else ""
    note = ' (only for "in an hour", "in half an hour").' if style == "classic" else ""
    lines = [f"Current time: {request.now:%H:%M}{zone}{note}"]
    if request.history:
        lines.append("Recent conversation:")
        for exchange in request.history:
            lines.append(f"User: {exchange.user}")
            lines.append(f"Assistant: {exchange.reply}")
        lines.append("")
    lines.append(f"User message: {request.message}")
    return "\n".join(lines)


def build_messages(
    request: ParseRequest, style: PromptStyle = DEFAULT_STYLE
) -> list[dict[str, str]]:
    """The messages to send to the model, with the part the cache reuses in front.

    'classic' (E4B): system (instructions, calendar, examples) and the real turn.
    'turns' (E2B): system (instructions), the examples as conversation turns, the calendar
    right before the question and the real turn. 'tuned': the same without the examples.
    """
    if style == "classic":
        return [
            {"role": "system", "content": system_message(request.now, "classic")},
            {"role": "user", "content": user_message(request, "classic")},
        ]
    examples = example_turns(request.now) if style == "turns" else []
    return [
        {"role": "system", "content": system_message(request.now, "turns")},
        *examples,
        *calendar_turn(request.now),
        {"role": "user", "content": user_message(request, "turns")},
    ]
