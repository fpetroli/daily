"""The parser's prompt: fixed instructions, calendar and current turn.

The order is the one the `llama-server` prompt cache rewards: first the instructions and
tools, which never change; then the calendar, which changes once a day; last the current
time, the recent conversation and the message, which change on every turn. The time stays
out of the prefix: inside it, it would invalidate the cache every minute.

The model **looks up** the calendar instead of computing dates; the code only checks that the
result is a valid date.
"""

from datetime import datetime, timedelta

from daily.agent.parser import ParseRequest

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CALENDAR_DAYS = 15

# The model is fine-tuned on generated examples (`daily.datagen`): the prompt has no examples
# of its own, only a "message shape → tool" list at the top, which a small model follows best.
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


INSTRUCTIONS = _paragraphs(
    _INTRO,
    _ROUTING_LIST,
    "How to fill the fields:\n" + _FIELDS_BEFORE + _FIELDS_AFTER,
    _ANSWERS,
)

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


def instructions_block() -> str:
    """The part that never changes: instructions and tools."""
    tools = "\n".join(f"- {name}: {help_}" for name, help_ in TOOL_HELP.items())
    return f'{INSTRUCTIONS}\nAvailable actions ("tool" field):\n{tools}\n\n{TIME_BANDS}\n'


def calendar_block(now: datetime, days: int = CALENDAR_DAYS) -> str:
    """Today and the next days, with the labels the user uses ("tomorrow", "in a week") and the
    first date of each weekday: it changes once a day.

    The model looks up this table instead of computing: a small model gets dates wrong when it
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


def calendar_turn(now: datetime) -> list[dict[str, str]]:
    """The calendar as the last element of the prefix, right before the real turn.

    Placed far from the question a small model got dates wrong: "the day after tomorrow"
    became the 25th instead of the 23rd. Close to the question it does not.
    """
    return [
        {"role": "user", "content": calendar_block(now)},
        {"role": "assistant", "content": "Received."},
    ]


def user_message(request: ParseRequest) -> str:
    """The current turn: time, recent conversation (if any) and message.

    The time changes every minute: from there on the prompt cache no longer holds and every
    token is processed again.
    """
    zone = f", {request.timezone}" if request.timezone else ""
    lines = [f"Current time: {request.now:%H:%M}{zone}"]
    if request.history:
        lines.append("Recent conversation:")
        for exchange in request.history:
            lines.append(f"User: {exchange.user}")
            lines.append(f"Assistant: {exchange.reply}")
        lines.append("")
    lines.append(f"User message: {request.message}")
    return "\n".join(lines)


def build_messages(request: ParseRequest) -> list[dict[str, str]]:
    """The messages to send to the model, with the part the cache reuses in front: system
    (instructions and tools, the same forever), the calendar (the same for the whole day) and
    the real turn.

    The model is fine-tuned on this exact shape (`daily.datagen` writes its training
    examples with this function): changing it means training the model again.
    """
    return [
        {"role": "system", "content": instructions_block()},
        *calendar_turn(request.now),
        {"role": "user", "content": user_message(request)},
    ]
