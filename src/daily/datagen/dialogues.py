"""Multi-turn examples: a short conversation run through the real agent.

A dialogue sets up the schedule (events, rules, usual times), then the turns: each with the
message, the call the model must write and a piece of the reply the agent must give. The
`TurnService` runs them with a scripted parser that returns those calls, so the bot's
questions in the history are the real ones ("How long does "X" last?", "Shall I also add…?",
"Which one do you mean?"), and so is the history the model receives (the last three
exchanges). A dialogue whose replies are not the expected ones is dropped: it means the
generator misread the agent, and the label could be wrong.

Half of the dialogues start after one or two unrelated requests: real use always has a
history, and the model must not copy from it.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from daily.agent.parser import ParseFailure, ParseRequest
from daily.agent.schemas import NoActionCall, ToolCall, parse_tool_call
from daily.agent.turn import TurnService
from daily.datagen import single
from daily.datagen.single import Context, _join, _put
from daily.datagen.values import (
    DURATION_WORDS,
    EVENT_DURATIONS,
    SHIFTS,
    hhmm,
    iso,
    pick_time,
    say_clock,
    say_day,
    say_duration,
    say_range,
    say_time,
    with_article,
)
from daily.services.context import Repositories
from daily.services.rules import GAP_TITLE
from daily.storage.db import open_database

Call = dict[str, Any]

YES = [
    "yes", "yes", "yes", "yes please", "sure", "ok", "okay", "yep", "yes thanks", "go ahead",
    "please do", "yeah", "of course", "sounds good", "yes, add it",
]  # fmt: skip
NO = ["no", "no", "no thanks", "not now", "nope", "no, never mind", "no need", "leave it"]


@dataclass(frozen=True, slots=True)
class Turn:
    message: str
    call: Call
    reply_has: str | None = None  # a piece the agent's reply must contain


@dataclass(slots=True)
class Dialogue:
    turns: list[Turn]
    # (title, date, start, end, parent title): the schedule before the first turn
    events: list[tuple[str, str, int, int, str | None]] = field(default_factory=list)
    # (trigger, related, position, offset, gap)
    rules: list[tuple[str, str, str, int | None, int | None]] = field(default_factory=list)
    # (title, typical start, typical duration)
    defaults: list[tuple[str, str | None, int | None]] = field(default_factory=list)
    # no unrelated requests before: "move it" right after "Done: X …" would mean X
    standalone: bool = False


# --- the ways of answering -------------------------------------------------------------


def _cap(rng: random.Random, text: str) -> str:
    """Real messages often start with a capital letter ("Yes", "Add lavoro")."""
    return text[0].upper() + text[1:] if rng.random() < 0.4 else text


def answer_time(rng: random.Random, minutes: int) -> str:
    """ "at 16", "13", "6:30 pm", "at half past 8 in the evening"."""
    text = say_time(rng, minutes)
    return text.removeprefix("at ") if rng.random() < 0.35 else text


def answer_duration(rng: random.Random, minutes: int) -> str:
    """ "1 hour", "30min", "it lasts 45 minutes", "for an hour"."""
    words = say_duration(rng, minutes)
    if words.endswith(" min") and rng.random() < 0.5:
        words = words.removesuffix(" min") + "min"
    return rng.choice(["", "", "", "for ", "it lasts ", "lasts ", "it takes "]) + words


def _duration(rng: random.Random, *, short: bool = False) -> int:
    choices = [10, 15, 15, 20, 30, 30, 45] if short else EVENT_DURATIONS
    return rng.choice([m for m in choices if m in DURATION_WORDS])


def _add(rng: random.Random, title: str, *parts: str) -> str:
    verb = rng.choice(["add", "put", "schedule", "add", "plan", "book"])
    rest = [part for part in parts if part]
    rng.shuffle(rest)
    return _join(verb, with_article(rng, title, indefinite=False), *rest)


def _for(rng: random.Random, minutes: int) -> str:
    return f"for {say_duration(rng, minutes)}"


def _today_or(day: str | None, ctx: Context) -> str:
    return day or iso(ctx.now)


def _quoted(title: str) -> str:
    return f'"{title}"'


# --- missing data: time, duration, title -------------------------------------------------


def ask_time(ctx: Context) -> Dialogue:
    """ "add X" → "What time do you want X?" → the time (and maybe the duration)."""
    rng = ctx.rng
    title = ctx.title()
    day_text, day = say_day(rng, ctx.now)
    start, minutes = pick_time(rng), _duration(rng)
    first = _put({"tool": "add_event", "title": title}, date=day)
    turns = [Turn(_add(rng, title, day_text), first, f"What time do you want {_quoted(title)}?")]
    shape = rng.random()
    if shape < 0.4:  # the time and the duration together
        answer = rng.choice([" for ", ", ", " "]).join(
            [answer_time(rng, start), say_duration(rng, minutes)]
        )
        if not answer.startswith("at "):
            answer = _join(answer_time(rng, start), _for(rng, minutes))
        turns.append(Turn(answer, {**first, "time": hhmm(start), "duration_minutes": minutes}))
    elif shape < 0.55 and start + minutes <= 23 * 60:  # a range
        turns.append(
            Turn(
                say_range(rng, start, start + minutes),
                {**first, "time": hhmm(start), "end_time": hhmm(start + minutes)},
            )
        )
    else:  # the time alone: the agent asks the duration next
        timed = {**first, "time": hhmm(start)}
        turns.append(Turn(answer_time(rng, start), timed, f"How long does {_quoted(title)} last?"))
        turns.append(Turn(answer_duration(rng, minutes), {**timed, "duration_minutes": minutes}))
    return Dialogue(turns)


def ask_duration(ctx: Context) -> Dialogue:
    """ "add X at H" → "How long does X last?" → a duration, or "until H"."""
    rng = ctx.rng
    title = ctx.title()
    day_text, day = say_day(rng, ctx.now)
    start, minutes = pick_time(rng, last=21), _duration(rng)
    first = _put({"tool": "add_event", "title": title}, date=day, time=hhmm(start))
    turns = [
        Turn(
            _add(rng, title, day_text, say_time(rng, start)),
            first,
            f"How long does {_quoted(title)} last?",
        )
    ]
    if rng.random() < 0.2 and start + minutes <= 23 * 60:
        answer = rng.choice(["until ", "till ", "it ends at "]) + say_clock(rng, start + minutes)
        turns.append(Turn(answer, {**first, "end_time": hhmm(start + minutes)}))
    else:
        turns.append(Turn(answer_duration(rng, minutes), {**first, "duration_minutes": minutes}))
    return Dialogue(turns)


def ask_title(ctx: Context) -> Dialogue:
    """ "add an event tomorrow at 10 for an hour" → "What is the event called?" → the name."""
    rng = ctx.rng
    title = ctx.title()
    noun = rng.choice(["an event", "a commitment", "an appointment", "something"])
    answer = rng.choice(["{}", "{}", "{}", "it's {}", "call it {}", "the {}"]).format(title)
    if title[0].isupper() and answer.startswith("the "):
        answer = title
    if rng.random() < 0.6:
        day_text, day = say_day(rng, ctx.now)
        start, minutes = pick_time(rng), _duration(rng)
        first = _put({"tool": "add_event"}, date=day, time=hhmm(start), duration_minutes=minutes)
        message = _join(
            rng.choice(["add", "put", "schedule"]),
            noun,
            day_text,
            say_time(rng, start),
            _for(rng, minutes),
        )
        return Dialogue(
            [
                Turn(message, first, "What is the event called?"),
                Turn(answer, {**first, "title": title}),
            ]
        )
    # "add an event" → the name → "What time…?" → day, time and duration
    day_text, day = say_day(rng, ctx.now, today=0.4)
    start, minutes = pick_time(rng), _duration(rng)
    named = {"tool": "add_event", "title": title}
    return Dialogue(
        [
            Turn(_join(rng.choice(["add", "put"]), noun), {"tool": "add_event"}, "called?"),
            Turn(answer, named, f"What time do you want {_quoted(title)}?"),
            Turn(
                _join(day_text, say_time(rng, start), _for(rng, minutes)),
                _put(dict(named), date=day, time=hhmm(start), duration_minutes=minutes),
            ),
        ]
    )


# --- proposals: "Shall I also add / move / delete …?" -------------------------------------


def related_event(ctx: Context) -> Dialogue:
    """A rule proposes the related event: yes, "yes but at H", or no."""
    rng = ctx.rng
    anchor, related = ctx.two_titles()
    position = rng.choice(["after", "after", "before"])
    offset = rng.choice([0, 0, 0, 10, 15, 30])
    related_minutes = _duration(rng, short=True)
    start, minutes = pick_time(rng, first=9, last=19), _duration(rng)
    if start + minutes + offset + related_minutes > 22 * 60:  # the proposal must stay in the day
        position = "before"
    if position == "after":
        r_start = start + minutes + offset
    else:
        r_start = start - offset - related_minutes
    r_end = r_start + related_minutes
    day_text, day = say_day(rng, ctx.now)
    first = _put(
        {"tool": "add_event", "title": anchor}, date=day, time=hhmm(start), duration_minutes=minutes
    )
    message = _add(rng, anchor, day_text, say_time(rng, start), _for(rng, minutes))
    question = f"Shall I also add {_quoted(related)} from {hhmm(r_start)} to {hhmm(r_end)}?"
    turns = [Turn(message, first, question)]
    roll = rng.random()
    proposed = _put({"tool": "add_event", "title": related}, date=day)
    if roll < 0.6:
        yes = {**proposed, "time": hhmm(r_start), "end_time": hhmm(r_end)}
        turns.append(Turn(rng.choice(YES), yes))
    elif roll < 0.75:
        other = r_start + rng.choice([-30, 30, 60, 15])
        lead = rng.choice(["yes but ", "yes, but ", "ok but ", "yes, "])
        turns.append(
            Turn(
                lead + say_time(rng, other),
                {**proposed, "time": hhmm(other), "end_time": hhmm(other + related_minutes)},
            )
        )
    else:
        turns.append(Turn(rng.choice(NO), {"tool": "none"}))
    return Dialogue(
        turns,
        rules=[(related, anchor, position, offset or None, None)],
        defaults=[(related, None, related_minutes)],
    )


def cascade(ctx: Context) -> Dialogue:
    """An event with a derived one: moving or deleting it proposes the same for the other."""
    rng = ctx.rng
    anchor, derived = ctx.two_titles()
    start, minutes = pick_time(rng, first=8, last=18), _duration(rng)
    d_minutes = _duration(rng, short=True)
    d_start = start + minutes
    today = iso(ctx.now)
    events = [
        (anchor, today, start, start + minutes, None),
        (derived, today, d_start, d_start + d_minutes, anchor),
    ]
    named = with_article(rng, anchor, indefinite=False)
    if rng.random() < 0.6:
        shift = rng.choice([s for s in SHIFTS if s <= 60])
        how = rng.choice(["move {} by {}", "postpone {} by {}", "push {} back {}"])
        message = how.format(named, say_duration(rng, shift))
        first: Call = {"tool": "update_event", "event_description": anchor, "shift_minutes": shift}
        new = (d_start + shift, d_start + d_minutes + shift)
        question = f"Shall I also move {_quoted(derived)} from {hhmm(new[0])} to {hhmm(new[1])}?"
        yes: Call = {
            "tool": "update_event",
            "event_description": derived,
            "new_time": hhmm(new[0]),
            "new_end_time": hhmm(new[1]),
        }
    else:
        verb = rng.choice(["delete", "remove", "cancel"])
        message = f"{verb} {named}"
        first = {"tool": "remove_event", "event_description": anchor}
        question = (
            f"Shall I also delete {_quoted(derived)} ({hhmm(d_start)}-{hhmm(d_start + d_minutes)})?"
        )
        yes = {"tool": "remove_event", "event_description": derived, "event_time": hhmm(d_start)}
    answer = (rng.choice(YES), yes) if rng.random() < 0.7 else (rng.choice(NO), {"tool": "none"})
    return Dialogue([Turn(message, first, question), Turn(*answer)], events=events)


# --- which one? --------------------------------------------------------------------------

_ORDINALS = ["first", "second", "third"]


def ambiguous(ctx: Context) -> Dialogue:
    """Two or three events match: the user picks one by number, time or name."""
    rng = ctx.rng
    heads: dict[str, list[str]] = {}
    for title in ctx.titles:
        if " " in title and not title[0].isupper():
            heads.setdefault(title.split()[-1], []).append(title)
    groups = [titles for titles in heads.values() if len(titles) >= 2]
    if groups and rng.random() < 0.7:
        head = rng.choice(groups)
        chosen = rng.sample(head, min(len(head), rng.choice([2, 2, 3])))
        description = chosen[0].split()[-1]
    else:  # the same event twice in the day
        title = ctx.title()
        chosen, description = [title, title], title
    starts = sorted(rng.sample(range(8 * 60, 21 * 60, 30), len(chosen)))
    while any(b - a < 60 for a, b in zip(starts, starts[1:], strict=False)):
        starts = sorted(rng.sample(range(8 * 60, 21 * 60, 30), len(chosen)))
    events = [(t, iso(ctx.now), s, s + 30, None) for t, s in zip(chosen, starts, strict=True)]
    index = rng.randrange(len(chosen))
    pick_title, pick_start = chosen[index], starts[index]

    action = rng.random()
    if action < 0.45:
        message = f"{rng.choice(['delete', 'remove', 'cancel'])} the {description}"
        base: Call = {"tool": "remove_event"}
    elif action < 0.8:
        shift = rng.choice([s for s in SHIFTS if s <= 60])
        message = f"move the {description} by {say_duration(rng, shift)}"
        base = {"tool": "update_event", "shift_minutes": shift}
    else:
        new = pick_time(rng, first=9, last=21)
        message = f"move the {description} {say_time(rng, new).replace('at ', 'to ', 1)}"
        base = {"tool": "update_event", "new_time": hhmm(new)}
    first = {**base, "event_description": description}

    forms = [
        f"the {_ORDINALS[index]} one", f"the {_ORDINALS[index]}", f"number {index + 1}",
        str(index + 1), f"the one {say_time(rng, pick_start)}",
        f"the {say_clock(rng, pick_start)} one",
    ]  # fmt: skip
    if pick_title != description:
        forms += [pick_title, pick_title, f"the {pick_title}"]
    answer = rng.choice(forms)
    picked = {**base, "event_description": pick_title, "event_time": hhmm(pick_start)}
    listed = f"{index + 1}) {pick_title} ({hhmm(pick_start)}"
    return Dialogue(
        [Turn(message, first, listed), Turn(answer, picked)],
        events=events,
    )


# --- rules that need an answer -----------------------------------------------------------


def keep_gap(ctx: Context) -> Dialogue:
    """A free gap around the anchor: "Shall I keep the gap?" → with_gap true or false."""
    rng = ctx.rng
    anchor, title = ctx.two_titles()
    position = rng.choice(["before", "after"])
    gap = rng.choice([10, 15, 20, 30, 45, 60])
    a_start = pick_time(rng, first=11, last=16)
    a_end = a_start + _duration(rng)
    events = [(anchor, iso(ctx.now), a_start, a_end, None)]
    rules = [(GAP_TITLE, anchor, position, None, gap), (title, anchor, position, 0, None)]
    defaults = [(title, None, _duration(rng, short=True))]
    yes = rng.choice(["yes", "yes", "keep it", "yes, keep the gap", "with the gap", "sure", "ok"])
    no = rng.choice(
        [
            "no",
            "no",
            "without",
            "no gap",
            "without the gap",
            "put it right away",
            "better not",
            "no, attach it",
        ]
    )
    keep = rng.random() < 0.5
    first: Call = {"tool": "add_event", "title": title}
    return Dialogue(
        [
            Turn(_add(rng, title), first, "Shall I keep the gap?"),
            Turn(yes if keep else no, {**first, "with_gap": keep}),
        ],
        events=events,
        rules=rules,
        defaults=defaults,
    )


def anchor_time(ctx: Context) -> Dialogue:
    """The rule's anchor is not in the schedule: "what time is it, so I can place X?"."""
    rng = ctx.rng
    anchor, title = ctx.two_titles()
    day_text, day = say_day(rng, ctx.now, today=0.6)
    at = pick_time(rng, first=7, last=20)
    first = _put({"tool": "add_event", "title": title}, date=day)
    return Dialogue(
        [
            Turn(_add(rng, title, day_text), first, "so I can place"),
            Turn(answer_time(rng, at), {**first, "anchor_time": hhmm(at)}),
        ],
        rules=[(title, anchor, rng.choice(["after", "before"]), None, None)],
        defaults=[(title, None, _duration(rng, short=True)), (anchor, None, _duration(rng))],
    )


def rule_duration(ctx: Context) -> Dialogue:
    """A rule places the event but its length is unknown: "How long does X usually last?"."""
    rng = ctx.rng
    anchor, title = ctx.two_titles()
    day_text, day = say_day(rng, ctx.now, today=0.5)
    if day is not None and day_text not in ("tomorrow",) and rng.random() < 0.5:
        day_text, day = "tomorrow", iso(ctx.now + timedelta(days=1))
    a_start = pick_time(rng, first=9, last=18)
    events = [(anchor, _today_or(day, ctx), a_start, a_start + _duration(rng), None)]
    minutes = _duration(rng, short=True)
    first = _put({"tool": "add_event", "title": title}, date=day)
    return Dialogue(
        [
            Turn(_add(rng, title, day_text), first, "usually last?"),
            Turn(answer_duration(rng, minutes), {**first, "duration_minutes": minutes}),
        ],
        events=events,
        rules=[(title, anchor, rng.choice(["after", "before"]), None, None)],
    )


def relative_duration(ctx: Context) -> Dialogue:
    """ "put X after Y" → "How long does X last?" → the duration."""
    rng = ctx.rng
    anchor, title = ctx.two_titles()
    position = rng.choice(["after", "before"])
    tomorrow = rng.random() < 0.3
    day = iso(ctx.now + timedelta(days=1)) if tomorrow else None
    a_start = pick_time(rng, first=9, last=19)
    events = [(anchor, _today_or(day, ctx), a_start, a_start + _duration(rng), None)]
    where = f"tomorrow's {anchor}" if tomorrow else with_article(rng, anchor, indefinite=False)
    named = with_article(rng, title, indefinite=False)
    message = _join(rng.choice(["add", "put"]), named, position, where)
    first = _put(
        {"tool": "add_event_relative", "title": title, "anchor_description": anchor},
        date=day,
        position=position,
    )
    minutes = _duration(rng, short=True)
    return Dialogue(
        [
            Turn(message, first, f"How long does {_quoted(title)} last?"),
            Turn(answer_duration(rng, minutes), {**first, "duration_minutes": minutes}),
        ],
        events=events,
    )


def rule_position(ctx: Context) -> Dialogue:
    """ "X always goes near Y" → "Does X go before or after Y?" → before / after."""
    rng = ctx.rng
    related, trigger = ctx.two_titles()
    how = rng.choice(
        [
            "{} always goes near {}",
            "{} always pairs with {}",
            "I always do {} near {}",
            "{} is always next to {}",
        ]
    )
    message = how.format(trigger, with_article(rng, related, indefinite=False))
    first: Call = {"tool": "add_event_rule", "trigger_title": trigger, "related_title": related}
    position = rng.choice(["before", "after"])
    answer = rng.choice([position, position, f"{position} it", f"{position} {related}"])
    return Dialogue(
        [
            Turn(message, first, "before or after"),
            Turn(answer, {**first, "position": position}),
        ]
    )


# --- which event, what to change -------------------------------------------------------


def missing_event(ctx: Context) -> Dialogue:
    """ "delete it" / "move it by an hour" → "Which event do you mean?" → the name."""
    rng = ctx.rng
    title = ctx.title()
    start = pick_time(rng, first=8, last=20)
    events = [(title, iso(ctx.now), start, start + 60, None)]
    if rng.random() < 0.5:
        message = rng.choice(["delete it", "remove it", "cancel it", "delete the event"])
        first: Call = {"tool": "remove_event"}
    else:
        shift = rng.choice([s for s in SHIFTS if s <= 60])
        verb = rng.choice(["move it", "postpone it", "push it back"])
        message = f"{verb} by {say_duration(rng, shift)}"
        first = {"tool": "update_event", "shift_minutes": shift}
    answer = rng.choice([title, title, f"the {title}"])
    if title[0].isupper():
        answer = title
    second = {**first, "event_description": title}
    return Dialogue(
        [Turn(message, first, "Which event do you mean?"), Turn(answer, second)],
        events=events,
        standalone=True,
    )


def missing_change(ctx: Context) -> Dialogue:
    """ "move X" → "What do you want to change…?" → the new time or a shift."""
    rng = ctx.rng
    title = ctx.title()
    start = pick_time(rng, first=8, last=19)
    events = [(title, iso(ctx.now), start, start + 60, None)]
    first: Call = {"tool": "update_event", "event_description": title}
    if rng.random() < 0.5:
        new = pick_time(rng, first=8, last=21)
        answer = say_time(rng, new).replace("at ", rng.choice(["to ", "at ", ""]), 1)
        second = {**first, "new_time": hhmm(new)}
    else:
        shift = rng.choice([s for s in SHIFTS if s <= 60])
        answer = rng.choice(["by {}", "{} later", "postpone it by {}"]).format(
            say_duration(rng, shift)
        )
        second = {**first, "shift_minutes": shift}
    verb = rng.choice(["move", "change", "reschedule"])
    message = f"{verb} {with_article(rng, title, indefinite=False)}"
    return Dialogue(
        [Turn(message, first, "What do you want to change"), Turn(answer, second)], events=events
    )


# --- habits ------------------------------------------------------------------------------


def habit(ctx: Context) -> Dialogue:
    """The same event three days in a row: the agent proposes a usual time. yes / no."""
    rng = ctx.rng
    title = ctx.title()
    start, minutes = pick_time(rng, first=7, last=20), _duration(rng)
    turns = []
    for offset in rng.sample(range(1, 7), 3):
        day = ctx.now + timedelta(days=offset)
        name = day.strftime("%A")
        day_text = rng.choice([f"on {name}", f"on {name.lower()}", name.lower()])
        turns.append(
            Turn(
                _add(rng, title, day_text, say_time(rng, start), _for(rng, minutes)),
                {
                    "tool": "add_event",
                    "title": title,
                    "date": iso(day),
                    "time": hhmm(start),
                    "duration_minutes": minutes,
                },
            )
        )
    turns[-1] = Turn(turns[-1].message, turns[-1].call, "shall I save it as a usual time?")
    if rng.random() < 0.6:
        turns.append(
            Turn(
                rng.choice(YES[:12]),
                {
                    "tool": "add_event_rule",
                    "trigger_title": title,
                    "typical_start": hhmm(start),
                    "typical_duration_minutes": minutes,
                },
            )
        )
    else:
        no = rng.choice(["no", "no", "no thanks", "nope", "don't", "no, don't save it"])
        turns.append(Turn(no, {"tool": "dismiss_habit"}))
    return Dialogue(turns)


# --- the conversation moves on ---------------------------------------------------------

_NEW_REQUESTS = ("get_schedule", "get_free_slots", "check_overlap", "add_event_rule")


def ignored_question(ctx: Context) -> Dialogue:
    """The bot asked something, the user asks something else: the new request wins."""
    dialogue = ctx.rng.choice([ask_time, ask_duration, relative_duration])(ctx)
    message, call = single.GENERATORS[ctx.rng.choice(_NEW_REQUESTS)][1](ctx)
    return Dialogue(
        [dialogue.turns[0], Turn(message, call)], dialogue.events, dialogue.rules, dialogue.defaults
    )


def plain_answer(ctx: Context) -> Dialogue:
    """ "yes", "ok", "thanks" after a reply that asks nothing: none."""
    rng = ctx.rng
    title = ctx.title()
    start, minutes = pick_time(rng), _duration(rng)
    first = {"tool": "add_event", "title": title, "time": hhmm(start), "duration_minutes": minutes}
    answer = rng.choice(["yes", "ok", "thanks", "great", "perfect", "no", "ok thanks", "cool"])
    return Dialogue(
        [
            Turn(_add(rng, title, say_time(rng, start), _for(rng, minutes)), first, "Done:"),
            Turn(answer, {"tool": "none"}),
        ]
    )


DIALOGUES: dict[str, tuple[float, Callable[[Context], Dialogue]]] = {
    "ask_time": (0.13, ask_time),
    "ask_duration": (0.09, ask_duration),
    "ask_title": (0.05, ask_title),
    "related_event": (0.12, related_event),
    "cascade": (0.08, cascade),
    "ambiguous": (0.11, ambiguous),
    "keep_gap": (0.05, keep_gap),
    "anchor_time": (0.05, anchor_time),
    "rule_duration": (0.04, rule_duration),
    "relative_duration": (0.07, relative_duration),
    "rule_position": (0.04, rule_position),
    "missing_event": (0.04, missing_event),
    "missing_change": (0.03, missing_change),
    "habit": (0.04, habit),
    "ignored_question": (0.04, ignored_question),
    "plain_answer": (0.02, plain_answer),
}


def pick(rng: random.Random) -> str:
    names = list(DIALOGUES)
    return rng.choices(names, weights=[DIALOGUES[name][0] for name in names])[0]


# --- running a dialogue through the agent -------------------------------------------------


class _Script:
    """The parser of a dialogue: returns the planned calls and keeps what the agent sent."""

    def __init__(self, calls: list[Call]) -> None:
        self._calls = calls
        self.requests: list[ParseRequest] = []

    async def parse(self, request: ParseRequest) -> ToolCall | ParseFailure:
        call = self._calls[len(self.requests)]
        self.requests.append(request)
        return parse_tool_call(call) if call["tool"] != "none" else NoActionCall(tool="none")


class UnexpectedReply(Exception):
    """The agent did not answer as the dialogue expected: the dialogue is dropped."""


def _seed(repos: Repositories, dialogue: Dialogue) -> None:
    ids: dict[str, str] = {}
    for title, day, start, end, parent in dialogue.events:
        ids[title] = repos.schedule.insert_event(
            title=title,
            start=f"{day}T{hhmm(start)}",
            end=f"{day}T{hhmm(end)}",
            derived_from_event_id=ids.get(parent) if parent else None,
        )
    for trigger, related, position, offset, gap in dialogue.rules:
        repos.event_rules.insert_rule(
            trigger_title=trigger,
            related_title=related,
            position=position,
            offset_minutes=offset,
            gap_minutes=gap,
        )
    for title, start, duration in dialogue.defaults:
        repos.event_defaults.upsert(title, typical_start=start, typical_duration_minutes=duration)


async def run(
    dialogue: Dialogue, now: datetime, prelude: list[tuple[str, Call]], rng: random.Random
) -> list[tuple[ParseRequest, Call]]:
    """Runs the prelude and the dialogue on a fresh database: (what the model received, the
    call it must write) for every turn of the dialogue. Raises `UnexpectedReply`."""
    turns = [Turn(_cap(rng, t.message), t.call, t.reply_has) for t in dialogue.turns]
    prelude = [] if dialogue.standalone else prelude
    conn = open_database()
    try:
        repos = Repositories.from_connection(conn)
        _seed(repos, dialogue)
        script = _Script([call for _, call in prelude] + [turn.call for turn in turns])
        service = TurnService(repos, script)
        for message, _ in prelude:
            result = await service.handle(_cap(rng, message), now=now)
            if result.reply.expects_reply:  # it would be a pending question
                raise UnexpectedReply(f"the prelude asks: {result.reply.text}")
        for turn in turns:
            result = await service.handle(turn.message, now=now)
            if turn.reply_has is not None and turn.reply_has not in result.reply.text:
                raise UnexpectedReply(f"{turn.message!r} → {result.reply.text!r}")
        requests = script.requests[len(prelude) :]
        return [(request, turn.call) for request, turn in zip(requests, turns, strict=True)]
    finally:
        conn.close()
