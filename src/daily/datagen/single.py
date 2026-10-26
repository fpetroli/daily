"""Single-message examples: one message, no conversation, one tool call.

Each generator returns the message and the expected call, built from the same values. The
conventions are the prompt's: fields the user does not say are omitted, today is an omitted
date, a shift is positive when later, a duration needs "for" (", 20 minutes" after an anchor
is the distance from it).
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from daily.agent.prompts import WEEKDAYS
from daily.datagen.values import (
    EVENT_DURATIONS,
    SHIFTS,
    hhmm,
    pick_time,
    say_clock,
    say_day,
    say_day_possessive,
    say_duration,
    say_range,
    say_time,
    weekday_name,
    with_article,
)

Call = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Context:
    rng: random.Random
    now: datetime
    titles: list[str]

    def title(self) -> str:
        return self.rng.choice(self.titles)

    def two_titles(self) -> tuple[str, str]:
        first, second = self.rng.sample(self.titles, 2)
        return first, second


def _join(*parts: str) -> str:
    text = " ".join(part for part in parts if part)
    return text.replace(" ,", ",").replace("  ", " ").strip()


def _put(call: Call, **fields: Any) -> Call:
    """Adds the fields that have a value: an unknown value is omitted, never `None`."""
    call.update({key: value for key, value in fields.items() if value is not None})
    return call


# --- add_event ---------------------------------------------------------------------------

_ADD_VERBS = ["add", "put", "schedule", "book", "set up", "plan", "block", "add", "add", "put in"]


def add_event(ctx: Context) -> tuple[str, Call]:
    rng = ctx.rng
    title = ctx.title()
    call: Call = {"tool": "add_event"}
    shape = rng.random()

    if shape < 0.03:  # no title: the system asks for it
        day_text, day = say_day(rng, ctx.now)
        start = pick_time(rng)
        noun = rng.choice(["an event", "a commitment", "an appointment", "something"])
        verb = rng.choice(["add", "put", "schedule"])
        message = _join(verb, noun, day_text, say_time(rng, start))
        return message, _put(call, date=day, time=hhmm(start))

    if shape < 0.08:  # "in an hour": the only use of the current time
        offset = rng.choice([30, 45, 60, 60, 90, 120])
        start = ctx.now.hour * 60 + ctx.now.minute + offset
        if start >= 24 * 60:
            offset, start = 30, ctx.now.hour * 60 + ctx.now.minute + 30
        if start < 24 * 60:
            when = {
                30: "in half an hour",
                60: "in an hour",
                90: "in an hour and a half",
                120: rng.choice(["in two hours", "in 2 hours"]),
            }.get(offset, f"in {offset} minutes")
            duration = rng.choice(EVENT_DURATIONS[:9])
            message = _join(
                rng.choice(_ADD_VERBS), with_article(rng, title), when,
                f"for {say_duration(rng, duration)}",
            )  # fmt: skip
            return message, _put(call, title=title, time=hhmm(start), duration_minutes=duration)

    if shape < 0.14:  # "tonight at 8", "tomorrow morning at 7"
        return _add_with_part_of_day(ctx, title)

    day_text, day = say_day(rng, ctx.now)
    has_time = rng.random() < 0.85
    start = pick_time(rng) if has_time else None
    time_text = say_time(rng, start) if start is not None else ""
    end_text, duration, end = "", None, None
    if start is not None:
        length = rng.random()
        if length < 0.55:
            duration = rng.choice(EVENT_DURATIONS)
            end_text = rng.choice(["for ", "for ", "for ", ", ", "lasting "]) + say_duration(
                rng, duration
            )
        elif length < 0.75 and start + 30 <= 23 * 60 + 30:
            end = min(start + rng.choice(EVENT_DURATIONS), 23 * 60 + 30)
            if rng.random() < 0.6:
                time_text = say_range(rng, start, end)
            else:
                end_text = f"until {say_clock(rng, end)}"
    elif rng.random() < 0.3:
        duration = rng.choice(EVENT_DURATIONS)
        end_text = f"for {say_duration(rng, duration)}"

    said_title = with_article(rng, title)
    verb = rng.choice(_ADD_VERBS)
    order = rng.random()
    if order < 0.45:
        message = _join(verb, said_title, day_text, time_text, end_text)
    elif order < 0.6:
        message = _join(verb, said_title, time_text, day_text, end_text)
    elif order < 0.72:
        message = _join(day_text, time_text, title, end_text)
    elif order < 0.82:
        message = _join(title, day_text, time_text, end_text)
    elif order < 0.9:
        lead = rng.choice(["can you", "could you", "please", "pls"])
        message = _join(lead, verb, said_title, day_text, time_text, end_text)
        if lead.startswith("c"):
            message += "?"
    else:
        lead = rng.choice(["remind me that I have", "I've got", "note that I have"])
        message = _join(lead, said_title, day_text, time_text, end_text)
    return message, _put(
        call, title=title, date=day,
        time=hhmm(start) if start is not None else None,
        end_time=hhmm(end) if end is not None else None, duration_minutes=duration,
    )  # fmt: skip


def _add_with_part_of_day(ctx: Context, title: str) -> tuple[str, Call]:
    rng = ctx.rng
    tomorrow = ctx.now + timedelta(days=1)
    options = [
        ("tonight", None, [8, 9, 10], 12),
        ("this evening", None, [7, 8, 9], 12),
        ("tomorrow morning", tomorrow, [7, 8, 9, 10, 11], 0),
        ("tomorrow afternoon", tomorrow, [2, 3, 4, 5], 12),
        ("tomorrow evening", tomorrow, [7, 8, 9], 12),
    ]
    day_ahead = ctx.now + timedelta(days=rng.randint(2, 7))
    name = weekday_name(rng, day_ahead.weekday())
    options += [
        (f"{name} morning", day_ahead, [8, 9, 10], 0),
        (f"{name} evening", day_ahead, [7, 8, 9], 12),
    ]
    part, day, hours, add = rng.choice(options)
    hour = rng.choice(hours)
    minute = rng.choice([0, 0, 0, 30, 15])
    start = (hour + add) * 60 + minute
    clock = f"{hour}" if minute == 0 else f"{hour}:{minute:02d}"
    duration = rng.choice(EVENT_DURATIONS) if rng.random() < 0.7 else None
    tail = f"for {say_duration(rng, duration)}" if duration else ""
    if rng.random() < 0.5:
        message = _join(rng.choice(_ADD_VERBS), with_article(rng, title), part, f"at {clock}", tail)
    else:
        message = _join(part, f"at {clock}", title, tail)
    return message, _put(
        {"tool": "add_event"}, title=title,
        date=f"{day:%Y-%m-%d}" if day is not None else None,
        time=hhmm(start), duration_minutes=duration,
    )  # fmt: skip


# --- add_event_relative ------------------------------------------------------------------


def add_event_relative(ctx: Context) -> tuple[str, Call]:
    rng = ctx.rng
    title, anchor = ctx.two_titles()
    position = rng.choice(["after", "after", "before"])
    call: Call = {"tool": "add_event_relative"}
    day, anchor_text = None, f"the {anchor}" if rng.random() < 0.7 else anchor
    if rng.random() < 0.2:
        possessive, day = say_day_possessive(rng, ctx.now)
        anchor_text = f"{possessive} {anchor}"
    offset = rng.choice([5, 10, 15, 20, 30]) if rng.random() < 0.3 else None
    duration = rng.choice(EVENT_DURATIONS[:10]) if rng.random() < 0.55 else None
    where = f"{position} {anchor_text}"
    if offset is not None:
        where = f"{say_duration(rng, offset)} {where}"
    elif rng.random() < 0.3:
        where = f"{'right' if position == 'after' else 'just'} {where}"
    tail = f"for {say_duration(rng, duration)}" if duration else ""
    verb = rng.choice(["add", "put", "schedule", "add", "fit in"])
    said = with_article(rng, title)
    if offset is not None and duration is None and rng.random() < 0.4:
        # ", 20 minutes" with no "for" after the anchor is the distance from it
        where = f"{position} {anchor_text}, {say_duration(rng, offset)}"
    shape = rng.random()
    if shape < 0.6:
        message = _join(verb, said, where, tail)
    elif shape < 0.8:
        message = _join(where, verb, said, tail)
    else:
        message = _join(verb, said, tail, where)
    return message, _put(
        call, title=title, anchor_description=anchor, date=day, position=position,
        offset_minutes=offset, duration_minutes=duration,
    )  # fmt: skip


# --- update_event ------------------------------------------------------------------------


def update_event(ctx: Context) -> tuple[str, Call]:
    rng = ctx.rng
    title = ctx.title()
    call: Call = {"tool": "update_event"}
    the = f"the {title}" if title[0].islower() else title
    target, event_date, event_time = the, None, None
    find = rng.random()
    if find < 0.2:
        possessive, event_date = say_day_possessive(rng, ctx.now)
        target = f"{possessive} {title}"
    elif find < 0.35:
        event_time = pick_time(rng)
        target = f"the {say_clock(rng, event_time)} {title}"
    elif find < 0.4:
        event_time = pick_time(rng)
        target = f"the {say_clock(rng, event_time)} {rng.choice(['event', 'appointment', 'thing'])}"
        title = None
    fields: Call = {}
    change = rng.random()
    if change < 0.25:
        new = pick_time(rng)
        message = rng.choice([
            f"move {target} to {say_clock(rng, new)}",
            f"push {target} to {say_clock(rng, new)}",
            f"reschedule {target} {say_time(rng, new)}",
            f"{target}, make it start {say_time(rng, new)}",
            f"change {target} to {say_clock(rng, new)}",
        ])  # fmt: skip
        fields["new_time"] = hhmm(new)
    elif change < 0.37:
        day_text, new_day = say_day(rng, ctx.now, today=0)
        while new_day == event_date:  # never "move tomorrow's X to tomorrow"
            day_text, new_day = say_day(rng, ctx.now, today=0)
        day_text = day_text.removeprefix("on ")
        new = pick_time(rng) if rng.random() < 0.5 else None
        at = f" {say_time(rng, new)}" if new is not None else ""
        verb = rng.choice(["move", "postpone", "push"])
        message = f"{verb} {target} to {day_text}{at}"
        fields.update(new_date=new_day, new_time=hhmm(new) if new is not None else None)
    elif change < 0.57:
        amount = rng.choice(SHIFTS)
        said = say_duration(rng, amount)
        if rng.random() < 0.6:
            message = rng.choice([
                f"move {target} by {said}", f"postpone {target} by {said}",
                f"push {target} back {said}", f"move {target} {said} later",
                f"delay {target} by {said}",
            ])  # fmt: skip
            fields["shift_minutes"] = amount
        else:
            message = rng.choice([
                f"bring {target} forward by {said}", f"move {target} {said} earlier",
                f"anticipate {target} by {said}", f"bring forward {target} {said}",
            ])  # fmt: skip
            fields["shift_minutes"] = -amount
    elif change < 0.72:
        amount = rng.choice(SHIFTS[:9])
        said = say_duration(rng, amount)
        if rng.random() < 0.6:
            message = rng.choice([
                f"extend {target} by {said}", f"lengthen {target} by {said}",
                f"make {target} {said} longer",
            ])  # fmt: skip
            fields["extend_minutes"] = amount
        else:
            message = rng.choice([f"shorten {target} by {said}", f"make {target} {said} shorter",
                                  f"cut {target} by {said}"])  # fmt: skip
            fields["extend_minutes"] = -amount
    elif change < 0.8:
        minutes = rng.choice(EVENT_DURATIONS)
        message = rng.choice([
            f"{target} lasts {say_duration(rng, minutes)}",
            f"make {target} {say_duration(rng, minutes)} long",
            f"{target} should take {say_duration(rng, minutes)} instead",
        ])  # fmt: skip
        fields["new_duration_minutes"] = minutes
    elif change < 0.87:
        end = pick_time(rng, 9, 23)
        message = rng.choice([f"{target} ends {say_time(rng, end)}",
                              f"make {target} end {say_time(rng, end)}"])  # fmt: skip
        fields["new_end_time"] = hhmm(end)
    elif change < 0.95 and title is not None:
        new_title = ctx.title()
        message = rng.choice([f"rename {target} to {new_title}",
                              f"change the name of {target} to {new_title}"])  # fmt: skip
        fields["new_title"] = new_title
    else:
        message = rng.choice([f"move {target}", f"I need to move {target}",
                              f"reschedule {target}"])  # fmt: skip
    return message, _put(
        call, event_description=title, event_date=event_date,
        event_time=hhmm(event_time) if event_time is not None else None, **fields,
    )  # fmt: skip


# --- remove_event ------------------------------------------------------------------------


def remove_event(ctx: Context) -> tuple[str, Call]:
    rng = ctx.rng
    title: str | None = ctx.title()
    the = f"the {title}" if title[0].islower() else title
    event_date, event_time = None, None
    find = rng.random()
    target = the if rng.random() < 0.8 else title
    if find < 0.2:
        possessive, event_date = say_day_possessive(rng, ctx.now)
        target = f"{possessive} {title}"
    elif find < 0.35:
        event_time = pick_time(rng)
        target = f"the {say_clock(rng, event_time)} {title}"
    elif find < 0.42:
        event_time = pick_time(rng)
        title = None
        target = rng.choice([
            f"the {say_clock(rng, event_time)} event",
            f"whatever I have {say_time(rng, event_time)}",
            f"the thing {say_time(rng, event_time)}",
        ])  # fmt: skip
    verb = rng.choice(["delete", "remove", "cancel", "drop", "take off", "get rid of", "clear"])
    shape = rng.random()
    if shape < 0.6:
        message = f"{verb} {target}"
    elif shape < 0.75:
        message = rng.choice([f"please {verb} {target}", f"can you {verb} {target}?"])
    elif shape < 0.88:
        reason = rng.choice(["I can't make it", "it got cancelled", "not happening anymore",
                             "I'm not going"])  # fmt: skip
        message = f"{verb} {target}, {reason}"
    else:
        message = rng.choice([f"{target} is cancelled", f"{target} is off"])
    return message, _put(
        {"tool": "remove_event"}, event_description=title, event_date=event_date,
        event_time=hhmm(event_time) if event_time is not None else None,
    )  # fmt: skip


# --- reads -------------------------------------------------------------------------------


def _day_and_band(ctx: Context, band: str | None) -> tuple[str, str | None]:
    """ "tomorrow afternoon", "on Friday in the morning", "this evening"."""
    rng = ctx.rng
    day_text, day = say_day(rng, ctx.now, today=0.35)
    if band is None:
        return (day_text or "today"), day
    if day is None:
        return f"this {band}", None
    bare = day_text == "tomorrow" or day_text.split()[-1].capitalize() in WEEKDAYS
    if bare and rng.random() < 0.6:
        return f"{day_text} {band}", day
    return f"{day_text} in the {band}", day


def get_schedule(ctx: Context) -> tuple[str, Call]:
    rng = ctx.rng
    band = rng.choice([None, None, None, "morning", "afternoon", "evening"])
    when, day = _day_and_band(ctx, band)
    message = rng.choice([
        f"what do I have {when}?", f"what's on {when}?", f"show me {when}",
        f"what's my schedule {when}?", f"what have I got {when}", f"agenda {when}",
        f"anything planned {when}?", f"tell me what I have {when}", f"how busy am I {when}?",
    ])  # fmt: skip
    if band is None and day is not None and rng.random() < 0.15:
        ahead = (date.fromisoformat(day) - ctx.now.date()).days
        if 1 <= ahead <= 7:
            weekday = date.fromisoformat(day).weekday()
            message = f"how does my {weekday_name(rng, weekday)} look?"
    return message, _put({"tool": "get_schedule"}, date=day, time_band=band)


def get_free_slots(ctx: Context) -> tuple[str, Call]:
    rng = ctx.rng
    band = rng.choice([None, None, "morning", "afternoon", "evening"])
    when, day = _day_and_band(ctx, band)
    minutes = rng.choice([15, 30, 30, 45, 60, 60, 90, 120]) if rng.random() < 0.45 else None
    if minutes is None:
        message = rng.choice([
            f"am I free {when}?", f"when am I free {when}?", f"any free time {when}?",
            f"do I have free time {when}?", f"find me a free slot {when}",
        ])  # fmt: skip
    else:
        said = say_duration(rng, minutes)
        message = rng.choice([
            f"find me {said} free {when}", f"find me a free slot of at least {said} {when}",
            f"do I have {said} free {when}?", f"is there a {said} gap {when}?",
            f"where can I fit {said} {when}?",
        ])  # fmt: skip
    return message, _put(
        {"tool": "get_free_slots"}, date=day, min_duration_minutes=minutes, time_band=band
    )


def check_overlap(ctx: Context) -> tuple[str, Call]:
    rng = ctx.rng
    day_text, day = say_day(rng, ctx.now, today=0.45)
    start = pick_time(rng, 7, 21)
    if rng.random() < 0.65:
        end = start + rng.choice([30, 45, 60, 60, 90, 120])
        span = say_range(rng, start, end)
        message = rng.choice([
            f"is {span} {day_text} free?", f"can I put something {day_text} {span}?",
            f"is the slot {span} {day_text} free?", f"would {span} {day_text} clash with anything?",
            f"do I have anything {span} {day_text}?",
        ])  # fmt: skip
        return _join(message), _put(
            {"tool": "check_overlap"}, date=day, time=hhmm(start), end_time=hhmm(end)
        )
    at = say_time(rng, start)
    message = rng.choice([
        f"do I already have something {day_text} {at}?",
        f"is {at.removeprefix('at ')} free {day_text}?",
        f"am I busy {day_text} {at}?", f"is there a conflict {day_text} {at}?",
    ])  # fmt: skip
    return _join(message), _put({"tool": "check_overlap"}, date=day, time=hhmm(start))


# --- rules -------------------------------------------------------------------------------


def add_event_rule(ctx: Context) -> tuple[str, Call]:
    rng = ctx.rng
    call: Call = {"tool": "add_event_rule"}
    kind = rng.random()
    if kind < 0.35:
        trigger, related = ctx.two_titles()
        position = rng.choice(["after", "before"])
        offset = rng.choice([5, 10, 15, 30]) if rng.random() < 0.25 else None
        gap = f"{say_duration(rng, offset)} " if offset is not None else ""
        message = rng.choice([
            f"{trigger} always goes {gap}{position} {related}",
            f"{position} {related} I always do {trigger}" if offset is None
            else f"I always do {trigger} {gap}{position} {related}",
            f"I always have {trigger} {gap}{position} {related}",
            f"remember that {trigger} comes {gap}{position} {related}",
        ])  # fmt: skip
        return message, _put(
            call, trigger_title=trigger, related_title=related, position=position,
            offset_minutes=offset,
        )  # fmt: skip
    if kind < 0.42:
        trigger, related = ctx.two_titles()
        message = rng.choice([f"I always do {trigger} near {related}",
                              f"{trigger} always goes next to {related}"])  # fmt: skip
        return message, _put(call, trigger_title=trigger, related_title=related)
    if kind < 0.5:
        related = ctx.title()
        position = rng.choice(["after", "before"])
        minutes = rng.choice([10, 15, 20, 30, 45])
        message = rng.choice([
            f"{position} {related} I always need {say_duration(rng, minutes)} free",
            f"I always want {say_duration(rng, minutes)} of break {position} {related}",
        ])  # fmt: skip
        return message, _put(call, related_title=related, position=position, gap_minutes=minutes)
    title = ctx.title()
    start = pick_time(rng, 7, 21) if kind < 0.85 else None
    duration = rng.choice(EVENT_DURATIONS) if kind >= 0.7 else None
    at = say_time(rng, start) if start is not None else ""
    lasts = say_duration(rng, duration) if duration is not None else ""
    if start is not None and duration is not None:
        message = rng.choice([
            f"I always have {title} {at} and it lasts {lasts}",
            f"{title} is usually {at} for {lasts}",
        ])  # fmt: skip
    elif start is not None:
        message = rng.choice([
            f"I always have {title} {at}", f"{title} is usually {at}",
            f"remember that {title} is {at}", f"my {title} is always {at}",
        ])  # fmt: skip
    else:
        message = rng.choice([
            f"{title} always lasts {lasts}", f"{title} usually takes {lasts}",
            f"remember that {title} lasts {lasts}",
        ])  # fmt: skip
    return message, _put(
        call, trigger_title=title, typical_start=hhmm(start) if start is not None else None,
        typical_duration_minutes=duration,
    )  # fmt: skip


def remove_event_rule(ctx: Context) -> tuple[str, Call]:
    rng = ctx.rng
    call: Call = {"tool": "remove_event_rule"}
    kind = rng.random()
    if kind < 0.4:
        trigger, related = ctx.two_titles()
        position = rng.choice(["after", "before"])
        message = rng.choice([
            f"forget that {trigger} goes {position} {related}",
            f"delete the rule that {trigger} comes {position} {related}",
            f"{trigger} no longer goes {position} {related}",
        ])  # fmt: skip
        return message, _put(call, trigger_title=trigger, related_title=related)
    title = ctx.title()
    if kind < 0.6:
        message = rng.choice([f"forget the usual time of {title}",
                              f"{title} no longer has a fixed time"])  # fmt: skip
        return message, _put(call, trigger_title=title, field="typical_start")
    if kind < 0.8:
        message = rng.choice([f"stop remembering how long {title} lasts",
                              f"forget the duration of {title}"])  # fmt: skip
        return message, _put(call, trigger_title=title, field="typical_duration_minutes")
    message = rng.choice([f"forget everything about {title}", f"forget the times of {title}",
                          f"delete the rules for {title}"])  # fmt: skip
    return message, _put(call, trigger_title=title)


# --- none --------------------------------------------------------------------------------

_NONE = [
    "hello", "hi there", "good morning", "hey", "thank you", "thanks a lot", "great, thanks",
    "perfect", "cool", "nice", "who made you?", "what can you do?", "how are you?",
    "tell me a joke", "what time is it in Tokyo?", "how do I use this?", "qwerty", "lol",
    "never mind", "sorry, wrong chat", "what's 2 plus 2?", "who won the match yesterday?",
    "I'm tired today", "good night", "see you", "bye", "ok cool", "sure", "yes", "no", "ok",
    "nope", "yep", "hmm", "...", "?", "test", "what is the capital of France?",
    "do you like music?", "recommend me a book", "I love pizza", "brb", "awesome",
    "what's your name?", "can you speak Italian?", "is it going to rain tomorrow?",
]  # fmt: skip


def none(ctx: Context) -> tuple[str, Call]:
    message = ctx.rng.choice(_NONE)
    if ctx.rng.random() < 0.2:
        message = message.capitalize()
    return message, {"tool": "none"}


GENERATORS: dict[str, tuple[float, Callable[[Context], tuple[str, Call]]]] = {
    "add_event": (0.28, add_event),
    "add_event_relative": (0.08, add_event_relative),
    "update_event": (0.18, update_event),
    "remove_event": (0.1, remove_event),
    "get_schedule": (0.07, get_schedule),
    "get_free_slots": (0.05, get_free_slots),
    "check_overlap": (0.05, check_overlap),
    "add_event_rule": (0.08, add_event_rule),
    "remove_event_rule": (0.04, remove_event_rule),
    "none": (0.07, none),
}


def generate(ctx: Context) -> tuple[str, Call]:
    names = list(GENERATORS)
    weights = [GENERATORS[name][0] for name in names]
    name = ctx.rng.choices(names, weights)[0]
    return GENERATORS[name][1](ctx)
