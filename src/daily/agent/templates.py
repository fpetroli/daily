"""The reply messages: fixed, never written by the model.

Every outcome has a template. Messages are **self-contained**: they quote title, day and time
of everything the user can confirm, because the model interprets the answer ("yes", "the
second one", "at 6 pm") by reading the recent history (up to 3 exchanges; still, a
well-written question must not rely on more than one step). Proposals have the Yes/No quick
replies, and `expects_reply` tells the workflow whether the last reply is waiting for an
answer from the user (used for "Ok."/"no action" after a "no").

It never raises: an outcome it does not know becomes a generic message.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from daily.agent.parser import ParseFailure
from daily.agent.prompts import WEEKDAYS
from daily.agent.results import (
    Executed,
    ExecutionError,
    FreeSlotsView,
    HabitDismissed,
    MissingField,
    NoAction,
    ParseFailed,
    ScheduleView,
)
from daily.agent.schemas import (
    AddEventCall,
    AddEventRelativeCall,
    AddEventRuleCall,
    CheckOverlapCall,
    RemoveEventRuleCall,
)
from daily.domain.models import EventDefault, EventRule
from daily.domain.timeutil import add_minutes_to_iso, combine_date_and_time, minutes_between_iso
from daily.services.habits import (
    EventPatternProposal,
    GapPatternProposal,
    HabitProposal,
    SequencePatternProposal,
)
from daily.services.relative import AnchorAmbiguous, AnchorNotFound, RelativeNeedsInfo
from daily.services.rules import (
    AddEventRuleInvalidFormat,
    AddEventRuleNeedsInfo,
    AddEventRuleNeedsPosition,
    AddEventRuleSuccess,
    RemoveEventRuleAmbiguous,
    RemoveEventRuleNotFound,
    RemoveEventRuleSuccess,
)
from daily.services.schedule import (
    EventAmbiguous,
    EventCreated,
    EventNotFound,
    EventRemoved,
    EventUpdated,
    InsufficientInfo,
    InvalidRange,
    NeedsAnchorConfirmation,
    NeedsAnchorDuration,
    NeedsDurationConfirmation,
    NeedsGapConfirmation,
)
from daily.services.types import CheckOverlapResult, EventInfo, ScheduleEntry
from daily.services.undo import Undone

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
YES_NO = ("Yes", "No")
BEFORE_AFTER = ("Before", "After")

GENERIC = "Done."
NOT_UNDERSTOOD = (
    "I didn't understand what to do. I can add, move or delete events, tell you what is on your "
    "schedule and find free slots."
)
ACKNOWLEDGED = "Ok."
REPHRASE = "I didn't understand, can you rephrase?"
MODEL_DOWN = "The model is not answering: try again shortly."
WENT_WRONG = "Something went wrong: try again."
NOTHING_TO_UNDO = "There is nothing to undo."
BAND_PHRASES = {
    "morning": "in the morning",
    "afternoon": "in the afternoon",
    "evening": "in the evening",
    "night": "at night",
}


@dataclass(frozen=True, slots=True)
class Styled:
    """A stretch of a reply's text in bold or struck through, `start` and `end` as string
    indices. The text reads the same without it: each channel renders it its own way (Telegram
    as a message entity) or ignores it, and the model's history keeps only the text."""

    style: Literal["bold", "strikethrough"]
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Reply:
    """The reply of a turn. `quick_replies` are the texts sent as the user's answer when
    tapped."""

    text: str
    quick_replies: tuple[str, ...] = ()
    expects_reply: bool = False
    styles: tuple[Styled, ...] = ()


def render(execution: object, now: datetime) -> Reply:
    """The message for an outcome. Never an exception: when in doubt, a generic message."""
    try:
        return _render(execution, now.date())
    except Exception:  # a broken template must not stop the turn
        return Reply(GENERIC)


# --- formatting ------------------------------------------------------------------------


def _q(title: str) -> str:
    return f'"{title}"'


def _name(title: str | None, fallback: str = "the event") -> str:
    """The quoted title, or a plain fallback when the title is unknown."""
    return _q(title) if title else fallback


def _date_of(iso: str) -> date:
    return date.fromisoformat(iso.split("T")[0])


def _time_of(iso: str) -> str:
    return iso.split("T")[1]


def _end_of(start: str, end: str) -> str:
    """The end time; midnight of the next day is written 24:00."""
    if _date_of(end) > _date_of(start) and _time_of(end) == "00:00":
        return "24:00"
    return _time_of(end)


def _span(start: str, end: str) -> str:
    return f"from {_time_of(start)} to {_end_of(start, end)}"


def _range(start: str, end: str) -> str:
    return f"{_time_of(start)}-{_end_of(start, end)}"


def day_label(day: date | str, today: date) -> str:
    """The label: "today", "tomorrow", "the day after tomorrow", "yesterday" or
    "Monday, September 28" (with the year if it is not the current one)."""
    target = date.fromisoformat(day) if isinstance(day, str) else day
    relative = {0: "today", 1: "tomorrow", 2: "the day after tomorrow", -1: "yesterday"}.get(
        (target - today).days
    )
    return relative or _calendar_date(target, today)


def _calendar_date(target: date, today: date) -> str:
    label = f"{WEEKDAYS[target.weekday()]}, {MONTHS[target.month - 1]} {target.day}"
    return label if target.year == today.year else f"{label}, {target.year}"


def _on(day: date | str, today: date) -> str:
    """The day as a complement: "today", "on Monday, September 28"."""
    label = day_label(day, today)
    return label if label[:1].islower() else f"on {label}"


def _capfirst(text: str) -> str:
    return text[:1].upper() + text[1:]


def _titles(events: tuple[EventInfo, ...]) -> str:
    return ", ".join(f"{e.title} ({_range(e.start, e.end)})" for e in events)


def _numbered(events: tuple[EventInfo, ...]) -> str:
    return ", ".join(
        f"{i}) {e.title} ({_range(e.start, e.end)})" for i, e in enumerate(events, start=1)
    )


def _candidate_replies(events: tuple[EventInfo, ...]) -> tuple[str, ...]:
    """Each candidate is a quick reply that identifies it on its own (title and time)."""
    return tuple(f"{e.title} at {_time_of(e.start)}" for e in events)


# --- dispatcher ------------------------------------------------------------------------


def _render(execution: object, today: date) -> Reply:
    match execution:
        case Executed():
            return _with_habit(_render_outcome(execution, today), execution.habit)
        case MissingField():
            return _missing_field(execution)
        case NoAction(after_question=True):
            return Reply(ACKNOWLEDGED)
        case NoAction():
            return Reply(NOT_UNDERSTOOD)
        case ParseFailed(failure=ParseFailure(kind="invalid_output")):
            return Reply(REPHRASE)
        case ParseFailed():
            return Reply(MODEL_DOWN)
        case ExecutionError():
            return Reply(WENT_WRONG)
    return Reply(GENERIC)


def _with_habit(reply: Reply, habit: HabitProposal | None) -> Reply:
    """Appends a habit proposal: it is the only question of the reply."""
    if habit is None:
        return reply
    proposal = _habit(habit)
    return Reply(f"{reply.text} {proposal.text}", proposal.quick_replies, expects_reply=True)


def _render_outcome(executed: Executed, today: date) -> Reply:
    call, outcome = executed.call, executed.outcome
    match outcome:
        case EventCreated():
            return _created(outcome, today)
        case InvalidRange():
            return Reply(
                "That time is not valid: the end must come after the start.", expects_reply=True
            )
        case InsufficientInfo(missing="start") if isinstance(call, AddEventCall):
            return Reply(f"What time do you want {_name(call.title)}?", expects_reply=True)
        case InsufficientInfo() if isinstance(call, AddEventCall):
            return Reply(f"How long does {_name(call.title)} last?", expects_reply=True)
        case InsufficientInfo():
            return Reply("I need the time (or the duration) to complete it.", expects_reply=True)
        case NeedsAnchorConfirmation(missing=missing):
            when = _on(_call_date(call, today), today)
            hint = (
                f" (it is usually at {missing.suggested_related_start})"
                if missing.suggested_related_start
                else ""
            )
            return Reply(
                f"I can't find {_q(missing.related_title)} in the schedule {when}: what time is "
                f"it, so I can place {_q(missing.trigger_title)} accordingly?{hint}",
                expects_reply=True,
            )
        case NeedsDurationConfirmation(missing=missing):
            return Reply(
                f"How long does {_q(missing.trigger_title)} usually last?", expects_reply=True
            )
        case NeedsAnchorDuration(needs=needs):
            return Reply(
                f"I don't know how long {_q(needs.anchor_title)} lasts, and I need it to place "
                f"{_q(needs.trigger_title)}. First add {needs.anchor_title} with a start and an "
                f'end (for example "{needs.anchor_title} from 18 to 19").'
            )
        case NeedsGapConfirmation(needs_confirmation=gap):
            return _gap_question(gap, today)
        case RelativeNeedsInfo():
            return Reply("I need to know relative to what, and for how long.", expects_reply=True)
        case AnchorNotFound():
            description = call.anchor_description if isinstance(call, AddEventRelativeCall) else ""
            return Reply(
                f"I can't find any event {_q(description or '?')} in the schedule "
                f"{_on(_call_date(call, today), today)}.",
                expects_reply=True,
            )
        case AnchorAmbiguous(candidates=candidates):
            return _ambiguous(candidates, today)
        case EventNotFound():
            description = getattr(call, "event_description", None)
            what = _q(description) if description else f"at {getattr(call, 'event_time', '?')}"
            when = _on(getattr(call, "event_date", None) or today, today)
            return Reply(
                f"I can't find any event {what} in the schedule {when}.", expects_reply=True
            )
        case EventAmbiguous(candidates=candidates):
            return _ambiguous(candidates, today)
        case EventUpdated():
            return _updated(outcome, today)
        case EventRemoved():
            return _removed(outcome, today)
        case CheckOverlapResult():
            return _overlap(call, outcome, today)
        case ScheduleView():
            return _schedule_reply(outcome, today)
        case FreeSlotsView():
            return Reply(_free_slots_text(outcome, today))
        case AddEventRuleSuccess() if isinstance(call, AddEventRuleCall):
            return Reply(_rule_added(call))
        case AddEventRuleNeedsPosition():
            return Reply(
                f"Does {_q(outcome.trigger_title)} go before or after {_q(outcome.related_title)}?",
                BEFORE_AFTER,
                expects_reply=True,
            )
        case AddEventRuleInvalidFormat():
            return Reply(
                "I didn't understand the time: write it as HH:MM, for example 18:30.",
                expects_reply=True,
            )
        case AddEventRuleNeedsInfo():
            return Reply(
                f"What do you want me to remember for {_q(outcome.trigger_title)}: a usual time, "
                "a duration or a rule relative to another event?",
                expects_reply=True,
            )
        case RemoveEventRuleSuccess() if isinstance(call, RemoveEventRuleCall):
            return Reply(_rule_removed(call))
        case RemoveEventRuleNotFound():
            return Reply("I can't find any matching rule.")
        case RemoveEventRuleAmbiguous(candidates=candidates):
            listing = ", ".join(
                f"{i}) {c.trigger_title} relative to {c.related_title}"
                for i, c in enumerate(candidates, start=1)
            )
            return Reply(
                f"I found several rules: {listing}. Which one do you mean?",
                tuple(f"{c.trigger_title} relative to {c.related_title}" for c in candidates),
                expects_reply=True,
            )
        case HabitDismissed(proposal=None):
            return Reply("There was no suggestion to reject.")
        case HabitDismissed():
            return Reply("Ok, I won't suggest it again.")
    return Reply(GENERIC)


def _call_date(call: object, today: date) -> date | str:
    return getattr(call, "date", None) or today


# --- creation, change, deletion --------------------------------------------------------


def _created(outcome: EventCreated, today: date) -> Reply:
    text = (
        f"Done: {outcome.title} {day_label(_date_of(outcome.start), today)} "
        f"{_span(outcome.start, outcome.end)}."
    )
    if outcome.conflicts:
        text += f" Warning: it overlaps with {_titles(outcome.conflicts)}."
    if outcome.rule_proposals:
        proposal = outcome.rule_proposals[0]
        text += (
            f" Shall I also add {_q(proposal.trigger_title)} {_span(proposal.start, proposal.end)}?"
        )
        return Reply(text, YES_NO, expects_reply=True)
    return Reply(text)


def _updated(outcome: EventUpdated, today: date) -> Reply:
    moved = (outcome.start, outcome.end) != (outcome.original_start, outcome.original_end)
    if moved:
        text = (
            f"Done: I moved {_q(outcome.title)} to "
            f"{day_label(_date_of(outcome.start), today)} {_span(outcome.start, outcome.end)}."
        )
    else:
        text = f"Done: {_q(outcome.original_title)} is now called {_q(outcome.title)}."
    if outcome.conflicts:
        text += f" Warning: it overlaps with {_titles(outcome.conflicts)}."
    if outcome.cascade_candidates:
        shift = minutes_between_iso(outcome.original_start, outcome.start)
        candidate = outcome.cascade_candidates[0]
        new_start = add_minutes_to_iso(candidate.start, shift)
        new_end = add_minutes_to_iso(candidate.end, shift)
        text += f" Shall I also move {_q(candidate.title)} {_span(new_start, new_end)}?"
        return Reply(text, YES_NO, expects_reply=True)
    return Reply(text)


def _removed(outcome: EventRemoved, today: date) -> Reply:
    text = (
        f"Done: I deleted {_q(outcome.removed_title)} "
        f"({day_label(_date_of(outcome.start), today)} {_range(outcome.start, outcome.end)})."
    )
    if outcome.cascade_candidates:
        candidate = outcome.cascade_candidates[0]
        span = _range(candidate.start, candidate.end)
        text += f" Shall I also delete {_q(candidate.title)} ({span})?"
        return Reply(text, YES_NO, expects_reply=True)
    return Reply(text)


def _ambiguous(candidates: tuple[EventInfo, ...], today: date) -> Reply:
    when = _on(_date_of(candidates[0].start), today)
    return Reply(
        f"I found several events {when}: {_numbered(candidates)}. Which one do you mean?",
        _candidate_replies(candidates),
        expects_reply=True,
    )


def _gap_question(gap, today: date) -> Reply:
    rule = gap.gaps[0]
    where = "After" if rule.position == "after" else "Before"
    when = _on(_date_of(gap.start_with_gap), today)
    return Reply(
        f"{where} {_q(gap.anchor_title)} there is usually a {rule.gap_minutes}-minute gap: "
        f"{_q(gap.trigger_title)} would be {when} at {_time_of(gap.start_with_gap)} with the "
        f"gap, at {_time_of(gap.start_without_gap)} without it. Shall I keep the gap?",
        YES_NO,
        expects_reply=True,
    )


def _overlap(call: object, outcome: CheckOverlapResult, today: date) -> Reply:
    assert isinstance(call, CheckOverlapCall)
    assert call.time is not None and call.end_time is not None
    day = call.date or today.isoformat()
    start, end = combine_date_and_time(day, call.time), combine_date_and_time(day, call.end_time)
    interval = f"{day_label(_date_of(start), today)} {_span(start, end)}"
    if outcome.overlap:
        return Reply(f"Yes: {interval} overlaps with {_titles(outcome.conflicts)}.")
    return Reply(f"No: {interval} you are free.")


# --- reads -----------------------------------------------------------------------------


def _prefix(day: str, band: object, today: date) -> str:
    label = _capfirst(day_label(day, today))
    return f"{label} {BAND_PHRASES[band.value]}" if band else label


class _StyledText:
    """A reply written piece by piece, each piece with its styles."""

    def __init__(self) -> None:
        self._text = ""
        self._styles: list[Styled] = []

    def add(self, piece: str, *styles: Literal["bold", "strikethrough"]) -> None:
        start = len(self._text)
        self._text += piece
        self._styles.extend(Styled(style, start, len(self._text)) for style in styles)

    def reply(self) -> Reply:
        return Reply(self._text, styles=tuple(self._styles))


def _schedule_reply(view: ScheduleView, today: date) -> Reply:
    """A header with the day, then one card per event: the title in bold, then start, end and
    duration; the events already over are struck through. A light line tells the free time
    between two events; a whole day ends with how much time passes until the first event of
    the next day, and that event. Gaps run from the latest end so far: overlaps have none."""
    out = _StyledText()
    out.add(_day_header(view.date, view.time_band, today))
    if not view.entries:
        out.add("\n\nNothing planned.")
    latest_end = None
    for entry in view.entries:
        if latest_end is not None and (gap := minutes_between_iso(latest_end, entry.start)) > 0:
            out.add(f"\n\n┄┄ {_duration(gap)} ┄┄")
        _card(out, entry)
        latest_end = max(latest_end or entry.end, entry.end)
    if view.following is not None and latest_end is not None:
        gap = minutes_between_iso(latest_end, view.following.start)
        out.add(f"\n\n──── {_duration(gap)} ────" if gap > 0 else "\n\n────────────")
        out.add(f"\n\n{_day_header(_date_of(view.following.start).isoformat(), None, today)}")
        _card(out, view.following)
    return out.reply()


def _day_header(day: str, band: object, today: date) -> str:
    """The day, with its date after a relative one: "Today · Monday, September 21"."""
    header = _prefix(day, band, today)
    if day_label(day, today)[:1].islower():
        header += f" · {_calendar_date(date.fromisoformat(day), today)}"
    return header


def _card(out: _StyledText, entry: ScheduleEntry) -> None:
    over = ("strikethrough",) if entry.done else ()
    out.add("\n\n")
    out.add(entry.title, "bold", *over)
    out.add("\n")
    minutes = minutes_between_iso(entry.start, entry.end)
    out.add(
        f"{_time_of(entry.start)} → {_end_of(entry.start, entry.end)} · {_duration(minutes)}",
        *over,
    )


def _duration(minutes: int) -> str:
    """The duration as "45 min", "1 h" or "1 h 30 min"."""
    hours, rest = divmod(minutes, 60)
    if not hours:
        return f"{rest} min"
    return f"{hours} h {rest} min" if rest else f"{hours} h"


def _free_slots_text(view: FreeSlotsView, today: date) -> str:
    prefix = _prefix(view.date, view.time_band, today)
    if not view.slots:
        minimum = (
            f" of at least {view.min_duration_minutes} minutes" if view.min_duration_minutes else ""
        )
        return f"{prefix} I found no free slots{minimum}."
    listing = ", ".join(_range(s.start, s.end) for s in view.slots)
    return f"{prefix} you are free: {listing}."


# --- rules -----------------------------------------------------------------------------


def _rule_added(call: AddEventRuleCall) -> str:
    title = _q(call.trigger_title or "?")
    if call.related_title is None:
        parts = []
        if call.typical_start:
            parts.append(f"is at {call.typical_start}")
        if call.typical_duration_minutes:
            parts.append(f"lasts {call.typical_duration_minutes} minutes")
        return f"Done: {title} usually " + " and ".join(parts) + "."
    related = _q(call.related_title)
    if call.gap_minutes is not None:
        where = "after" if call.position == "after" else "before"
        return f"Done: {where} {related} I will always leave a {call.gap_minutes}-minute gap."
    if call.position == "after":
        offset = (
            f", {call.offset_minutes} minutes after" if call.offset_minutes else ", right after"
        )
        return f"Done: {title} always goes after {related}{offset}."
    offset = f", {call.offset_minutes} minutes before" if call.offset_minutes else ", right before"
    return f"Done: {title} always goes before {related}{offset}."


def _rule_removed(call: RemoveEventRuleCall) -> str:
    title = _q(call.trigger_title or "?")
    if call.related_title:
        return f"Done: I removed the rule of {title} relative to {_q(call.related_title)}."
    what = {
        "typical_start": "the usual time",
        "typical_duration_minutes": "the usual duration",
    }.get(call.field or "", "the usual time and duration")
    return f"Done: I removed {what} of {title}."


def rules_text(rules: list[EventRule], defaults: list[EventDefault]) -> str:
    """What the schedule knows: relations between events, free slots and usual times."""
    lines = []
    for rule in rules:
        related = _q(rule.related_title)
        where = "after" if rule.position == "after" else "before"
        if rule.is_gap:
            lines.append(f"- {rule.gap_minutes} minutes free {where} {related}")
        elif rule.offset_minutes:
            lines.append(
                f"- {_q(rule.trigger_title)} {rule.offset_minutes} minutes {where} {related}"
            )
        else:
            lines.append(f"- {_q(rule.trigger_title)} right {where} {related}")
    for default in defaults:
        parts = []
        if default.typical_start:
            parts.append(f"at {default.typical_start}")
        if default.typical_duration_minutes:
            parts.append(f"lasts {default.typical_duration_minutes} minutes")
        if parts:
            lines.append(f"- {_q(default.title)} usually " + " and ".join(parts))
    if not lines:
        return (
            "There are no rules yet. Teach me one by writing, for example: "
            '"the shower always goes after the gym".'
        )
    return "Rules and usual times:\n" + "\n".join(lines)


# --- undo ------------------------------------------------------------------------------


def undone_text(undone: Undone | None, today: date) -> str:
    if undone is None:
        return NOTHING_TO_UNDO
    title = _q(undone.title)
    when = f"{_on(undone.start.split('T')[0], today)} {_span(undone.start, undone.end)}"
    if undone.kind == "add_event":
        return f"Undone: I deleted {title} {when}."
    return f"Undone: {title} is back {when}."


# --- habits ----------------------------------------------------------------------------


def _habit(habit: HabitProposal) -> Reply:
    match habit:
        case EventPatternProposal():
            return Reply(
                f"I noticed you often do {_q(habit.title)} around {habit.typical_start} "
                f"({habit.typical_duration_minutes} minutes): shall I save it as a usual time?",
                YES_NO,
                expects_reply=True,
            )
        case SequencePatternProposal():
            gap = (
                "right after"
                if habit.offset_minutes == 0
                else f"{habit.offset_minutes} minutes after"
            )
            return Reply(
                f"You often add {_q(habit.trigger_title)} {gap} {_q(habit.related_title)}: "
                "shall I save it as a rule?",
                YES_NO,
                expects_reply=True,
            )
        case GapPatternProposal():
            where = "after" if habit.position == "after" else "before"
            return Reply(
                f"You often leave about {habit.gap_minutes} minutes free {where} "
                f"{_q(habit.related_title)}. If you want to save it as a rule, tell me what to "
                'call it (for example "break").',
                ("No",),
                expects_reply=True,
            )
    raise AssertionError(f"Unhandled proposal: {habit!r}")


# --- missing data ----------------------------------------------------------------------


def _missing_field(missing: MissingField) -> Reply:
    field, subject = missing.field, missing.subject
    match field:
        case "title":
            return Reply("What is the event called?", expects_reply=True)
        case "event_description":
            return Reply("Which event do you mean?", expects_reply=True)
        case "trigger_title" if subject == "gap":
            return Reply("What do you want to call this free slot?", expects_reply=True)
        case "trigger_title":
            return Reply("Which event is it about?", expects_reply=True)
        case "time":
            return Reply("At what time?", expects_reply=True)
        case "end_time":
            return Reply("Until what time?", expects_reply=True)
        case "anchor_description":
            return Reply(
                f"Relative to which event do you want to place {_name(subject)}?",
                expects_reply=True,
            )
        case "position":
            return Reply(
                f"Does {_name(subject)} go before or after?",
                BEFORE_AFTER,
                expects_reply=True,
            )
        case "duration_minutes":
            return Reply(f"How long does {_name(subject)} last?", expects_reply=True)
        case "change":
            return Reply(
                f"What do you want to change about {_name(subject)}: the time, the duration or "
                "the name?",
                expects_reply=True,
            )
    return Reply("Can you give me more details?", expects_reply=True)
