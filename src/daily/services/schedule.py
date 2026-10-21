"""Schedule operations: create, move, delete, read, look for free slots.

Exact functions with typed outcomes, used by the adapters of the model's tools. They do
not raise on plausible input: every case that needs a question to the user is an outcome,
not an exception. Dates/times arrive already as ISO `YYYY-MM-DDTHH:MM` (validated by the
adapters) and the code does not reinterpret them. A conflict never cancels the write: it is
only reported.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

from daily.domain.timeband import TimeBand, band_bounds, filter_by_band
from daily.domain.timeutil import (
    add_minutes_to_iso,
    combine_date_and_time,
    minutes_between_iso,
    now_iso_string,
    parse_iso,
    today_date_string,
)
from daily.services.context import Repositories
from daily.services.habits import record_habit_signals
from daily.services.placement import (
    AnchorDurationMissing,
    AnchorGapConfirmationNeeded,
    AnchorMissing,
    AnchorNeedsDuration,
    AnchorResolved,
    AnchorUnknown,
    resolve_anchor_placement,
)
from daily.services.resolver import ResolveError
from daily.services.rules import compute_rule_proposals, infer_provenance
from daily.services.types import (
    CandidateInfo,
    CascadeCandidate,
    CheckOverlapResult,
    ConflictInfo,
    EventInfo,
    FreeSlot,
    RuleProposal,
    ScheduleEntry,
)
from daily.storage.db import transaction

# How many days after today an event is looked for when the user does not say its day.
LOOK_AHEAD_DAYS = 14

# --- shared outcomes ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InvalidRange:
    """`end` not after `start`: checked before writing, never saved."""


@dataclass(frozen=True, slots=True)
class EventNotFound:
    pass


@dataclass(frozen=True, slots=True)
class EventAlreadyThere:
    """An event with the same title, start and end is already in the schedule: a second copy
    is never what the user wants (the model sometimes repeats the previous addition)."""

    title: str
    start: str
    end: str


@dataclass(frozen=True, slots=True)
class EventAmbiguous:
    candidates: tuple[CandidateInfo, ...]


# --- create --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventCreated:
    event_id: str
    title: str
    start: str
    end: str
    conflicts: tuple[ConflictInfo, ...] = ()
    rule_proposals: tuple[RuleProposal, ...] = ()


@dataclass(frozen=True, slots=True)
class InsufficientInfo:
    """A value no rule can fill in is missing: `start` or the duration."""

    missing: Literal["start", "duration"]


@dataclass(frozen=True, slots=True)
class NeedsAnchorConfirmation:
    missing: AnchorMissing


@dataclass(frozen=True, slots=True)
class NeedsDurationConfirmation:
    missing: AnchorDurationMissing


@dataclass(frozen=True, slots=True)
class NeedsAnchorDuration:
    needs: AnchorNeedsDuration


@dataclass(frozen=True, slots=True)
class NeedsGapConfirmation:
    needs_confirmation: AnchorGapConfirmationNeeded


CreateEventOutcome = (
    EventCreated
    | EventAlreadyThere
    | InvalidRange
    | InsufficientInfo
    | NeedsAnchorConfirmation
    | NeedsDurationConfirmation
    | NeedsAnchorDuration
    | NeedsGapConfirmation
)


def persist_new_event(
    repos: Repositories,
    *,
    title: str,
    start: str,
    end: str,
    derived_from_event_id: str | None = None,
    derived_from_rule_id: str | None = None,
    now: datetime | None = None,
) -> EventCreated | EventAlreadyThere | InvalidRange:
    """Common path of `create_event` and `create_event_relative`, from already resolved times.

    Order: range check → exact copy check → provenance (explicit, otherwise inferred) →
    insertion → conflicts → downstream proposals → habit tracking. The writes are in a single
    transaction.
    """
    if parse_iso(end) <= parse_iso(start):
        return InvalidRange()
    for same in repos.schedule.find_conflicts(start=start, end=end):
        if same.title.lower() == title.lower() and (same.start, same.end) == (start, end):
            return EventAlreadyThere(same.title, same.start, same.end)

    if derived_from_event_id is None and derived_from_rule_id is None:
        derived_from_event_id, derived_from_rule_id = infer_provenance(
            repos, title=title, start=start, end=end
        )

    with transaction(repos.conn):
        event_id = repos.schedule.insert_event(
            title=title,
            start=start,
            end=end,
            derived_from_event_id=derived_from_event_id,
            derived_from_rule_id=derived_from_rule_id,
        )
        conflicts = repos.schedule.find_conflicts(start=start, end=end, exclude_event_id=event_id)
        added = repos.schedule.get_by_id(event_id)
        assert added is not None  # just inserted in the same transaction
        proposals = compute_rule_proposals(repos, anchor=added)
        record_habit_signals(
            repos, added_event=added, derived_from_rule_id=derived_from_rule_id, now=now
        )

    return EventCreated(
        event_id=event_id,
        title=title,
        start=start,
        end=end,
        conflicts=tuple(EventInfo.from_event(c) for c in conflicts),
        rule_proposals=tuple(proposals),
    )


def create_event(
    repos: Repositories,
    *,
    title: str,
    date: str,
    start: str | None = None,
    end: str | None = None,
    duration_minutes: int | None = None,
    with_gap: bool | None = None,
    anchor_time: str | None = None,
    derived_from_event_id: str | None = None,
    derived_from_rule_id: str | None = None,
    now: datetime | None = None,
) -> CreateEventOutcome:
    """Creates an event, with an absolute time or one resolved from rules and usual times.

    With `start` the end comes, in order, from `duration_minutes` (it wins over an explicit
    `end`: the code does the arithmetic, never the model), from `end`, from the title's usual
    duration; if there is none, the duration is asked. Without `start` the "before"
    resolution runs on day `date`. `with_gap` answers the "with or without a break"
    question: without it the outcome is `NeedsGapConfirmation` and nothing is written.
    `anchor_time` (`HH:MM`) is the time of the rule's anchor when it is not in the schedule:
    whoever answers "what time is it at?" passes it here, with no pending state.
    """
    if start is not None:
        if duration_minutes is not None:
            end = add_minutes_to_iso(start, duration_minutes)
        elif end is None:
            defaults = repos.event_defaults.get_by_title(title)
            if defaults is None or defaults.typical_duration_minutes is None:
                return InsufficientInfo(missing="duration")
            end = add_minutes_to_iso(start, defaults.typical_duration_minutes)
    else:
        placement = resolve_anchor_placement(
            repos,
            trigger_title=title,
            date=date,
            duration_minutes=duration_minutes,
            anchor_time=anchor_time,
        )
        match placement:
            case AnchorResolved():
                start, end = placement.start, placement.end
            case AnchorGapConfirmationNeeded():
                if with_gap is None:
                    return NeedsGapConfirmation(placement)
                if with_gap:
                    start, end = placement.start_with_gap, placement.end_with_gap
                else:
                    start, end = placement.start_without_gap, placement.end_without_gap
            case AnchorMissing():
                return NeedsAnchorConfirmation(placement)
            case AnchorDurationMissing():
                return NeedsDurationConfirmation(placement)
            case AnchorNeedsDuration():
                return NeedsAnchorDuration(placement)
            case AnchorUnknown():
                return InsufficientInfo(missing="start")

    return persist_new_event(
        repos,
        title=title,
        start=start,
        end=end,
        derived_from_event_id=derived_from_event_id,
        derived_from_rule_id=derived_from_rule_id,
        now=now,
    )


# --- move ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventUpdated:
    """The `original_*` values serve whoever computes how far to move the derived events
    (same delta) and whoever tells what changed."""

    event_id: str
    title: str
    start: str
    end: str
    original_start: str
    original_end: str
    original_title: str
    conflicts: tuple[ConflictInfo, ...] = ()
    cascade_candidates: tuple[CascadeCandidate, ...] = ()


def event_day(
    repos: Repositories,
    *,
    description: str,
    at_time: str | None,
    today: str,
    days: int = LOOK_AHEAD_DAYS,
) -> str:
    """The day of an event the user names without a day: today if it is there, otherwise the
    nearest of the next `days` days that has it. Today again if no day has it, so that the
    "not found" is about today."""
    first = date.fromisoformat(today)
    for offset in range(days + 1):
        day = (first + timedelta(days=offset)).isoformat()
        found = repos.resolver.resolve(description=description, date=day, at_time=at_time)
        if found.error != ResolveError.NOT_FOUND:
            return day
    return today


UpdateEventOutcome = EventUpdated | EventNotFound | EventAmbiguous | InvalidRange


def update_event(
    repos: Repositories,
    *,
    event_description: str,
    default_date: str | None = None,
    event_date: str | None = None,
    event_time: str | None = None,
    new_start: str | None = None,
    new_end: str | None = None,
    new_duration_minutes: int | None = None,
    new_title: str | None = None,
    shift_minutes: int | None = None,
    extend_minutes: int | None = None,
) -> UpdateEventOutcome:
    """Moves or changes an existing event.

    The event is found with `event_description` on day `event_date` (or `default_date`);
    `event_time` disambiguates events with the same name. Without a date there is nothing to
    search: not found.

    Precedence for the new range: `shift_minutes` only if there is no `new_start`, `new_end`
    or `new_duration_minutes` (it is an alternative to absolute values); otherwise the start
    is `new_start` or the existing one and the end is, in order, `new_duration_minutes`,
    `new_end`, the original duration if only the start changed, the existing end.
    `extend_minutes` (signed) is then added to the end computed this way: "make it half an
    hour longer" does not require knowing the current end.
    """
    resolved = repos.resolver.resolve(
        description=event_description,
        date=event_date or default_date or "",
        at_time=event_time,
    )
    if resolved.error == ResolveError.NOT_FOUND:
        return EventNotFound()
    if resolved.error == ResolveError.AMBIGUOUS:
        return EventAmbiguous(tuple(EventInfo.from_event(e) for e in resolved.candidates))
    event = resolved.event
    assert event is not None

    if (
        shift_minutes is not None
        and new_start is None
        and new_end is None
        and new_duration_minutes is None
    ):
        start = add_minutes_to_iso(event.start, shift_minutes)
        end = add_minutes_to_iso(event.end, shift_minutes)
    else:
        start = new_start or event.start
        if new_duration_minutes is not None:
            end = add_minutes_to_iso(start, new_duration_minutes)
        elif new_end is not None:
            end = new_end
        elif new_start is not None:
            end = add_minutes_to_iso(start, minutes_between_iso(event.start, event.end))
        else:
            end = event.end
    if extend_minutes is not None:
        end = add_minutes_to_iso(end, extend_minutes)

    if parse_iso(end) <= parse_iso(start):
        return InvalidRange()

    repos.schedule.update_event(event.event_id, title=new_title, start=start, end=end)
    conflicts = repos.schedule.find_conflicts(start=start, end=end, exclude_event_id=event.event_id)
    # Cascade only if the event really moved: a title-only change has no derived events to
    # recompute.
    moved = start != event.start or end != event.end
    derived = repos.schedule.events_derived_from(event.event_id) if moved else []

    return EventUpdated(
        event_id=event.event_id,
        title=new_title or event.title,
        start=start,
        end=end,
        original_start=event.start,
        original_end=event.end,
        original_title=event.title,
        conflicts=tuple(EventInfo.from_event(c) for c in conflicts),
        cascade_candidates=tuple(EventInfo.from_event(d) for d in derived),
    )


# --- delete --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventRemoved:
    """Title and time of the deleted event are needed by the (self-contained) message."""

    event_id: str
    removed_title: str
    start: str
    end: str
    cascade_candidates: tuple[CascadeCandidate, ...] = ()


RemoveEventOutcome = EventRemoved | EventNotFound | EventAmbiguous


def remove_event(
    repos: Repositories,
    *,
    event_description: str,
    default_date: str | None = None,
    event_date: str | None = None,
    event_time: str | None = None,
) -> RemoveEventOutcome:
    """Deletes an event. Derived events are proposed, never removed along with it.

    Covers only the cascade with tracked provenance (`derived_from_event_id`).
    """
    resolved = repos.resolver.resolve(
        description=event_description,
        date=event_date or default_date or "",
        at_time=event_time,
    )
    if resolved.error == ResolveError.NOT_FOUND:
        return EventNotFound()
    if resolved.error == ResolveError.AMBIGUOUS:
        return EventAmbiguous(tuple(EventInfo.from_event(e) for e in resolved.candidates))
    event = resolved.event
    assert event is not None

    with transaction(repos.conn):
        derived = repos.schedule.events_derived_from(event.event_id)
        repos.schedule.delete_event(event.event_id)

    return EventRemoved(
        event_id=event.event_id,
        removed_title=event.title,
        start=event.start,
        end=event.end,
        cascade_candidates=tuple(EventInfo.from_event(d) for d in derived),
    )


# --- read ----------------------------------------------------------------------------


def check_overlap(repos: Repositories, *, start: str, end: str) -> CheckOverlapResult:
    conflicts = repos.schedule.find_conflicts(start=start, end=end)
    return CheckOverlapResult(
        overlap=bool(conflicts), conflicts=tuple(EventInfo.from_event(c) for c in conflicts)
    )


def get_schedule(
    repos: Repositories,
    *,
    default_date: str,
    date: str | None = None,
    time_band: TimeBand | None = None,
    now: datetime | None = None,
) -> list[ScheduleEntry]:
    """Schedule of one day. `done` is derived: the event is already over relative to `now`."""
    now_iso = now_iso_string(now)
    entries = [
        ScheduleEntry(e.event_id, e.title, e.start, e.end, done=e.end <= now_iso)
        for e in repos.schedule.events_for_date(date or default_date)
    ]
    return filter_by_band(entries, time_band) if time_band else entries


def get_free_slots(
    repos: Repositories,
    *,
    default_date: str,
    date: str | None = None,
    min_duration_minutes: int | None = None,
    time_band: TimeBand | None = None,
    now: datetime | None = None,
) -> list[FreeSlot]:
    """Gaps between the events of the whole day (00:00-24:00, no "working hours").

    Overlaps are merged first (a conflict does not cancel the write, so overlapping events
    can coexist). If the day is today the lower bound is the current time, not midnight:
    slots already in the past are not proposed. With `time_band` the slots are clipped to
    the band.
    """
    resolved_date = date or default_date
    midnight = combine_date_and_time(resolved_date, "00:00")
    day_end = add_minutes_to_iso(midnight, 24 * 60)
    now_iso = now_iso_string(now)
    is_today = resolved_date == today_date_string(now)
    day_start = now_iso if is_today and now_iso > midnight else midnight

    events = repos.schedule.events_for_date(resolved_date)
    merged: list[tuple[str, str]] = []
    for start, end in sorted((e.start, e.end) for e in events):
        if merged and parse_iso(start) <= parse_iso(merged[-1][1]):
            if parse_iso(end) > parse_iso(merged[-1][1]):
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    slots: list[FreeSlot] = []
    cursor = day_start
    for start, end in merged:
        if parse_iso(start) > parse_iso(cursor):
            slots.append(FreeSlot(cursor, start))
        if parse_iso(end) > parse_iso(cursor):
            cursor = end
    if parse_iso(day_end) > parse_iso(cursor):
        slots.append(FreeSlot(cursor, day_end))

    if time_band:
        slots = _clip_to_band(slots, resolved_date, time_band)
    minimum = min_duration_minutes or 0
    return [s for s in slots if minutes_between_iso(s.start, s.end) >= minimum]


def _clip_to_band(slots: list[FreeSlot], date: str, band: TimeBand) -> list[FreeSlot]:
    """Clips the slots to the band: a 17:00-24:00 gap is free "in the evening" too.

    Unlike the start-time filter of the other reads: for a gap what matters is how much of
    the band is free, not when it starts. The minimum duration applies after clipping ("a
    free hour tonight").
    """
    band_start, band_end = band_bounds(date, band)
    clipped = [FreeSlot(max(s.start, band_start), min(s.end, band_end)) for s in slots]
    return [s for s in clipped if s.start < s.end]
