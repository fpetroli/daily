"""Habit detection: counting on every added event and a proposal when the threshold is hit.

Counting (`record_habit_signals`) runs after every insertion, whatever its origin. The
proposal (`next_habit_proposal`) is a read: there is no pending state, what was proposed lives
in the chat history and the user's answer becomes a tool call (`add_event_rule` for yes,
`dismiss_habit` for no).
"""

from dataclasses import dataclass
from datetime import datetime

from daily.domain.models import ScheduleEvent
from daily.domain.patterns import NoteStatus
from daily.domain.timeutil import minutes_between_iso, time_of_day_from_iso
from daily.services.context import Repositories


@dataclass(frozen=True, slots=True)
class EventPatternProposal:
    note_id: str
    title: str
    typical_start: str
    typical_duration_minutes: int


@dataclass(frozen=True, slots=True)
class SequencePatternProposal:
    note_id: str
    trigger_title: str
    related_title: str
    position: str
    offset_minutes: int


@dataclass(frozen=True, slots=True)
class GapPatternProposal:
    note_id: str
    related_title: str
    position: str
    gap_minutes: int


HabitProposal = EventPatternProposal | SequencePatternProposal | GapPatternProposal


def record_habit_signals(
    repos: Repositories,
    *,
    added_event: ScheduleEvent,
    derived_from_rule_id: str | None,
    now: datetime | None = None,
) -> None:
    """Updates the tracking after `added_event` is inserted.

    - `event_patterns`: only if the event does not come from a rule and the title has no
      complete usual time yet (do not track what is already known);
    - `sequence_patterns`: with the immediate neighbour before/after on the same day, only
      if no rule exists yet for the pair;
    - `gap_patterns`: same neighbour, regardless of rules.

    No proposal here: counting only.
    """
    tolerance = repos.app_settings.time_tolerance()

    if derived_from_rule_id is None:
        existing = repos.event_defaults.get_by_title(added_event.title)
        already_complete = (
            existing is not None
            and existing.typical_start is not None
            and existing.typical_duration_minutes is not None
        )
        if not already_complete:
            repos.event_patterns.record_occurrence(
                title=added_event.title,
                observed_start=time_of_day_from_iso(added_event.start),
                observed_duration_minutes=minutes_between_iso(added_event.start, added_event.end),
                time_tolerance_minutes=tolerance,
                duration_tolerance_minutes=repos.app_settings.duration_tolerance(),
                now=now,
            )

    date = added_event.start.split("T")[0]
    others = [e for e in repos.schedule.events_for_date(date) if e.event_id != added_event.event_id]

    predecessors = [e for e in others if e.end <= added_event.start]
    successors = [e for e in others if e.start >= added_event.end]
    predecessor = max(predecessors, key=lambda e: e.end, default=None)
    successor = min(successors, key=lambda e: e.start, default=None)

    if predecessor is not None:
        _record_adjacency(
            repos,
            trigger=added_event,
            related=predecessor,
            gap_minutes=minutes_between_iso(predecessor.end, added_event.start),
            tolerance=tolerance,
            now=now,
        )
    if successor is not None:
        _record_adjacency(
            repos,
            trigger=successor,
            related=added_event,
            gap_minutes=minutes_between_iso(added_event.end, successor.start),
            tolerance=tolerance,
            now=now,
        )


def _record_adjacency(
    repos: Repositories,
    *,
    trigger: ScheduleEvent,
    related: ScheduleEvent,
    gap_minutes: int,
    tolerance: int,
    now: datetime | None,
) -> None:
    """`trigger` right after `related` (no event in between): `position` is always `after`."""
    if gap_minutes < 0:
        return  # overlapping, not adjacent

    if not repos.event_rules.find_matching(trigger.title, related.title):
        repos.sequence_patterns.record_occurrence(
            trigger_title=trigger.title,
            related_title=related.title,
            position="after",
            observed_offset_minutes=gap_minutes,
            time_tolerance_minutes=tolerance,
            now=now,
        )
    repos.gap_patterns.record_occurrence(
        related_title=related.title,
        position="after",
        observed_gap_minutes=gap_minutes,
        time_tolerance_minutes=tolerance,
        now=now,
    )


def next_habit_proposal(repos: Repositories) -> HabitProposal | None:
    """The first pending pattern that has reached the threshold (event → sequence → gap).

    A broad read over the three tables: it also picks patterns that crossed the threshold in
    an earlier turn without ever being proposed. One candidate at a time.
    """
    threshold = repos.app_settings.repetition_threshold()

    for e in repos.event_patterns.list_pending():
        if e.occurrence_count >= threshold:
            return EventPatternProposal(
                e.note_id, e.title, e.typical_start, e.typical_duration_minutes
            )
    for s in repos.sequence_patterns.list_pending():
        if s.occurrence_count >= threshold:
            return SequencePatternProposal(
                s.note_id, s.trigger_title, s.related_title, s.position, s.offset_minutes
            )
    for g in repos.gap_patterns.list_pending():
        if g.occurrence_count >= threshold:
            return GapPatternProposal(g.note_id, g.related_title, g.position, g.gap_minutes)
    return None


def dismiss_habit(repos: Repositories) -> HabitProposal | None:
    """The user rejects the current habit proposal: it will not be proposed again.

    The current proposal is the one `next_habit_proposal` returns now: it was shown at the
    end of the turn that triggered it, and nothing replaces it until the user answers.
    Returns what was rejected, `None` if there was nothing.
    """
    proposal = next_habit_proposal(repos)
    if proposal is None:
        return None
    match proposal:
        case EventPatternProposal():
            repos.event_patterns.set_status(proposal.note_id, NoteStatus.REJECTED)
        case SequencePatternProposal():
            repos.sequence_patterns.set_status(proposal.note_id, NoteStatus.REJECTED)
        case GapPatternProposal():
            repos.gap_patterns.set_status(proposal.note_id, NoteStatus.REJECTED)
    return proposal


def confirm_event_patterns(repos: Repositories, *, title: str) -> None:
    """A complete usual time for `title` closes the pending patterns with the same title."""
    for pattern in repos.event_patterns.list_pending():
        if pattern.title.lower() == title.lower():
            repos.event_patterns.set_status(pattern.note_id, NoteStatus.CONFIRMED)


def confirm_sequence_patterns(
    repos: Repositories, *, trigger_title: str, related_title: str
) -> None:
    """A rule for the pair closes the pending sequence patterns of the same pair."""
    for pattern in repos.sequence_patterns.list_pending():
        if (
            pattern.trigger_title.lower() == trigger_title.lower()
            and pattern.related_title.lower() == related_title.lower()
        ):
            repos.sequence_patterns.set_status(pattern.note_id, NoteStatus.CONFIRMED)


def confirm_gap_patterns(
    repos: Repositories, *, related_title: str, position: str, gap_minutes: int
) -> None:
    """A gap rule closes the pending gap patterns for the same anchor and position.

    Only those whose minutes are within the tolerance: a different gap is another habit.
    """
    tolerance = repos.app_settings.time_tolerance()
    for pattern in repos.gap_patterns.list_pending():
        if (
            pattern.related_title.lower() == related_title.lower()
            and pattern.position == position
            and abs(pattern.gap_minutes - gap_minutes) <= tolerance
        ):
            repos.gap_patterns.set_status(pattern.note_id, NoteStatus.CONFIRMED)
