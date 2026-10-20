"""Domain models: one per table row, immutable.

`start`/`end` stay ISO strings (``YYYY-MM-DDTHH:MM``) rather than `datetime`: the exact
format used by tools and model, with no conversion that could introduce a different time
zone or precision.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from daily.domain.patterns import NoteStatus

Position = Literal["before", "after"]


@dataclass(frozen=True, slots=True)
class ScheduleEvent:
    event_id: str
    title: str
    start: str
    end: str
    derived_from_event_id: str | None = None
    derived_from_rule_id: str | None = None


@dataclass(frozen=True, slots=True)
class EventRule:
    """Relational rule.

    `trigger_title` is the dependent event (whose position the rule decides),
    `related_title` the independent anchor. When `gap_minutes` is set the row is a gap
    (a recurring empty slot tied to the anchor), not a relation with a duration.
    """

    rule_id: str
    trigger_title: str
    related_title: str
    position: str
    offset_minutes: int | None = None
    gap_minutes: int | None = None

    @property
    def is_gap(self) -> bool:
        return self.gap_minutes is not None


@dataclass(frozen=True, slots=True)
class EventDefault:
    """Usual time and duration of a single event, keyed by title (NOCASE)."""

    title: str
    typical_start: str | None = None
    typical_duration_minutes: int | None = None


@dataclass(frozen=True, slots=True)
class EventPattern:
    """Standalone event repeated at a similar time and duration."""

    note_id: str
    title: str
    typical_start: str
    typical_duration_minutes: int
    occurrence_count: int
    first_occurrence_at: datetime
    status: NoteStatus


@dataclass(frozen=True, slots=True)
class TriggerDurationPattern:
    """Repeated duration of an event that has a rule but no known duration."""

    note_id: str
    title: str
    typical_duration_minutes: int
    occurrence_count: int
    first_occurrence_at: datetime
    status: NoteStatus


@dataclass(frozen=True, slots=True)
class SequencePattern:
    """Relation between two consecutive events, not taught yet."""

    note_id: str
    trigger_title: str
    related_title: str
    position: str
    offset_minutes: int
    occurrence_count: int
    first_occurrence_at: datetime
    status: NoteStatus


@dataclass(frozen=True, slots=True)
class GapPattern:
    """Recurring empty slot tied to an anchor."""

    note_id: str
    related_title: str
    position: str
    gap_minutes: int
    occurrence_count: int
    first_occurrence_at: datetime
    status: NoteStatus
