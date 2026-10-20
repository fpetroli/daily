"""Resolution of a natural-language description into an event of the schedule.

Shared by `remove_event`, `update_event` and the `anchor_description` of
`add_event_relative`. Algorithm: case-insensitive bidirectional substring match between
description and title. No inference on the date: the day comes already resolved from the
caller.
"""

from dataclasses import dataclass
from enum import StrEnum

from daily.domain.models import ScheduleEvent
from daily.domain.timeutil import time_of_day_from_iso
from daily.storage.schedule_repo import ScheduleRepository


class ResolveError(StrEnum):
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class ResolveResult:
    """Exactly one of `event` (success) and `error`; `candidates` only when ambiguous."""

    event: ScheduleEvent | None = None
    error: ResolveError | None = None
    candidates: tuple[ScheduleEvent, ...] = ()


class EventDescriptionResolver:
    def __init__(self, schedule: ScheduleRepository) -> None:
        self._schedule = schedule

    def resolve(
        self,
        *,
        description: str,
        date: str,
        at_time: str | None = None,
    ) -> ResolveResult:
        """Finds the event described by `description` on day `date`.

        `at_time` (`HH:MM`) picks among events with the same name: with a single match it is
        ignored, so a wrong time does not produce a false "not found". Without a description
        ("the 6 pm event") `at_time` is the only criterion: the day's events starting at that
        time are searched.
        """

        needle = description.lower().strip()
        if not needle and at_time is None:
            return ResolveResult(error=ResolveError.NOT_FOUND)

        day = self._schedule.events_for_date(date)
        if needle:
            matches = [e for e in day if needle in (t := e.title.lower()) or t in needle]
        else:
            matches = [e for e in day if time_of_day_from_iso(e.start) == at_time]
        if not matches:
            return ResolveResult(error=ResolveError.NOT_FOUND)
        if needle and len(matches) > 1 and at_time is not None:
            narrowed = [e for e in matches if time_of_day_from_iso(e.start) == at_time]
            if len(narrowed) == 1:
                return ResolveResult(event=narrowed[0])
        if len(matches) > 1:
            return ResolveResult(error=ResolveError.AMBIGUOUS, candidates=tuple(matches))
        return ResolveResult(event=matches[0])
