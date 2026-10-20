"""Types shared by the services' outcomes.

Internal identifiers (`event_id`, `rule_id`) identify the rows; they are never shown to the
model: the adapters filter them out.
"""

from dataclasses import dataclass
from typing import Self

from daily.domain.models import ScheduleEvent


@dataclass(frozen=True, slots=True)
class EventInfo:
    """Reference to an event: in conflict, an ambiguous candidate or a cascade candidate."""

    event_id: str
    title: str
    start: str
    end: str

    @classmethod
    def from_event(cls, event: ScheduleEvent) -> Self:
        return cls(event.event_id, event.title, event.start, event.end)


# Same shape, three roles: the aliases keep it readable what each list is for.
ConflictInfo = EventInfo
CandidateInfo = EventInfo
CascadeCandidate = EventInfo


@dataclass(frozen=True, slots=True)
class RuleProposal:
    """Downstream candidate from `event_rules`: always proposed, never created on its own."""

    trigger_title: str
    start: str
    end: str


@dataclass(frozen=True, slots=True)
class RuleCandidateInfo:
    """Candidate of an ambiguous match on `remove_event_rule` (rules have no times)."""

    rule_id: str
    trigger_title: str
    related_title: str


@dataclass(frozen=True, slots=True)
class ScheduleEntry:
    """Row of a day's schedule; `done` is derived (the event is already over)."""

    event_id: str
    title: str
    start: str
    end: str
    done: bool = False


@dataclass(frozen=True, slots=True)
class FreeSlot:
    start: str
    end: str


@dataclass(frozen=True, slots=True)
class CheckOverlapResult:
    overlap: bool
    conflicts: tuple[EventInfo, ...]
