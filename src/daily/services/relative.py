"""Creation relative to an existing event: "the shower after the gym"."""

from dataclasses import dataclass
from datetime import datetime

from daily.services.context import Repositories
from daily.services.placement import place_relative_to_boundary
from daily.services.resolver import ResolveError
from daily.services.schedule import EventCreated, InvalidRange, persist_new_event
from daily.services.types import CandidateInfo, EventInfo


@dataclass(frozen=True, slots=True)
class RelativeNeedsInfo:
    """`anchor_description`, `position` or `duration_minutes` is missing: no sensible default.

    There are no defaults for these fields, so the user is always asked.
    """


@dataclass(frozen=True, slots=True)
class AnchorNotFound:
    pass


@dataclass(frozen=True, slots=True)
class AnchorAmbiguous:
    candidates: tuple[CandidateInfo, ...]


CreateRelativeOutcome = (
    EventCreated | InvalidRange | RelativeNeedsInfo | AnchorNotFound | AnchorAmbiguous
)


def create_event_relative(
    repos: Repositories,
    *,
    title: str,
    date: str,
    anchor_description: str | None,
    position: str | None,
    duration_minutes: int | None,
    offset_minutes: int | None = None,
    anchor_time: str | None = None,
    now: datetime | None = None,
) -> CreateRelativeOutcome:
    """Creates `title` right before or after the event described by `anchor_description`.

    The anchor is always an explicit phrase of the user, never a learned rule: neither
    `event_rules` nor gaps are read here. The search is on day `date`; `anchor_time`
    (`HH:MM`) disambiguates events with the same name. A missing offset is 0. Provenance is
    inferred as in `create_event` only if the position matches a rule for that title.
    """
    if anchor_description is None or position is None or duration_minutes is None:
        return RelativeNeedsInfo()

    resolved = repos.resolver.resolve(
        description=anchor_description, date=date, at_time=anchor_time
    )
    if resolved.error == ResolveError.NOT_FOUND:
        return AnchorNotFound()
    if resolved.error == ResolveError.AMBIGUOUS:
        return AnchorAmbiguous(tuple(EventInfo.from_event(e) for e in resolved.candidates))
    anchor = resolved.event
    assert anchor is not None

    boundary = anchor.end if position == "after" else anchor.start
    start, end = place_relative_to_boundary(
        boundary=boundary,
        position=position,
        offset_minutes=offset_minutes or 0,
        duration_minutes=duration_minutes,
    )
    return persist_new_event(repos, title=title, start=start, end=end, now=now)
