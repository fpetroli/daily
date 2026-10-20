"""Rules (`event_rules`, `event_defaults`) and what follows from them: downstream proposals
and provenance.

`add_event_rule` and `remove_event_rule` route on `related_title`: when present they write a
relation or a gap in `event_rules`, when absent the usual time in `event_defaults`. Creating
a rule closes the pending patterns that anticipated it (no pending state: the confirmation
is the tool call itself).
"""

import re
from dataclasses import dataclass

from daily.domain.models import ScheduleEvent
from daily.services.context import Repositories
from daily.services.habits import (
    confirm_event_patterns,
    confirm_gap_patterns,
    confirm_sequence_patterns,
)
from daily.services.placement import (
    AnchorGapConfirmationNeeded,
    AnchorResolved,
    resolve_rule_placement,
)
from daily.services.types import RuleCandidateInfo, RuleProposal
from daily.storage.db import transaction

_TIME_OF_DAY = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def compute_rule_proposals(repos: Repositories, *, anchor: ScheduleEvent) -> list[RuleProposal]:
    """The "after" moment: downstream candidates of the event just placed.

    Always proposed, never created on their own; a list, not a single one. A candidate that
    would need a gap confirmation or whose duration is unknown is left out: the
    `{trigger_title, start, end}` shape has no room for those states, and the row stays in
    `event_rules` to come back on a later occasion.
    """
    proposals: list[RuleProposal] = []
    for rule in repos.event_rules.find_by_related(anchor.title):
        placement = resolve_rule_placement(
            repos, rule=rule, trigger_title=rule.trigger_title, anchor=anchor
        )
        if isinstance(placement, AnchorResolved):
            proposals.append(RuleProposal(rule.trigger_title, placement.start, placement.end))
    return proposals


def infer_provenance(
    repos: Repositories, *, title: str, start: str, end: str
) -> tuple[str | None, str | None]:
    """`(derived_from_event_id, derived_from_rule_id)` of an event about to be created.

    With no pending state nobody hands the provenance to the confirmation of a proposal
    (provenance is recorded when the event is born *through the rule*): the service infers
    it. An event inherits event and rule if `start`/`end` match what a rule for its title
    would give relative to an anchor present on the same day; for a gap, they match one of
    the two candidates (with or without the break).
    """
    date = start.split("T")[0]
    for rule in repos.event_rules.find_by_trigger(title):
        anchor = repos.resolver.resolve(description=rule.related_title, date=date).event
        if anchor is None:
            continue
        placement = resolve_rule_placement(repos, rule=rule, trigger_title=title, anchor=anchor)
        if isinstance(placement, AnchorResolved):
            candidates = [(placement.start, placement.end)]
        elif isinstance(placement, AnchorGapConfirmationNeeded):
            candidates = [
                (placement.start_with_gap, placement.end_with_gap),
                (placement.start_without_gap, placement.end_without_gap),
            ]
        else:
            continue
        if (start, end) in candidates:
            return anchor.event_id, rule.rule_id
    return None, None


@dataclass(frozen=True, slots=True)
class AddEventRuleSuccess:
    pass


@dataclass(frozen=True, slots=True)
class AddEventRuleNeedsPosition:
    """`related_title` present but `position` missing: before and after are not interchangeable."""

    trigger_title: str
    related_title: str
    offset_minutes: int | None = None
    gap_minutes: int | None = None


@dataclass(frozen=True, slots=True)
class AddEventRuleInvalidFormat:
    """`typical_start` cannot be normalized to `HH:MM`: never written as it is.

    A malformed value would be concatenated to a date and produce an unparsable string at
    the first creation that resolves that time, not right away.
    """


@dataclass(frozen=True, slots=True)
class AddEventRuleNeedsInfo:
    """With no `related_title`, no time and no duration there is nothing to store."""

    trigger_title: str


AddEventRuleOutcome = (
    AddEventRuleSuccess
    | AddEventRuleNeedsPosition
    | AddEventRuleInvalidFormat
    | AddEventRuleNeedsInfo
)


def _extract_time_of_day(value: str) -> str:
    """The model sometimes gives an ISO datetime for `typical_start`: keep the part after `T`."""
    return value.rsplit("T", 1)[-1]


def add_event_rule(
    repos: Repositories,
    *,
    trigger_title: str,
    related_title: str | None = None,
    position: str | None = None,
    offset_minutes: int | None = None,
    gap_minutes: int | None = None,
    typical_start: str | None = None,
    typical_duration_minutes: int | None = None,
) -> AddEventRuleOutcome:
    """Stores a rule or a usual time.

    Relational branch: if a row already exists for the trigger/related pair it is updated
    instead of duplicated (two rows would give double proposals).
    """
    if related_title is None:
        return _add_default(
            repos,
            title=trigger_title,
            typical_start=typical_start,
            typical_duration_minutes=typical_duration_minutes,
        )

    if position is None:
        return AddEventRuleNeedsPosition(trigger_title, related_title, offset_minutes, gap_minutes)

    with transaction(repos.conn):
        existing = repos.event_rules.find_matching(trigger_title, related_title)
        if not existing:
            repos.event_rules.insert_rule(
                trigger_title=trigger_title,
                related_title=related_title,
                position=position,
                offset_minutes=offset_minutes,
                gap_minutes=gap_minutes,
            )
        else:
            repos.event_rules.update_rule(
                existing[0].rule_id,
                position=position,
                offset_minutes=offset_minutes,
                gap_minutes=gap_minutes,
            )
        if gap_minutes is not None:
            confirm_gap_patterns(
                repos, related_title=related_title, position=position, gap_minutes=gap_minutes
            )
        else:
            confirm_sequence_patterns(
                repos, trigger_title=trigger_title, related_title=related_title
            )
    return AddEventRuleSuccess()


def _add_default(
    repos: Repositories,
    *,
    title: str,
    typical_start: str | None,
    typical_duration_minutes: int | None,
) -> AddEventRuleOutcome:
    if typical_start is None and typical_duration_minutes is None:
        return AddEventRuleNeedsInfo(title)

    normalized_start: str | None = None
    if typical_start is not None:
        normalized_start = _extract_time_of_day(typical_start)
        if not _TIME_OF_DAY.match(normalized_start):
            return AddEventRuleInvalidFormat()

    with transaction(repos.conn):
        repos.event_defaults.upsert(
            title, typical_start=normalized_start, typical_duration_minutes=typical_duration_minutes
        )
        saved = repos.event_defaults.get_by_title(title)
        if (
            saved is not None
            and saved.typical_start is not None
            and saved.typical_duration_minutes is not None
        ):
            confirm_event_patterns(repos, title=title)
    return AddEventRuleSuccess()


@dataclass(frozen=True, slots=True)
class RemoveEventRuleSuccess:
    pass


@dataclass(frozen=True, slots=True)
class RemoveEventRuleNotFound:
    pass


@dataclass(frozen=True, slots=True)
class RemoveEventRuleAmbiguous:
    candidates: tuple[RuleCandidateInfo, ...]


RemoveEventRuleOutcome = RemoveEventRuleSuccess | RemoveEventRuleNotFound | RemoveEventRuleAmbiguous


def remove_event_rule(
    repos: Repositories,
    *,
    trigger_title: str,
    related_title: str | None = None,
    field: str | None = None,
) -> RemoveEventRuleOutcome:
    """Deletes a rule or a usual time.

    Without `related_title` it acts on `event_defaults`, never with
    a fuzzy match on `event_rules` by title alone; `field` (only there) clears that field
    instead of deleting the row.
    """
    if related_title is None:
        if repos.event_defaults.get_by_title(trigger_title) is None:
            return RemoveEventRuleNotFound()
        if field is not None:
            repos.event_defaults.clear_field(trigger_title, field)
        else:
            repos.event_defaults.delete_row(trigger_title)
        return RemoveEventRuleSuccess()

    matches = repos.event_rules.find_matching(trigger_title, related_title)
    if not matches:
        return RemoveEventRuleNotFound()
    if len(matches) > 1:
        return RemoveEventRuleAmbiguous(
            tuple(RuleCandidateInfo(r.rule_id, r.trigger_title, r.related_title) for r in matches)
        )
    repos.event_rules.delete_rule(matches[0].rule_id)
    return RemoveEventRuleSuccess()
