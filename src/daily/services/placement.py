"""Where to place an event with no explicit time: rules, anchors, gaps.

`resolve_anchor_placement` is the "before" moment (the event is about to be written);
`resolve_rule_placement` is the core shared with the "after" moment (downstream
candidates: the anchor is the event just placed). Neither of them writes.
"""

from dataclasses import dataclass

from daily.domain.models import EventRule, Position, ScheduleEvent
from daily.domain.timeutil import add_minutes_to_iso, combine_date_and_time
from daily.services.context import Repositories


@dataclass(frozen=True, slots=True)
class AnchorResolved:
    """Resolved with no confirmation needed: no gap involved, known duration."""

    start: str
    end: str


@dataclass(frozen=True, slots=True)
class AnchorGapConfirmationNeeded:
    """One or more gap rows on the same anchor and position are involved.

    Nothing is written: the user is asked, with both candidates already computed.
    `anchor_boundary` is the raw boundary (before the gap): the anchor's end if `after`,
    its start if `before`.
    """

    trigger_title: str
    anchor_title: str
    anchor_boundary: str
    start_with_gap: str
    end_with_gap: str
    start_without_gap: str
    end_without_gap: str
    gaps: tuple[EventRule, ...]


@dataclass(frozen=True, slots=True)
class AnchorMissing:
    """The rule's anchor is not in the schedule that day: never created automatically.

    `suggested_related_start` (`HH:MM`) comes from the anchor's `event_defaults`, if known.
    """

    trigger_title: str
    related_title: str
    suggested_related_start: str | None = None


@dataclass(frozen=True, slots=True)
class AnchorDurationMissing:
    """Unknown event duration: never a default, the user is always asked."""

    trigger_title: str


@dataclass(frozen=True, slots=True)
class AnchorNeedsDuration:
    """The anchor is not in the schedule and its duration, needed to place "after", is unknown.

    It happens with an anchor time given by the user (`anchor_time`): the anchor's end cannot
    be computed. The user is asked to add it to the schedule with a start and an end.
    """

    trigger_title: str
    anchor_title: str


@dataclass(frozen=True, slots=True)
class AnchorUnknown:
    """No rule and no usual time for the title: nothing to resolve."""


AnchorResolution = (
    AnchorResolved
    | AnchorGapConfirmationNeeded
    | AnchorMissing
    | AnchorDurationMissing
    | AnchorNeedsDuration
    | AnchorUnknown
)


def place_relative_to_boundary(
    *, boundary: str, position: Position | str, offset_minutes: int, duration_minutes: int
) -> tuple[str, str]:
    """`after`: start = boundary + offset. `before`: end = boundary - offset (exact formula)."""
    if position == "after":
        start = add_minutes_to_iso(boundary, offset_minutes)
        return start, add_minutes_to_iso(start, duration_minutes)
    end = add_minutes_to_iso(boundary, -offset_minutes)
    return add_minutes_to_iso(end, -duration_minutes), end


def resolve_anchor_placement(
    repos: Repositories,
    *,
    trigger_title: str,
    date: str,
    duration_minutes: int | None = None,
    anchor_time: str | None = None,
) -> AnchorResolution:
    """Places `trigger_title` on day `date` with no explicit time.

    `duration_minutes` is the duration said by the user in the same sentence: when present
    it wins over the usual one and avoids asking for it. With several rules for the same
    dependent event the first one found is used (unspecified case).

    `anchor_time` (`HH:MM`) is the anchor time said by the user. It disambiguates events
    with the same name and, if the anchor is not in the schedule, replaces it with a
    synthetic event that is never written, so the event is placed without the anchor
    entering the schedule ("yes, usually at 6 pm").
    """
    rules = repos.event_rules.find_by_trigger(trigger_title)

    if not rules:
        standalone = repos.event_defaults.get_by_title(trigger_title)
        if standalone is None or standalone.typical_start is None:
            return AnchorUnknown()
        duration = duration_minutes or standalone.typical_duration_minutes
        if duration is None:
            return AnchorDurationMissing(trigger_title)
        start = combine_date_and_time(date, standalone.typical_start)
        return AnchorResolved(start, add_minutes_to_iso(start, duration))

    rule = rules[0]
    found = repos.resolver.resolve(description=rule.related_title, date=date, at_time=anchor_time)
    anchor = found.event
    if anchor is None:
        if anchor_time is not None:
            anchor = _synthetic_anchor(repos, rule=rule, date=date, anchor_time=anchor_time)
            if anchor is None:
                return AnchorNeedsDuration(trigger_title, rule.related_title)
        else:
            suggestion = repos.event_defaults.get_by_title(rule.related_title)
            return AnchorMissing(
                trigger_title=trigger_title,
                related_title=rule.related_title,
                suggested_related_start=suggestion.typical_start if suggestion else None,
            )

    return resolve_rule_placement(
        repos,
        rule=rule,
        trigger_title=trigger_title,
        anchor=anchor,
        duration_minutes=duration_minutes,
    )


def _synthetic_anchor(
    repos: Repositories, *, rule: EventRule, date: str, anchor_time: str
) -> ScheduleEvent | None:
    """The rule's anchor at the given time: `None` if its end is needed and unknown.

    "After" starts from the anchor's end, so its usual duration is needed; "before" starts
    from its start and needs nothing else.
    """
    start = combine_date_and_time(date, anchor_time)
    defaults = repos.event_defaults.get_by_title(rule.related_title)
    duration = defaults.typical_duration_minutes if defaults else None
    if duration is None and rule.position == "after":
        return None
    end = add_minutes_to_iso(start, duration or 0)
    return ScheduleEvent(event_id="", title=rule.related_title, start=start, end=end)


def resolve_rule_placement(
    repos: Repositories,
    *,
    rule: EventRule,
    trigger_title: str,
    anchor: ScheduleEvent,
    duration_minutes: int | None = None,
) -> AnchorResolution:
    """Places `trigger_title` relative to an already resolved anchor, according to `rule`."""
    offset = rule.offset_minutes or 0
    raw_boundary = anchor.end if rule.position == "after" else anchor.start

    if rule.gap_minutes is not None:
        # The event is the gap itself: `gap_minutes` is also its duration, with no
        # auto-extension and no read of event_defaults.
        start, end = place_relative_to_boundary(
            boundary=raw_boundary,
            position=rule.position,
            offset_minutes=offset,
            duration_minutes=rule.gap_minutes,
        )
        return AnchorResolved(start, end)

    duration = duration_minutes
    if duration is None:
        defaults = repos.event_defaults.get_by_title(trigger_title)
        duration = defaults.typical_duration_minutes if defaults else None
    if duration is None:
        return AnchorDurationMissing(trigger_title)

    start_without_gap, end_without_gap = place_relative_to_boundary(
        boundary=raw_boundary,
        position=rule.position,
        offset_minutes=offset,
        duration_minutes=duration,
    )

    gaps = repos.event_rules.find_gaps_for(rule.related_title, rule.position)
    if not gaps:
        return AnchorResolved(start_without_gap, end_without_gap)

    gap_minutes = gaps[0].gap_minutes or 0
    extended_boundary = add_minutes_to_iso(
        raw_boundary, gap_minutes if rule.position == "after" else -gap_minutes
    )
    start_with_gap, end_with_gap = place_relative_to_boundary(
        boundary=extended_boundary,
        position=rule.position,
        offset_minutes=offset,
        duration_minutes=duration,
    )
    return AnchorGapConfirmationNeeded(
        trigger_title=trigger_title,
        anchor_title=rule.related_title,
        anchor_boundary=raw_boundary,
        start_with_gap=start_with_gap,
        end_with_gap=end_with_gap,
        start_without_gap=start_without_gap,
        end_without_gap=end_without_gap,
        gaps=tuple(gaps),
    )
