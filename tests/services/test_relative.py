from daily.services.relative import (
    AnchorAmbiguous,
    AnchorNotFound,
    RelativeNeedsInfo,
    create_event_relative,
)
from daily.services.schedule import EventCreated

from .conftest import DAY, NOW, add_event


def create(repos, title, **kwargs):
    kwargs.setdefault("date", DAY)
    kwargs.setdefault("now", NOW)
    kwargs.setdefault("anchor_description", None)
    kwargs.setdefault("position", None)
    kwargs.setdefault("duration_minutes", None)
    return create_event_relative(repos, title=title, **kwargs)


def test_records_a_repeated_event_never_with_rule_provenance(repos):
    add_event(repos, "shower", f"{DAY}T19:00", f"{DAY}T19:15")

    create(
        repos,
        "breakfast",
        anchor_description="shower",
        position="after",
        offset_minutes=5,
        duration_minutes=20,
    )

    [pattern] = repos.event_patterns.list_pending()
    assert (pattern.title, pattern.typical_start, pattern.typical_duration_minutes) == (
        "breakfast",
        "19:20",
        20,
    )


def test_places_after_the_anchor_with_explicit_offset(repos):
    add_event(repos, "shower", f"{DAY}T19:00", f"{DAY}T19:15")

    outcome = create(
        repos,
        "breakfast",
        anchor_description="shower",
        position="after",
        offset_minutes=5,
        duration_minutes=20,
    )

    assert isinstance(outcome, EventCreated)
    saved = repos.schedule.get_by_id(outcome.event_id)
    assert (saved.start, saved.end) == (f"{DAY}T19:20", f"{DAY}T19:40")


def test_missing_offset_is_zero(repos):
    add_event(repos, "shower", f"{DAY}T19:00", f"{DAY}T19:15")

    outcome = create(
        repos, "breakfast", anchor_description="shower", position="after", duration_minutes=20
    )

    assert outcome.start == f"{DAY}T19:15"


def test_position_before_computes_end_from_boundary_and_start_from_duration(repos):
    add_event(repos, "work", f"{DAY}T09:00", f"{DAY}T17:00")

    outcome = create(
        repos,
        "breakfast",
        anchor_description="work",
        position="before",
        offset_minutes=30,
        duration_minutes=20,
    )

    saved = repos.schedule.get_by_id(outcome.event_id)
    assert (saved.start, saved.end) == (f"{DAY}T08:10", f"{DAY}T08:30")


def test_anchor_not_found(repos):
    outcome = create(
        repos, "breakfast", anchor_description="shower", position="after", duration_minutes=20
    )

    assert outcome == AnchorNotFound()


def test_ambiguous_anchor_with_candidates(repos):
    add_event(repos, "call with Marco", f"{DAY}T10:00", f"{DAY}T10:30")
    add_event(repos, "call with the client", f"{DAY}T15:00", f"{DAY}T15:30")

    outcome = create(
        repos, "follow-up", anchor_description="call", position="after", duration_minutes=15
    )

    assert isinstance(outcome, AnchorAmbiguous)
    assert len(outcome.candidates) == 2


def test_missing_duration_asks_without_writing(repos):
    add_event(repos, "shower", f"{DAY}T19:00", f"{DAY}T19:15")

    outcome = create(repos, "breakfast", anchor_description="shower", position="after")

    assert outcome == RelativeNeedsInfo()
    assert len(repos.schedule.events_for_date(DAY)) == 1


def test_missing_duration_is_the_usual_one(repos):
    add_event(repos, "shower", f"{DAY}T19:00", f"{DAY}T19:15")
    repos.event_defaults.upsert("Breakfast", typical_duration_minutes=20)

    outcome = create(repos, "breakfast", anchor_description="shower", position="after")

    assert isinstance(outcome, EventCreated)
    assert (outcome.start, outcome.end) == (f"{DAY}T19:15", f"{DAY}T19:35")


def test_missing_position_asks(repos):
    add_event(repos, "shower", f"{DAY}T19:00", f"{DAY}T19:15")

    outcome = create(repos, "breakfast", anchor_description="shower", duration_minutes=20)

    assert outcome == RelativeNeedsInfo()


def test_missing_anchor_asks(repos):
    assert create(repos, "breakfast", position="after", duration_minutes=20) == RelativeNeedsInfo()


def test_downstream_proposals_after_insertion(repos):
    add_event(repos, "shower", f"{DAY}T19:00", f"{DAY}T19:15")
    repos.event_rules.insert_rule(
        trigger_title="getting dressed",
        related_title="breakfast",
        position="after",
        offset_minutes=0,
    )
    repos.event_defaults.upsert("getting dressed", typical_duration_minutes=10)

    outcome = create(
        repos, "breakfast", anchor_description="shower", position="after", duration_minutes=20
    )

    assert [p.trigger_title for p in outcome.rule_proposals] == ["getting dressed"]


def test_reports_conflicts_without_cancelling(repos):
    add_event(repos, "shower", f"{DAY}T19:00", f"{DAY}T19:15")
    add_event(repos, "dinner", f"{DAY}T19:20", f"{DAY}T20:00")

    outcome = create(
        repos,
        "breakfast",
        anchor_description="shower",
        position="after",
        offset_minutes=5,
        duration_minutes=20,
    )

    assert [c.title for c in outcome.conflicts] == ["dinner"]
    assert repos.schedule.get_by_id(outcome.event_id) is not None


def test_non_positive_duration_is_an_invalid_range(repos):
    from daily.services.schedule import InvalidRange

    add_event(repos, "shower", f"{DAY}T19:00", f"{DAY}T19:15")

    outcome = create(
        repos, "breakfast", anchor_description="shower", position="after", duration_minutes=0
    )

    assert outcome == InvalidRange()
    assert len(repos.schedule.events_for_date(DAY)) == 1


# --- further cases -------------------------------------------------------------------


def test_anchor_time_disambiguates_same_name_events(repos):
    add_event(repos, "call with Marco", f"{DAY}T10:00", f"{DAY}T10:30")
    add_event(repos, "call with Marco", f"{DAY}T15:00", f"{DAY}T15:30")

    outcome = create(
        repos,
        "follow-up",
        anchor_description="call with Marco",
        anchor_time="15:00",
        position="after",
        duration_minutes=15,
    )

    assert outcome.start == f"{DAY}T15:30"
