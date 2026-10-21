import pytest

from daily.services import schedule
from daily.services.placement import (
    AnchorDurationMissing,
    AnchorGapConfirmationNeeded,
    AnchorMissing,
)
from daily.services.schedule import (
    EventAlreadyThere,
    EventCreated,
    InsufficientInfo,
    InvalidRange,
    NeedsAnchorConfirmation,
    NeedsAnchorDuration,
    NeedsDurationConfirmation,
    NeedsGapConfirmation,
    create_event,
)

from .conftest import DAY, NOW, add_event


def create(repos, title, **kwargs):
    kwargs.setdefault("date", DAY)
    kwargs.setdefault("now", NOW)
    return create_event(repos, title=title, **kwargs)


def rule(repos, trigger, related, position, **kwargs):
    return repos.event_rules.insert_rule(
        trigger_title=trigger, related_title=related, position=position, **kwargs
    )


def test_explicit_derived_from_are_saved_in_the_schedule(repos):
    parent = add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    rule_id = rule(repos, "shower", "gym", "after", offset_minutes=0)

    outcome = create(
        repos,
        "shower",
        start=f"{DAY}T19:00",
        end=f"{DAY}T19:15",
        derived_from_event_id=parent.event_id,
        derived_from_rule_id=rule_id,
    )

    saved = repos.schedule.get_by_id(outcome.event_id)
    assert saved.derived_from_event_id == parent.event_id
    assert saved.derived_from_rule_id == rule_id


def test_without_provenance_records_a_repeated_event(repos):
    create(repos, "yoga", start=f"{DAY}T19:00", end=f"{DAY}T19:45")

    [pattern] = repos.event_patterns.list_pending()
    assert (pattern.title, pattern.typical_start, pattern.typical_duration_minutes) == (
        "yoga",
        "19:00",
        45,
    )
    assert pattern.occurrence_count == 1


def test_with_rule_provenance_does_not_record_a_repeated_event(repos):
    parent = add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    rule_id = rule(repos, "shower", "gym", "after", offset_minutes=0)

    create(
        repos,
        "shower",
        start=f"{DAY}T19:00",
        end=f"{DAY}T19:15",
        derived_from_event_id=parent.event_id,
        derived_from_rule_id=rule_id,
    )

    assert repos.event_patterns.list_pending() == []


def test_explicit_start_and_end_inserts_without_conflicts(repos):
    outcome = create(repos, "gym", start=f"{DAY}T18:00", end=f"{DAY}T19:00")

    assert isinstance(outcome, EventCreated)
    assert outcome.conflicts == ()
    assert outcome.rule_proposals == ()
    assert repos.schedule.get_by_id(outcome.event_id).title == "gym"


def test_invalid_range_writes_nothing(repos):
    outcome = create(repos, "x", start=f"{DAY}T18:00", end=f"{DAY}T17:00")

    assert outcome == InvalidRange()
    assert repos.schedule.events_for_date(DAY) == []


def test_end_equal_to_start_is_invalid(repos):
    assert create(repos, "x", start=f"{DAY}T18:00", end=f"{DAY}T18:00") == InvalidRange()


def test_detects_conflicts_without_cancelling_the_insertion(repos):
    add_event(repos, "meeting", f"{DAY}T10:00", f"{DAY}T11:00")

    outcome = create(repos, "call", start=f"{DAY}T10:30", end=f"{DAY}T11:30")

    assert [c.title for c in outcome.conflicts] == ["meeting"]
    assert repos.schedule.get_by_id(outcome.event_id) is not None


def test_an_exact_copy_is_not_added_again(repos):
    add_event(repos, "Gym", f"{DAY}T18:00", f"{DAY}T19:00")

    copy = create(repos, "gym", start=f"{DAY}T18:00", end=f"{DAY}T19:00")
    later = create(repos, "gym", start=f"{DAY}T18:30", end=f"{DAY}T19:30")

    assert copy == EventAlreadyThere("Gym", f"{DAY}T18:00", f"{DAY}T19:00")
    assert isinstance(later, EventCreated)  # another time is another event
    assert len(repos.schedule.events_for_date(DAY)) == 2


def test_null_start_resolved_by_rule(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    rule(repos, "shower", "gym", "after", offset_minutes=0)
    repos.event_defaults.upsert("shower", typical_duration_minutes=15)

    outcome = create(repos, "shower")

    assert isinstance(outcome, EventCreated)
    saved = repos.schedule.get_by_id(outcome.event_id)
    assert (saved.start, saved.end) == (f"{DAY}T19:00", f"{DAY}T19:15")


def test_null_start_with_missing_anchor_asks_for_anchor_without_writing(repos):
    rule(repos, "shower", "gym", "after", offset_minutes=0)

    outcome = create(repos, "shower")

    assert isinstance(outcome, NeedsAnchorConfirmation)
    assert isinstance(outcome.missing, AnchorMissing)
    assert repos.schedule.events_for_date(DAY) == []


def test_null_start_with_missing_duration_asks_duration_without_writing(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    rule(repos, "shower", "gym", "after", offset_minutes=0)

    outcome = create(repos, "shower")

    assert isinstance(outcome, NeedsDurationConfirmation)
    assert isinstance(outcome.missing, AnchorDurationMissing)
    assert len(repos.schedule.events_for_date(DAY)) == 1  # only "palestra"


def seed_gap_scenario(repos):
    add_event(repos, "work", f"{DAY}T09:00", f"{DAY}T17:00")
    rule(repos, "commute", "work", "before", gap_minutes=45)
    rule(repos, "dinner", "work", "before", offset_minutes=0)
    repos.event_defaults.upsert("dinner", typical_duration_minutes=60)


def test_null_start_with_gap_asks_confirmation_without_writing(repos):
    seed_gap_scenario(repos)

    outcome = create(repos, "dinner")

    assert isinstance(outcome, NeedsGapConfirmation)
    assert isinstance(outcome.needs_confirmation, AnchorGapConfirmationNeeded)
    assert len(repos.schedule.events_for_date(DAY)) == 1  # only "work"


def test_null_start_with_nothing_to_resolve_asks_the_time(repos):
    assert create(repos, "who knows") == InsufficientInfo(missing="start")


def test_start_without_end_duration_or_usual_duration_asks_duration(repos):
    assert create(repos, "x", start=f"{DAY}T10:00") == InsufficientInfo(missing="duration")
    assert repos.schedule.events_for_date(DAY) == []


def test_downstream_proposals_after_placing_the_anchor_are_a_list(repos):
    rule(repos, "shower", "gym", "after", offset_minutes=0)
    rule(repos, "stretching", "gym", "before", offset_minutes=10)
    repos.event_defaults.upsert("shower", typical_duration_minutes=15)
    repos.event_defaults.upsert("stretching", typical_duration_minutes=10)

    outcome = create(repos, "gym", start=f"{DAY}T18:00", end=f"{DAY}T19:00")

    assert {p.trigger_title for p in outcome.rule_proposals} == {"shower", "stretching"}
    by_title = {p.trigger_title: p for p in outcome.rule_proposals}
    assert (by_title["shower"].start, by_title["shower"].end) == (f"{DAY}T19:00", f"{DAY}T19:15")
    assert (by_title["stretching"].start, by_title["stretching"].end) == (
        f"{DAY}T17:40",
        f"{DAY}T17:50",
    )


def test_a_proposal_needing_gap_or_duration_is_omitted(repos):
    rule(repos, "shower", "gym", "after", offset_minutes=0)  # unknown duration

    outcome = create(repos, "gym", start=f"{DAY}T18:00", end=f"{DAY}T19:00")

    assert outcome.rule_proposals == ()


# --- further cases -------------------------------------------------------------------


def test_explicit_duration_computes_the_end(repos):
    outcome = create(repos, "gym", start=f"{DAY}T18:00", duration_minutes=60)

    assert outcome.end == f"{DAY}T19:00"


def test_explicit_duration_wins_over_an_explicit_end(repos):
    outcome = create(repos, "gym", start=f"{DAY}T18:00", end=f"{DAY}T18:10", duration_minutes=90)

    assert outcome.end == f"{DAY}T19:30"


def test_without_end_uses_the_titles_usual_duration(repos):
    repos.event_defaults.upsert("yoga", typical_duration_minutes=45)

    outcome = create(repos, "yoga", start=f"{DAY}T19:00")

    assert isinstance(outcome, EventCreated)
    assert outcome.end == f"{DAY}T19:45"


def test_explicit_duration_avoids_asking_a_rule_for_duration(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    rule(repos, "shower", "gym", "after", offset_minutes=0)

    outcome = create(repos, "shower", duration_minutes=20)

    assert isinstance(outcome, EventCreated)
    assert (outcome.start, outcome.end) == (f"{DAY}T19:00", f"{DAY}T19:20")


def test_gap_with_break(repos):
    seed_gap_scenario(repos)

    outcome = create(repos, "dinner", with_gap=True)

    assert (outcome.start, outcome.end) == (f"{DAY}T07:15", f"{DAY}T08:15")


def test_gap_without_break(repos):
    seed_gap_scenario(repos)

    outcome = create(repos, "dinner", with_gap=False)

    assert (outcome.start, outcome.end) == (f"{DAY}T08:00", f"{DAY}T09:00")


def test_with_gap_without_a_gap_involved_has_no_effect(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    rule(repos, "shower", "gym", "after", offset_minutes=0)
    repos.event_defaults.upsert("shower", typical_duration_minutes=15)

    outcome = create(repos, "shower", with_gap=True)

    assert (outcome.start, outcome.end) == (f"{DAY}T19:00", f"{DAY}T19:15")


def test_an_error_after_insertion_also_rolls_back_the_insertion(repos, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("tracking broken")

    monkeypatch.setattr(schedule, "record_habit_signals", boom)

    with pytest.raises(RuntimeError):
        create(repos, "gym", start=f"{DAY}T18:00", end=f"{DAY}T19:00")

    assert repos.schedule.events_for_date(DAY) == []


def test_anchor_time_not_in_schedule_places_event_without_creating_anchor(repos):
    rule(repos, "shower", "gym", "after", offset_minutes=0)
    repos.event_defaults.upsert("shower", typical_duration_minutes=15)
    repos.event_defaults.upsert("gym", typical_duration_minutes=60)

    outcome = create(repos, "shower", anchor_time="18:00")

    assert isinstance(outcome, EventCreated)
    assert (outcome.start, outcome.end) == (f"{DAY}T19:00", f"{DAY}T19:15")
    assert [e.title for e in repos.schedule.events_for_date(DAY)] == ["shower"]
    assert repos.schedule.get_by_id(outcome.event_id).derived_from_rule_id is None


def test_anchor_time_without_anchor_duration_asks_to_add_it(repos):
    rule(repos, "shower", "gym", "after", offset_minutes=0)
    repos.event_defaults.upsert("shower", typical_duration_minutes=15)

    outcome = create(repos, "shower", anchor_time="18:00")

    assert isinstance(outcome, NeedsAnchorDuration)
    assert (outcome.needs.trigger_title, outcome.needs.anchor_title) == ("shower", "gym")
    assert repos.schedule.events_for_date(DAY) == []
