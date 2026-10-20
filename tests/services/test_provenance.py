"""Inferred provenance: with no pending state, nobody hands it to the confirmation."""

from daily.services.relative import create_event_relative
from daily.services.rules import infer_provenance
from daily.services.schedule import create_event, update_event

from .conftest import DAY, NOW, add_event


def setup_shower_after_gym(repos, *, offset=0):
    gym = add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    rule_id = repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=offset
    )
    repos.event_defaults.upsert("shower", typical_duration_minutes=15)
    return gym, rule_id


def test_event_matching_the_rule_inherits_event_and_rule(repos):
    gym, rule_id = setup_shower_after_gym(repos)

    outcome = create_event(
        repos, title="shower", date=DAY, start=f"{DAY}T19:00", end=f"{DAY}T19:15", now=NOW
    )

    saved = repos.schedule.get_by_id(outcome.event_id)
    assert (saved.derived_from_event_id, saved.derived_from_rule_id) == (
        gym.event_id,
        rule_id,
    )


def test_event_resolved_by_the_rule_inherits_provenance(repos):
    gym, rule_id = setup_shower_after_gym(repos)

    outcome = create_event(repos, title="shower", date=DAY, now=NOW)

    saved = repos.schedule.get_by_id(outcome.event_id)
    assert (saved.derived_from_event_id, saved.derived_from_rule_id) == (
        gym.event_id,
        rule_id,
    )


def test_event_not_matching_the_rule_inherits_nothing(repos):
    setup_shower_after_gym(repos)

    outcome = create_event(
        repos, title="shower", date=DAY, start=f"{DAY}T21:00", end=f"{DAY}T21:15", now=NOW
    )

    saved = repos.schedule.get_by_id(outcome.event_id)
    assert (saved.derived_from_event_id, saved.derived_from_rule_id) == (None, None)


def test_without_the_anchor_in_schedule_inherits_nothing(repos):
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )
    repos.event_defaults.upsert("shower", typical_duration_minutes=15)

    outcome = create_event(
        repos, title="shower", date=DAY, start=f"{DAY}T19:00", end=f"{DAY}T19:15", now=NOW
    )

    assert repos.schedule.get_by_id(outcome.event_id).derived_from_rule_id is None


def test_without_usual_duration_the_rule_cannot_match(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )

    assert infer_provenance(repos, title="shower", start=f"{DAY}T19:00", end=f"{DAY}T19:15") == (
        None,
        None,
    )


def test_gap_matches_one_of_the_two_candidates(repos):
    work = add_event(repos, "work", f"{DAY}T09:00", f"{DAY}T17:00")
    repos.event_rules.insert_rule(
        trigger_title="commute", related_title="work", position="before", gap_minutes=45
    )
    dinner_rule = repos.event_rules.insert_rule(
        trigger_title="dinner", related_title="work", position="before", offset_minutes=0
    )
    repos.event_defaults.upsert("dinner", typical_duration_minutes=60)

    with_gap = infer_provenance(repos, title="dinner", start=f"{DAY}T07:15", end=f"{DAY}T08:15")
    without_gap = infer_provenance(repos, title="dinner", start=f"{DAY}T08:00", end=f"{DAY}T09:00")

    assert with_gap == (work.event_id, dinner_rule)
    assert without_gap == (work.event_id, dinner_rule)


def test_explicit_provenance_is_not_overwritten_by_inference(repos):
    setup_shower_after_gym(repos)
    other = add_event(repos, "other", f"{DAY}T08:00", f"{DAY}T09:00")
    other_rule = repos.event_rules.insert_rule(
        trigger_title="shower", related_title="other", position="after", offset_minutes=0
    )

    outcome = create_event(
        repos,
        title="shower",
        date=DAY,
        start=f"{DAY}T19:00",
        end=f"{DAY}T19:15",
        derived_from_event_id=other.event_id,
        derived_from_rule_id=other_rule,
        now=NOW,
    )

    saved = repos.schedule.get_by_id(outcome.event_id)
    assert (saved.derived_from_event_id, saved.derived_from_rule_id) == (
        other.event_id,
        other_rule,
    )


def test_relative_creation_infers_provenance_too(repos):
    gym, rule_id = setup_shower_after_gym(repos)

    outcome = create_event_relative(
        repos,
        title="shower",
        date=DAY,
        anchor_description="gym",
        position="after",
        duration_minutes=15,
        now=NOW,
    )

    saved = repos.schedule.get_by_id(outcome.event_id)
    assert (saved.derived_from_event_id, saved.derived_from_rule_id) == (
        gym.event_id,
        rule_id,
    )


def test_inferred_provenance_makes_the_derived_event_a_cascade_candidate(repos):
    setup_shower_after_gym(repos)
    create_event(repos, title="shower", date=DAY, now=NOW)

    outcome = update_event(repos, event_description="gym", default_date=DAY, shift_minutes=30)

    assert [c.title for c in outcome.cascade_candidates] == ["shower"]
