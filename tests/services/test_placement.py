from daily.services.placement import (
    AnchorDurationMissing,
    AnchorGapConfirmationNeeded,
    AnchorMissing,
    AnchorNeedsDuration,
    AnchorResolved,
    AnchorUnknown,
    place_relative_to_boundary,
    resolve_anchor_placement,
)

from .conftest import DAY, add_event


def resolve(repos, title, **kwargs):
    return resolve_anchor_placement(repos, trigger_title=title, date=DAY, **kwargs)


def test_no_rule_and_no_default_unknown(repos):
    assert resolve(repos, "yoga") == AnchorUnknown()


def test_no_rule_known_default_resolves_from_usual_time_and_duration(repos):
    repos.event_defaults.upsert("yoga", typical_start="19:00", typical_duration_minutes=45)

    result = resolve(repos, "yoga")

    assert result == AnchorResolved(f"{DAY}T19:00", f"{DAY}T19:45")


def test_default_with_time_but_no_duration_asks_duration(repos):
    repos.event_defaults.upsert("yoga", typical_start="19:00")

    assert resolve(repos, "yoga") == AnchorDurationMissing("yoga")


def test_default_without_time_is_unknown(repos):
    repos.event_defaults.upsert("yoga", typical_duration_minutes=45)

    assert resolve(repos, "yoga") == AnchorUnknown()


def test_rule_found_but_anchor_missing_from_schedule_is_not_created(repos):
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )

    result = resolve(repos, "shower")

    assert result == AnchorMissing("shower", "gym", suggested_related_start=None)


def test_missing_anchor_proposes_the_anchors_usual_time_if_known(repos):
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )
    repos.event_defaults.upsert("gym", typical_start="18:00", typical_duration_minutes=60)

    result = resolve(repos, "shower")

    assert isinstance(result, AnchorMissing)
    assert result.suggested_related_start == "18:00"


def test_present_anchor_without_gap_and_known_duration_places_after(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=5
    )
    repos.event_defaults.upsert("shower", typical_duration_minutes=15)

    result = resolve(repos, "shower")

    # 19:00 (gym end) + 5 of offset, then 15 of shower duration
    assert result == AnchorResolved(f"{DAY}T19:05", f"{DAY}T19:20")


def test_position_before_computes_end_from_boundary_and_start_from_duration(repos):
    add_event(repos, "work", f"{DAY}T09:00", f"{DAY}T17:00")
    repos.event_rules.insert_rule(
        trigger_title="breakfast", related_title="work", position="before", offset_minutes=30
    )
    repos.event_defaults.upsert("breakfast", typical_duration_minutes=20)

    result = resolve(repos, "breakfast")

    # end = 09:00 - 30, start = end - 20
    assert result == AnchorResolved(f"{DAY}T08:10", f"{DAY}T08:30")


def test_unknown_trigger_duration_never_uses_a_default(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )

    assert resolve(repos, "shower") == AnchorDurationMissing("shower")


def test_trigger_that_is_itself_a_gap_uses_gap_minutes_as_duration(repos):
    add_event(repos, "work", f"{DAY}T09:00", f"{DAY}T17:00")
    repos.event_rules.insert_rule(
        trigger_title="commute", related_title="work", position="before", gap_minutes=45
    )

    result = resolve(repos, "commute")

    assert result == AnchorResolved(f"{DAY}T08:15", f"{DAY}T09:00")


def test_gap_on_same_anchor_and_position_asks_confirmation_without_writing(repos):
    add_event(repos, "work", f"{DAY}T09:00", f"{DAY}T17:00")
    repos.event_rules.insert_rule(
        trigger_title="commute", related_title="work", position="before", gap_minutes=45
    )
    repos.event_rules.insert_rule(
        trigger_title="dinner", related_title="work", position="before", offset_minutes=0
    )
    repos.event_defaults.upsert("dinner", typical_duration_minutes=60)

    result = resolve(repos, "dinner")

    assert isinstance(result, AnchorGapConfirmationNeeded)
    assert result.trigger_title == "dinner"
    assert result.anchor_title == "work"
    assert result.anchor_boundary == f"{DAY}T09:00"
    # "dinner" is before "work": the boundary is the start of "work"
    assert (result.start_without_gap, result.end_without_gap) == (f"{DAY}T08:00", f"{DAY}T09:00")
    # with the gap the boundary extends 45 minutes before 09:00
    assert (result.start_with_gap, result.end_with_gap) == (f"{DAY}T07:15", f"{DAY}T08:15")
    assert len(result.gaps) == 1
    assert len(repos.schedule.events_for_date(DAY)) == 1  # no write


# --- further cases -------------------------------------------------------------------


def test_explicit_duration_wins_over_usual_one_when_placing_from_usual_time(repos):
    repos.event_defaults.upsert("yoga", typical_start="19:00", typical_duration_minutes=45)

    result = resolve(repos, "yoga", duration_minutes=90)

    assert result == AnchorResolved(f"{DAY}T19:00", f"{DAY}T20:30")


def test_explicit_duration_avoids_asking_duration_for_rule_without_usual_duration(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )

    result = resolve(repos, "shower", duration_minutes=20)

    assert result == AnchorResolved(f"{DAY}T19:00", f"{DAY}T19:20")


def test_explicit_duration_does_not_change_a_gaps_duration(repos):
    add_event(repos, "work", f"{DAY}T09:00", f"{DAY}T17:00")
    repos.event_rules.insert_rule(
        trigger_title="commute", related_title="work", position="before", gap_minutes=45
    )

    result = resolve(repos, "commute", duration_minutes=10)

    assert result == AnchorResolved(f"{DAY}T08:15", f"{DAY}T09:00")


def test_place_relative_to_boundary_after_and_before():
    assert place_relative_to_boundary(
        boundary=f"{DAY}T10:00", position="after", offset_minutes=10, duration_minutes=30
    ) == (f"{DAY}T10:10", f"{DAY}T10:40")
    assert place_relative_to_boundary(
        boundary=f"{DAY}T10:00", position="before", offset_minutes=10, duration_minutes=30
    ) == (f"{DAY}T09:20", f"{DAY}T09:50")


# --- anchor time given by the user (anchor_time) ---------------------------------------


def seed_shower_after_gym(repos, *, gym_duration=60, shower_duration=15):
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )
    if shower_duration:
        repos.event_defaults.upsert("shower", typical_duration_minutes=shower_duration)
    if gym_duration:
        repos.event_defaults.upsert("gym", typical_duration_minutes=gym_duration)


def test_anchor_not_in_schedule_with_user_time_uses_a_synthetic_anchor(repos):
    seed_shower_after_gym(repos)

    result = resolve(repos, "shower", anchor_time="18:00")

    # gym 18:00-19:00 (usual duration), shower right after for 15 minutes
    assert result == AnchorResolved(f"{DAY}T19:00", f"{DAY}T19:15")
    assert repos.schedule.events_for_date(DAY) == []  # the anchor does not enter the schedule


def test_synthetic_anchor_without_anchor_duration_asks_to_add_it(repos):
    seed_shower_after_gym(repos, gym_duration=None)

    result = resolve(repos, "shower", anchor_time="18:00")

    assert result == AnchorNeedsDuration("shower", "gym")


def test_synthetic_anchor_with_position_before_does_not_need_its_duration(repos):
    repos.event_rules.insert_rule(
        trigger_title="breakfast", related_title="work", position="before", offset_minutes=30
    )
    repos.event_defaults.upsert("breakfast", typical_duration_minutes=20)

    result = resolve(repos, "breakfast", anchor_time="09:00")

    assert result == AnchorResolved(f"{DAY}T08:10", f"{DAY}T08:30")


def test_anchor_in_schedule_wins_over_the_given_time(repos):
    seed_shower_after_gym(repos)
    add_event(repos, "gym", f"{DAY}T17:00", f"{DAY}T18:00")

    result = resolve(repos, "shower", anchor_time="21:00")

    assert result == AnchorResolved(f"{DAY}T18:00", f"{DAY}T18:15")


def test_anchor_time_disambiguates_same_name_events(repos):
    seed_shower_after_gym(repos)
    add_event(repos, "gym", f"{DAY}T07:00", f"{DAY}T08:00")
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")

    result = resolve(repos, "shower", anchor_time="18:00")

    assert result == AnchorResolved(f"{DAY}T19:00", f"{DAY}T19:15")


def test_without_anchor_time_the_anchor_stays_missing(repos):
    seed_shower_after_gym(repos)

    assert isinstance(resolve(repos, "shower"), AnchorMissing)
