from daily.services.schedule import (
    EventAmbiguous,
    EventNotFound,
    EventUpdated,
    InvalidRange,
    update_event,
)

from .conftest import DAY, add_event


def update(repos, description, **kwargs):
    kwargs.setdefault("event_date", DAY)
    return update_event(repos, event_description=description, default_date=DAY, **kwargs)


def lunch(repos):
    return add_event(repos, "lunch", f"{DAY}T13:00", f"{DAY}T14:00")


def saved(repos, event):
    return repos.schedule.get_by_id(event.event_id)


def test_new_start_alone_keeps_the_original_duration(repos):
    event = lunch(repos)

    outcome = update(repos, "lunch", new_start=f"{DAY}T13:30")

    assert isinstance(outcome, EventUpdated)
    assert (saved(repos, event).start, saved(repos, event).end) == (f"{DAY}T13:30", f"{DAY}T14:30")


def test_new_end_alone_leaves_the_start_unchanged(repos):
    event = lunch(repos)

    update(repos, "lunch", new_end=f"{DAY}T15:00")

    assert (saved(repos, event).start, saved(repos, event).end) == (f"{DAY}T13:00", f"{DAY}T15:00")


def test_explicit_start_and_end_are_used_literally(repos):
    event = lunch(repos)

    update(repos, "lunch", new_start=f"{DAY}T12:00", new_end=f"{DAY}T12:45")

    assert (saved(repos, event).start, saved(repos, event).end) == (f"{DAY}T12:00", f"{DAY}T12:45")


def test_duration_computes_the_end_from_the_existing_start(repos):
    event = lunch(repos)

    update(repos, "lunch", new_duration_minutes=30)

    assert (saved(repos, event).start, saved(repos, event).end) == (f"{DAY}T13:00", f"{DAY}T13:30")


def test_duration_with_new_start_computes_the_end_from_the_new_start(repos):
    event = lunch(repos)

    update(repos, "lunch", new_start=f"{DAY}T13:30", new_duration_minutes=20)

    assert (saved(repos, event).start, saved(repos, event).end) == (f"{DAY}T13:30", f"{DAY}T13:50")


def test_duration_wins_over_an_explicit_end(repos):
    event = lunch(repos)

    update(repos, "lunch", new_end=f"{DAY}T18:00", new_duration_minutes=30)

    assert saved(repos, event).end == f"{DAY}T13:30"


def test_positive_shift_moves_start_and_end_forward(repos):
    event = add_event(repos, "work", f"{DAY}T09:00", f"{DAY}T17:00")

    update(repos, "work", shift_minutes=30)

    assert (saved(repos, event).start, saved(repos, event).end) == (f"{DAY}T09:30", f"{DAY}T17:30")


def test_negative_shift_moves_both_earlier(repos):
    event = add_event(repos, "work", f"{DAY}T09:00", f"{DAY}T17:00")

    update(repos, "work", shift_minutes=-15)

    assert (saved(repos, event).start, saved(repos, event).end) == (f"{DAY}T08:45", f"{DAY}T16:45")


def test_shift_is_ignored_with_absolute_values(repos):
    event = lunch(repos)

    update(repos, "lunch", shift_minutes=120, new_start=f"{DAY}T13:30")

    assert saved(repos, event).start == f"{DAY}T13:30"


def test_title_alone_does_not_touch_the_time(repos):
    event = lunch(repos)

    outcome = update(repos, "lunch", new_title="work lunch")

    assert saved(repos, event).title == "work lunch"
    assert (saved(repos, event).start, saved(repos, event).end) == (f"{DAY}T13:00", f"{DAY}T14:00")
    assert outcome.title == "work lunch"


def test_invalid_range_does_not_write(repos):
    event = lunch(repos)

    outcome = update(repos, "lunch", new_start=f"{DAY}T15:00", new_end=f"{DAY}T14:30")

    assert outcome == InvalidRange()
    assert saved(repos, event).start == f"{DAY}T13:00"


def test_not_found_and_ambiguous(repos):
    assert update(repos, "missing") == EventNotFound()

    add_event(repos, "call with Marco", f"{DAY}T10:00", f"{DAY}T10:30")
    add_event(repos, "call with the client", f"{DAY}T15:00", f"{DAY}T15:30")
    ambiguous = update(repos, "call", new_start=f"{DAY}T11:00")
    assert isinstance(ambiguous, EventAmbiguous)
    assert len(ambiguous.candidates) == 2


def test_conflicts_detected_after_the_action_event_saved_anyway(repos):
    add_event(repos, "meeting", f"{DAY}T10:00", f"{DAY}T11:00")
    event = lunch(repos)

    outcome = update(repos, "lunch", new_start=f"{DAY}T10:30")

    assert [c.title for c in outcome.conflicts] == ["meeting"]
    assert saved(repos, event).start == f"{DAY}T10:30"


def test_cascade_candidates_are_the_derived_events_and_are_untouched(repos):
    parent = add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    shower = add_event(
        repos, "shower", f"{DAY}T19:00", f"{DAY}T19:15", derived_from_event_id=parent.event_id
    )

    outcome = update(repos, "gym", shift_minutes=30)

    assert [c.title for c in outcome.cascade_candidates] == ["shower"]
    assert saved(repos, shower).start == f"{DAY}T19:00"


def test_no_cascade_candidate_if_the_event_did_not_move(repos):
    parent = add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    add_event(
        repos, "shower", f"{DAY}T19:00", f"{DAY}T19:15", derived_from_event_id=parent.event_id
    )

    outcome = update(repos, "gym", new_title="intense gym")

    assert outcome.cascade_candidates == ()


def test_outcome_reports_the_original_start_for_computing_the_shift(repos):
    lunch(repos)

    outcome = update(repos, "lunch", shift_minutes=30)

    assert outcome.original_start == f"{DAY}T13:00"
    assert outcome.start == f"{DAY}T13:30"


def test_without_date_uses_the_default_date(repos):
    event = lunch(repos)

    outcome = update_event(
        repos, event_description="lunch", default_date=DAY, new_start=f"{DAY}T13:30"
    )

    assert isinstance(outcome, EventUpdated)
    assert saved(repos, event).start == f"{DAY}T13:30"


# --- further cases -------------------------------------------------------------------


def test_event_time_disambiguates_same_name_events(repos):
    add_event(repos, "call with Marco", f"{DAY}T10:00", f"{DAY}T10:30")
    afternoon = add_event(repos, "call with Marco", f"{DAY}T15:00", f"{DAY}T15:30")

    outcome = update(repos, "call with Marco", event_time="15:00", shift_minutes=60)

    assert isinstance(outcome, EventUpdated)
    assert saved(repos, afternoon).start == f"{DAY}T16:00"


def test_without_date_there_is_nothing_to_search(repos):
    lunch(repos)

    assert update_event(repos, event_description="lunch", shift_minutes=30) == EventNotFound()


def test_outcome_also_reports_previous_end_and_title(repos):
    lunch(repos)

    outcome = update(repos, "lunch", new_end=f"{DAY}T14:30", new_title="long lunch")

    assert (outcome.original_start, outcome.original_end, outcome.original_title) == (
        f"{DAY}T13:00",
        f"{DAY}T14:00",
        "lunch",
    )
    assert (outcome.start, outcome.end, outcome.title) == (
        f"{DAY}T13:00",
        f"{DAY}T14:30",
        "long lunch",
    )


def test_extend_lengthens_the_end_without_knowing_it(repos):
    event = lunch(repos)

    outcome = update(repos, "lunch", extend_minutes=30)

    assert (saved(repos, event).start, saved(repos, event).end) == (f"{DAY}T13:00", f"{DAY}T14:30")
    assert isinstance(outcome, EventUpdated) and outcome.original_end == f"{DAY}T14:00"


def test_negative_extend_shortens_and_cannot_cancel_the_event(repos):
    event = lunch(repos)

    assert isinstance(update(repos, "lunch", extend_minutes=-15), EventUpdated)
    assert saved(repos, event).end == f"{DAY}T13:45"
    assert update(repos, "lunch", extend_minutes=-60) == InvalidRange()
    assert saved(repos, event).end == f"{DAY}T13:45"


def test_extend_adds_to_the_end_computed_from_other_values(repos):
    event = lunch(repos)

    update(repos, "lunch", new_start=f"{DAY}T13:30", extend_minutes=15)  # duration 60 kept

    assert (saved(repos, event).start, saved(repos, event).end) == (f"{DAY}T13:30", f"{DAY}T14:45")
