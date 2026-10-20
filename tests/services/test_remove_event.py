from daily.services.schedule import EventAmbiguous, EventNotFound, EventRemoved, remove_event

from .conftest import DAY, add_event


def remove(repos, description, **kwargs):
    kwargs.setdefault("event_date", DAY)
    return remove_event(repos, event_description=description, default_date=DAY, **kwargs)


def test_removes_a_unique_match_and_really_deletes_it(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")

    outcome = remove(repos, "gym")

    assert isinstance(outcome, EventRemoved)
    assert repos.schedule.events_for_date(DAY) == []


def test_outcome_reports_title_and_time_of_the_removed_event(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")

    outcome = remove(repos, "gym")

    assert (outcome.removed_title, outcome.start, outcome.end) == (
        "gym",
        f"{DAY}T18:00",
        f"{DAY}T19:00",
    )


def test_no_match_touches_nothing(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")

    assert remove(repos, "dentist") == EventNotFound()
    assert len(repos.schedule.events_for_date(DAY)) == 1


def test_several_matches_is_ambiguous_and_removes_nothing(repos):
    add_event(repos, "call with Marco", f"{DAY}T10:00", f"{DAY}T10:30")
    add_event(repos, "call with the client", f"{DAY}T15:00", f"{DAY}T15:30")

    outcome = remove(repos, "call")

    assert isinstance(outcome, EventAmbiguous)
    assert len(outcome.candidates) == 2
    assert len(repos.schedule.events_for_date(DAY)) == 2


def test_without_date_uses_the_default_date(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")

    outcome = remove_event(repos, event_description="gym", default_date=DAY)

    assert isinstance(outcome, EventRemoved)


def test_derived_events_are_proposed_not_removed(repos):
    parent = add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    add_event(
        repos, "shower", f"{DAY}T19:00", f"{DAY}T19:15", derived_from_event_id=parent.event_id
    )

    outcome = remove(repos, "gym")

    assert [c.title for c in outcome.cascade_candidates] == ["shower"]
    assert [e.title for e in repos.schedule.events_for_date(DAY)] == ["shower"]


def test_no_derived_event_no_candidate(repos):
    add_event(repos, "standalone event", f"{DAY}T10:00", f"{DAY}T11:00")

    assert remove(repos, "standalone event").cascade_candidates == ()


# --- further cases -------------------------------------------------------------------


def test_time_disambiguates_same_name_events(repos):
    add_event(repos, "call with Marco", f"{DAY}T10:00", f"{DAY}T10:30")
    add_event(repos, "call with Marco", f"{DAY}T15:00", f"{DAY}T15:30")

    outcome = remove(repos, "call with Marco", event_time="10:00")

    assert isinstance(outcome, EventRemoved)
    assert [e.start for e in repos.schedule.events_for_date(DAY)] == [f"{DAY}T15:00"]


def test_without_date_there_is_nothing_to_search(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")

    assert remove_event(repos, event_description="gym") == EventNotFound()
    assert len(repos.schedule.events_for_date(DAY)) == 1
