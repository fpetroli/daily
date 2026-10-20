from daily.services.resolver import ResolveError

from .conftest import DAY, add_event


def test_description_contained_in_title_resolves_with_a_single_match(repos):
    add_event(repos, "team meeting", f"{DAY}T10:00", f"{DAY}T11:00")

    result = repos.resolver.resolve(description="meeting", date=DAY)

    assert result.event is not None
    assert result.event.title == "team meeting"


def test_title_contained_in_description_resolves_with_a_single_match(repos):
    add_event(repos, "lunch", f"{DAY}T13:00", f"{DAY}T14:00")

    result = repos.resolver.resolve(description="today's lunch", date=DAY)

    assert result.event is not None
    assert result.event.title == "lunch"


def test_case_insensitive_match(repos):
    add_event(repos, "Gym", f"{DAY}T18:00", f"{DAY}T19:00")

    assert repos.resolver.resolve(description="gym", date=DAY).event is not None


def test_no_match_not_found(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")

    result = repos.resolver.resolve(description="dentist", date=DAY)

    assert result.event is None
    assert result.error == ResolveError.NOT_FOUND


def test_more_than_one_match_ambiguous_with_candidates(repos):
    add_event(repos, "call with Marco", f"{DAY}T10:00", f"{DAY}T10:30")
    add_event(repos, "call with the client", f"{DAY}T15:00", f"{DAY}T15:30")

    result = repos.resolver.resolve(description="call", date=DAY)

    assert result.error == ResolveError.AMBIGUOUS
    assert len(result.candidates) == 2


def test_ignores_events_of_other_days(repos):
    add_event(repos, "gym", "2026-08-06T18:00", "2026-08-06T19:00")

    assert repos.resolver.resolve(description="gym", date=DAY).error == ResolveError.NOT_FOUND


def test_empty_description_not_found(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")

    assert repos.resolver.resolve(description="  ", date=DAY).error == ResolveError.NOT_FOUND


# --- further cases -------------------------------------------------------------------


def test_time_disambiguates_same_name_events(repos):
    add_event(repos, "call with Marco", f"{DAY}T10:00", f"{DAY}T10:30")
    afternoon = add_event(repos, "call with Marco", f"{DAY}T15:00", f"{DAY}T15:30")

    result = repos.resolver.resolve(description="call with Marco", date=DAY, at_time="15:00")

    assert result.event == afternoon


def test_time_ignored_if_the_match_is_unique(repos):
    only = add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")

    result = repos.resolver.resolve(description="gym", date=DAY, at_time="07:00")

    assert result.event == only


def test_time_that_does_not_distinguish_stays_ambiguous(repos):
    add_event(repos, "call with Marco", f"{DAY}T10:00", f"{DAY}T10:30")
    add_event(repos, "call with Marco", f"{DAY}T15:00", f"{DAY}T15:30")

    result = repos.resolver.resolve(description="call with Marco", date=DAY, at_time="12:00")

    assert result.error == ResolveError.AMBIGUOUS
    assert len(result.candidates) == 2


def test_without_description_the_time_finds_the_event_starting_then(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    add_event(repos, "dinner", f"{DAY}T20:00", f"{DAY}T21:00")

    result = repos.resolver.resolve(description="", date=DAY, at_time="18:00")

    assert result.event.title == "gym"


def test_without_description_the_time_may_be_ambiguous_or_find_nothing(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    add_event(repos, "yoga", f"{DAY}T18:00", f"{DAY}T18:45")

    ambiguous = repos.resolver.resolve(description=" ", date=DAY, at_time="18:00")
    nothing = repos.resolver.resolve(description="", date=DAY, at_time="07:00")

    assert ambiguous.error == ResolveError.AMBIGUOUS and len(ambiguous.candidates) == 2
    assert nothing.error == ResolveError.NOT_FOUND
