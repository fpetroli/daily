import pytest

from daily.storage.schedule_repo import ScheduleRepository


@pytest.fixture
def repo(conn):
    return ScheduleRepository(conn)


def test_insert_event_and_get_by_id(repo):
    event_id = repo.insert_event(title="gym", start="2026-08-05T18:00", end="2026-08-05T19:00")

    event = repo.get_by_id(event_id)

    assert event is not None
    assert event.title == "gym"
    assert event.start == "2026-08-05T18:00"
    assert event.end == "2026-08-05T19:00"
    assert event.derived_from_event_id is None


def test_get_by_id_unknown_returns_none(repo):
    assert repo.get_by_id("missing") is None


def test_events_for_date_only_that_day_ordered_by_start(repo):
    repo.insert_event(title="lunch", start="2026-08-05T13:00", end="2026-08-05T14:00")
    repo.insert_event(title="gym", start="2026-08-05T08:00", end="2026-08-05T09:00")
    repo.insert_event(title="another day", start="2026-08-06T08:00", end="2026-08-06T09:00")

    events = repo.events_for_date("2026-08-05")

    assert [e.title for e in events] == ["gym", "lunch"]


def test_events_for_date_day_boundaries(repo):
    repo.insert_event(title="midnight", start="2026-08-05T00:00", end="2026-08-05T00:30")
    repo.insert_event(title="last minute", start="2026-08-05T23:59", end="2026-08-06T00:30")
    repo.insert_event(title="next day", start="2026-08-06T00:00", end="2026-08-06T01:00")

    titles = [e.title for e in repo.events_for_date("2026-08-05")]

    assert titles == ["midnight", "last minute"]


def test_find_conflicts_partial_overlap_and_adjacent(repo):
    repo.insert_event(title="meeting", start="2026-08-05T10:00", end="2026-08-05T11:00")

    overlapping = repo.find_conflicts(start="2026-08-05T10:30", end="2026-08-05T11:30")
    assert [e.title for e in overlapping] == ["meeting"]

    adjacent = repo.find_conflicts(start="2026-08-05T11:00", end="2026-08-05T12:00")
    assert adjacent == []


def test_find_conflicts_excludes_the_event_itself(repo):
    event_id = repo.insert_event(title="yoga", start="2026-08-05T18:00", end="2026-08-05T19:00")

    conflicts = repo.find_conflicts(
        start="2026-08-05T18:00", end="2026-08-05T19:00", exclude_event_id=event_id
    )

    assert conflicts == []


def test_events_derived_from_finds_cascade_candidates(repo):
    parent_id = repo.insert_event(title="gym", start="2026-08-05T18:00", end="2026-08-05T19:00")
    repo.insert_event(
        title="shower",
        start="2026-08-05T19:00",
        end="2026-08-05T19:15",
        derived_from_event_id=parent_id,
    )
    repo.insert_event(title="independent event", start="2026-08-05T20:00", end="2026-08-05T21:00")

    derived = repo.events_derived_from(parent_id)

    assert [e.title for e in derived] == ["shower"]


def test_update_event_changes_only_the_given_fields(repo):
    event_id = repo.insert_event(title="work", start="2026-08-05T09:00", end="2026-08-05T17:00")

    repo.update_event(event_id, start="2026-08-05T09:30")
    updated = repo.get_by_id(event_id)

    assert updated.start == "2026-08-05T09:30"
    assert updated.end == "2026-08-05T17:00"  # unchanged
    assert updated.title == "work"  # unchanged


def test_update_event_without_fields_does_nothing(repo):
    event_id = repo.insert_event(title="work", start="2026-08-05T09:00", end="2026-08-05T17:00")

    repo.update_event(event_id)

    assert repo.get_by_id(event_id).title == "work"


def test_delete_event_removes_the_row(repo):
    event_id = repo.insert_event(
        title="to remove", start="2026-08-05T09:00", end="2026-08-05T10:00"
    )

    repo.delete_event(event_id)

    assert repo.get_by_id(event_id) is None
