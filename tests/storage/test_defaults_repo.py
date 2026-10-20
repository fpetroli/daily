import pytest

from daily.storage.defaults_repo import EventDefaultsRepository


@pytest.fixture
def repo(conn):
    return EventDefaultsRepository(conn)


def test_upsert_creates_a_new_row_if_missing(repo):
    repo.upsert("gym", typical_start="18:00", typical_duration_minutes=60)

    row = repo.get_by_title("gym")

    assert row is not None
    assert row.typical_start == "18:00"
    assert row.typical_duration_minutes == 60


def test_case_insensitive_match_on_title(repo):
    repo.upsert("Gym", typical_start="18:00", typical_duration_minutes=60)

    assert repo.get_by_title("Gym") is not None


def test_upsert_with_different_case_updates_the_row_and_creates_none(repo, conn):
    repo.upsert("Gym", typical_start="18:00")
    repo.upsert("gym", typical_duration_minutes=60)

    assert conn.execute("SELECT COUNT(*) FROM event_defaults").fetchone()[0] == 1
    row = repo.get_by_title("gym")
    assert row.typical_start == "18:00"
    assert row.typical_duration_minutes == 60


def test_upsert_with_omitted_field_does_not_overwrite_existing_value(repo):
    repo.upsert("meeting", typical_start="09:00", typical_duration_minutes=30)

    repo.upsert("meeting", typical_duration_minutes=45)  # typical_start omitted

    row = repo.get_by_title("meeting")
    assert row.typical_start == "09:00"  # untouched by the second upsert
    assert row.typical_duration_minutes == 45


def test_clear_field_clears_only_the_given_field(repo):
    repo.upsert("dentist", typical_start="10:00", typical_duration_minutes=30)

    repo.clear_field("dentist", "typical_duration_minutes")

    row = repo.get_by_title("dentist")
    assert row is not None
    assert row.typical_start == "10:00"
    assert row.typical_duration_minutes is None


def test_clear_field_rejects_an_unknown_field(repo):
    repo.upsert("dentist", typical_start="10:00")

    with pytest.raises(ValueError):
        repo.clear_field("dentist", "title; DROP TABLE event_defaults")


def test_delete_row_deletes_the_whole_row(repo):
    repo.upsert("to remove", typical_start="08:00", typical_duration_minutes=15)

    repo.delete_row("to remove")

    assert repo.get_by_title("to remove") is None


def test_list_all_without_rows_is_empty(repo):
    assert repo.list_all() == []


def test_list_all_returns_every_row(repo):
    repo.upsert("gym", typical_start="18:00", typical_duration_minutes=60)
    repo.upsert("breakfast", typical_duration_minutes=15)

    titles = {d.title for d in repo.list_all()}

    assert titles == {"gym", "breakfast"}
