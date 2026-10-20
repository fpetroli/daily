from datetime import datetime

import pytest

from daily.storage.action_log_repo import ActionLogRepository

NOW = datetime(2026, 9, 22, 10, 0)


@pytest.fixture
def repo(conn):
    return ActionLogRepository(conn)


def test_without_rows_latest_is_none(repo):
    assert repo.latest() is None


def test_record_and_latest(repo):
    repo.record(kind="add_event", summary="added 'gym'", undo_data={"event_id": "e1"}, now=NOW)

    entry = repo.latest()

    assert entry is not None
    assert entry.kind == "add_event"
    assert entry.summary == "added 'gym'"
    assert entry.undo_data == {"event_id": "e1"}


def test_latest_is_the_most_recent_row(repo):
    repo.record(kind="add_event", summary="first", undo_data={}, now=datetime(2026, 9, 22, 9, 0))
    repo.record(
        kind="remove_event", summary="second", undo_data={}, now=datetime(2026, 9, 22, 9, 30)
    )

    assert repo.latest().summary == "second"


def test_delete_removes_the_row(repo):
    repo.record(kind="add_event", summary="x", undo_data={}, now=NOW)
    entry = repo.latest()

    repo.delete(entry.log_id)

    assert repo.latest() is None
