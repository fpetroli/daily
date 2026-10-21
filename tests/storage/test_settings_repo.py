import pytest

from daily.storage.settings_repo import AppSettingsRepository


@pytest.fixture
def repo(conn):
    return AppSettingsRepository(conn)


def test_default_values_seeded(repo):
    assert repo.repetition_threshold() == 3
    assert repo.time_tolerance() == 15
    assert repo.duration_tolerance() == 10
    assert repo.reminder_minutes() == 5


def test_set_int_overwrites(repo):
    repo.set_int("repetition_threshold", 5)

    assert repo.repetition_threshold() == 5


def test_missing_key_raises(repo):
    with pytest.raises(KeyError, match="missing"):
        repo.get_int("missing")
