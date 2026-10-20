import pytest
from pydantic import ValidationError

from daily.config import Settings


def test_allowed_users_are_a_comma_separated_list(monkeypatch):
    monkeypatch.setenv("DAILY_TELEGRAM_ALLOWED_USERS", "42, 7 1001")

    assert Settings(_env_file=None).telegram_allowed_users == frozenset({42, 7, 1001})


def test_nobody_is_allowed_by_default():
    assert Settings(_env_file=None).telegram_allowed_users == frozenset()


def test_an_unknown_time_zone_is_a_startup_error():
    with pytest.raises(ValidationError, match="unknown time zone"):
        Settings(_env_file=None, timezone="Mars/Olympus")


def test_a_short_bot_token_is_a_startup_error():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, telegram_bot_token="short")
