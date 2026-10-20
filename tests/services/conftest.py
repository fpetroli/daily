import sqlite3
from datetime import datetime

import pytest

from daily.domain.models import ScheduleEvent
from daily.services.context import Repositories

DAY = "2026-08-05"
# Noon of the test day: the "today" of the tests that depend on `now`.
NOW = datetime(2026, 8, 5, 12, 0)


@pytest.fixture
def repos(conn: sqlite3.Connection) -> Repositories:
    return Repositories.from_connection(conn)


def add_event(
    repos: Repositories, title: str, start: str, end: str, **derived: str
) -> ScheduleEvent:
    """Inserts an event directly in the repository (a fixture: it skips the services)."""
    event_id = repos.schedule.insert_event(title=title, start=start, end=end, **derived)
    event = repos.schedule.get_by_id(event_id)
    assert event is not None
    return event
