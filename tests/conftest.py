import sqlite3
from collections.abc import Iterator

import pytest

from daily.services.context import Repositories
from daily.storage.db import open_database


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    """In-memory SQLite database with the full schema and the default settings."""
    connection = open_database()
    yield connection
    connection.close()


@pytest.fixture
def repos(conn: sqlite3.Connection) -> Repositories:
    return Repositories.from_connection(conn)
