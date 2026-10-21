import sqlite3

import pytest

from daily.storage import db
from daily.storage.db import open_database, transaction


def test_creates_all_tables_and_seeds_app_settings(conn):
    tables = {
        row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {
        "schedule",
        "event_rules",
        "event_defaults",
        "app_settings",
        "event_patterns",
        "trigger_duration_patterns",
        "sequence_patterns",
        "gap_patterns",
        "action_log",
        "turn_log",
        "chat_messages",
    } <= tables

    settings = {
        row["setting_key"]: row["setting_value"]
        for row in conn.execute("SELECT * FROM app_settings")
    }
    assert settings["repetition_threshold"] == "3"
    assert settings["time_tolerance"] == "15"
    assert settings["duration_tolerance"] == "10"


def test_user_version_reaches_the_last_migration(conn):
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 4


def test_reopening_a_file_does_not_reapply_migrations(tmp_path):
    path = tmp_path / "data" / "daily.db"  # the folder does not exist yet
    first = open_database(path)
    first.execute("INSERT INTO event_defaults (title) VALUES ('gym')")
    first.close()

    second = open_database(path)

    assert second.execute("SELECT COUNT(*) FROM event_defaults").fetchone()[0] == 1
    assert second.execute("PRAGMA user_version").fetchone()[0] == 4
    second.close()


def test_a_file_uses_wal(tmp_path):
    connection = open_database(tmp_path / "daily.db")
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    connection.close()


def test_database_newer_than_the_code_is_rejected(tmp_path):
    path = tmp_path / "daily.db"
    raw = sqlite3.connect(path)
    raw.execute("PRAGMA user_version = 99")
    raw.close()

    with pytest.raises(RuntimeError, match="99"):
        open_database(path)


def test_a_failed_migration_leaves_no_half_schema(monkeypatch):
    monkeypatch.setattr(
        db, "_load_migrations", lambda: [(1, "CREATE TABLE a (x INTEGER); THIS IS NOT SQL;")]
    )
    with pytest.raises(sqlite3.Error):
        open_database()

    # New in-memory connection: we check atomicity on the same connection.
    connection = sqlite3.connect(":memory:", isolation_level=None)
    with pytest.raises(sqlite3.Error):
        db.migrate(connection)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
    leftovers = connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'a'")
    assert leftovers.fetchone()[0] == 0
    assert not connection.in_transaction


def test_transaction_commits_if_it_completes(conn):
    with transaction(conn):
        conn.execute("INSERT INTO event_defaults (title) VALUES ('a')")
        conn.execute("INSERT INTO event_defaults (title) VALUES ('b')")

    assert conn.execute("SELECT COUNT(*) FROM event_defaults").fetchone()[0] == 2


def test_transaction_rolls_back_everything_if_it_raises(conn):
    with pytest.raises(RuntimeError), transaction(conn):
        conn.execute("INSERT INTO event_defaults (title) VALUES ('a')")
        raise RuntimeError("boom")

    assert conn.execute("SELECT COUNT(*) FROM event_defaults").fetchone()[0] == 0
    assert not conn.in_transaction


def test_nested_transaction_joins_the_outer_one(conn):
    with pytest.raises(RuntimeError), transaction(conn):
        conn.execute("INSERT INTO event_defaults (title) VALUES ('outer')")
        with transaction(conn):
            conn.execute("INSERT INTO event_defaults (title) VALUES ('inner')")
        raise RuntimeError("boom after the inner block")

    assert conn.execute("SELECT COUNT(*) FROM event_defaults").fetchone()[0] == 0
