"""Opening of the SQLite database and numbered migrations.

Migrations are `NNN_name.sql` files in `migrations/`, applied in order; the current version
is kept in `PRAGMA user_version` (no dedicated table).

The connection is in autocommit mode (`isolation_level=None`): every statement commits on
its own. Operations that write several rows together use [`transaction`].
"""

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

MEMORY = ":memory:"
_MIGRATION_FILE = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")


def _load_migrations() -> list[tuple[int, str]]:
    folder = resources.files("daily.storage") / "migrations"
    found: list[tuple[int, str]] = []
    for entry in folder.iterdir():
        match = _MIGRATION_FILE.match(entry.name)
        if match:
            found.append((int(match.group(1)), entry.read_text(encoding="utf-8")))
    found.sort()
    if [version for version, _ in found] != list(range(1, len(found) + 1)):
        raise RuntimeError("Migrations must be numbered 001, 002, ... with no gaps")
    return found


def migrate(conn: sqlite3.Connection) -> None:
    """Brings the schema to the latest version, applying the missing steps in order."""
    migrations = _load_migrations()
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > len(migrations):
        raise RuntimeError(
            f"Database at version {current}, the code only knows up to {len(migrations)}"
        )
    for version, sql in migrations:
        if version <= current:
            continue
        try:
            conn.executescript(f"BEGIN;\n{sql}\nPRAGMA user_version = {version};\nCOMMIT;")
        except sqlite3.Error:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise


def open_database(path: str | Path = MEMORY) -> sqlite3.Connection:
    """Opens (or creates) the database and applies the migrations. `:memory:` for tests."""
    location = str(path)
    if location != MEMORY:
        Path(location).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(location, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if location != MEMORY:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Atomic block: commits if it completes, rolls everything back if it raises.

    If a transaction is already open the block joins it (SQLite does not nest): commit or
    rollback belong to the outer one.
    """
    if conn.in_transaction:
        yield
        return
    conn.execute("BEGIN")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
