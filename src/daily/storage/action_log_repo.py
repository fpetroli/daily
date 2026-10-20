"""Access to `action_log`: the last write performed, for undo."""

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class ActionLogEntry:
    log_id: str
    created_at: str
    kind: str
    summary: str
    undo_data: dict[str, Any]


def _to_entry(row: sqlite3.Row) -> ActionLogEntry:
    return ActionLogEntry(
        log_id=row["log_id"],
        created_at=row["created_at"],
        kind=row["kind"],
        summary=row["summary"],
        undo_data=json.loads(row["undo_data"]),
    )


class ActionLogRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def record(
        self, *, kind: str, summary: str, undo_data: dict[str, Any], now: datetime | None = None
    ) -> str:
        log_id = str(uuid.uuid4())
        created_at = (now or datetime.now()).isoformat(timespec="seconds")
        self._conn.execute(
            "INSERT INTO action_log (log_id, created_at, kind, summary, undo_data)"
            " VALUES (?, ?, ?, ?, ?)",
            (log_id, created_at, kind, summary, json.dumps(undo_data)),
        )
        return log_id

    def latest(self) -> ActionLogEntry | None:
        # `rowid` (implicit, the table is not `WITHOUT ROWID`) reflects the insertion order
        # even when two writes share the same `created_at` down to the second (a turn's `now`
        # is the same for the main action and its downstream proposals): `created_at` alone
        # would not tell them apart.
        row = self._conn.execute("SELECT * FROM action_log ORDER BY rowid DESC LIMIT 1").fetchone()
        return _to_entry(row) if row else None

    def delete(self, log_id: str) -> None:
        self._conn.execute("DELETE FROM action_log WHERE log_id = ?", (log_id,))
