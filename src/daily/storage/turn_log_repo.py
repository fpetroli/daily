"""Access to `turn_log`: every chat turn, to build eval cases and user flags."""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Any

# How many turns go into the context saved with a flag: the same window the model sees
# (`agent.turn._MAX_HISTORY_RUNS`), not a second independent magic number.
_FLAG_CONTEXT_TURNS = 3


class TurnLogRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def record(
        self,
        *,
        message: str,
        tool: str | None,
        call_json: dict[str, Any] | None,
        kind: str,
        reply_text: str,
        duration_ms: int | None = None,
        now: datetime | None = None,
    ) -> str:
        turn_id = str(uuid.uuid4())
        created_at = (now or datetime.now()).isoformat(timespec="seconds")
        self._conn.execute(
            "INSERT INTO turn_log (turn_id, created_at, message, tool, call_json, kind,"
            " reply_text, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                turn_id,
                created_at,
                message,
                tool,
                json.dumps(call_json) if call_json is not None else None,
                kind,
                reply_text,
                duration_ms,
            ),
        )
        return turn_id

    def flag(self, turn_id: str, note: str) -> bool:
        """Stores a note on a turn, together with the context of the last
        `_FLAG_CONTEXT_TURNS` turns at most (the flagged one included). `False` if `turn_id`
        does not exist: in that case the subquery on `rowid` yields no rows."""
        rows = self._conn.execute(
            "SELECT turn_id, created_at, message, tool, call_json, kind, reply_text"
            " FROM turn_log WHERE rowid <= (SELECT rowid FROM turn_log WHERE turn_id = ?)"
            " ORDER BY rowid DESC LIMIT ?",
            (turn_id, _FLAG_CONTEXT_TURNS),
        ).fetchall()
        if not rows:
            return False
        context = [
            {
                "turn_id": row["turn_id"],
                "created_at": row["created_at"],
                "message": row["message"],
                "tool": row["tool"],
                "call_json": json.loads(row["call_json"]) if row["call_json"] is not None else None,
                "kind": row["kind"],
                "reply_text": row["reply_text"],
            }
            for row in reversed(rows)
        ]
        cursor = self._conn.execute(
            "UPDATE turn_log SET flag_note = ?, flag_context_json = ? WHERE turn_id = ?",
            (note, json.dumps(context, ensure_ascii=False), turn_id),
        )
        return cursor.rowcount > 0

    def latest_turn_id(self) -> str | None:
        """The most recent turn (by `rowid`, as `flag`), the one a report refers to."""
        row = self._conn.execute(
            "SELECT turn_id FROM turn_log ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return row["turn_id"] if row else None
