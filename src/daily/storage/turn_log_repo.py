"""Access to `turn_log`: every chat turn, to build eval cases and user flags."""

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

# How many turns go into the context saved with a flag: the same window the model sees
# (`agent.turn._MAX_HISTORY_RUNS`), not a second independent magic number.
_FLAG_CONTEXT_TURNS = 3

# Where the fix of a report went: the values `flag_outcome` accepts.
Outcome = Literal["code", "prompt", "model", "no_change"]
OUTCOMES: tuple[Outcome, ...] = ("code", "prompt", "model", "no_change")


@dataclass(frozen=True, slots=True)
class Report:
    """A flagged turn: what was said, what the bot did, what the user noted."""

    turn_id: str
    created_at: str
    message: str
    kind: str
    reply_text: str
    note: str
    handled_at: str | None = None
    outcome: str | None = None


class AmbiguousTurnId(ValueError):
    """The id prefix matches more than one report."""


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
            "UPDATE turn_log SET flag_note = ?, flag_context_json = ?, flag_handled_at = NULL,"
            " flag_outcome = NULL WHERE turn_id = ?",
            (note, json.dumps(context, ensure_ascii=False), turn_id),
        )
        return cursor.rowcount > 0

    def reports(
        self, *, include_handled: bool = False, outcome: Outcome | None = None
    ) -> list[Report]:
        """The flagged turns, oldest first: only the open ones unless `include_handled`, only
        those going to `outcome` if given."""
        where, args = "flag_note IS NOT NULL", []
        if not include_handled:
            where += " AND flag_handled_at IS NULL"
        if outcome is not None:
            where += " AND flag_outcome = ?"
            args.append(outcome)
        rows = self._conn.execute(
            "SELECT turn_id, created_at, message, kind, reply_text, flag_note, flag_handled_at,"
            f" flag_outcome FROM turn_log WHERE {where} ORDER BY rowid",
            args,
        ).fetchall()
        return [
            Report(
                row["turn_id"],
                row["created_at"],
                row["message"],
                row["kind"],
                row["reply_text"],
                row["flag_note"],
                row["flag_handled_at"],
                row["flag_outcome"],
            )
            for row in rows
        ]

    def handle(self, turn_id: str, outcome: Outcome, *, now: datetime | None = None) -> str | None:
        """The report is fixed, and `outcome` says where: it is closed. `turn_id` may be a
        prefix of the id, as long as it matches a single report. Returns the full id, `None`
        if no report matches; raises `AmbiguousTurnId` if more than one does."""
        handled_at = (now or datetime.now()).isoformat(timespec="seconds")
        return self._set_outcome(turn_id, outcome, handled_at)

    def assign(self, turn_id: str, outcome: Outcome) -> str | None:
        """Where the fix of the report will go, while it is still to fix: it stays open (and
        is opened again if it was closed). Same matching as `handle`."""
        return self._set_outcome(turn_id, outcome, None)

    def _set_outcome(self, turn_id: str, outcome: Outcome, handled_at: str | None) -> str | None:
        rows = self._conn.execute(
            "SELECT turn_id FROM turn_log WHERE flag_note IS NOT NULL AND turn_id LIKE ? || '%'",
            (turn_id,),
        ).fetchall()
        if len(rows) > 1:
            raise AmbiguousTurnId(turn_id)
        if not rows:
            return None
        full = rows[0]["turn_id"]
        self._conn.execute(
            "UPDATE turn_log SET flag_handled_at = ?, flag_outcome = ? WHERE turn_id = ?",
            (handled_at, outcome, full),
        )
        return full

    def latest_turn_id(self) -> str | None:
        """The most recent turn (by `rowid`, as `flag`), the one a report refers to."""
        row = self._conn.execute(
            "SELECT turn_id FROM turn_log ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return row["turn_id"] if row else None
