"""Reads a `turn_log` row back, to check what the service wrote. The service itself never
reads it: the log is for people (`/report`, eval cases from real use)."""

import json
import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TurnRow:
    message: str
    tool: str | None
    call_json: dict[str, Any] | None
    kind: str
    reply_text: str
    flag_note: str | None
    duration_ms: int | None
    flag_context: list[dict[str, Any]] | None


def read_turn(conn: sqlite3.Connection, turn_id: str) -> TurnRow | None:
    row = conn.execute("SELECT * FROM turn_log WHERE turn_id = ?", (turn_id,)).fetchone()
    if row is None:
        return None
    return TurnRow(
        message=row["message"],
        tool=row["tool"],
        call_json=json.loads(row["call_json"]) if row["call_json"] is not None else None,
        kind=row["kind"],
        reply_text=row["reply_text"],
        flag_note=row["flag_note"],
        duration_ms=row["duration_ms"],
        flag_context=json.loads(row["flag_context_json"])
        if row["flag_context_json"] is not None
        else None,
    )
