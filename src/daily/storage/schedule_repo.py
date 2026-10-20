"""Access to the `schedule` table. CRUD and queries only: domain logic lives in the services."""

import sqlite3
import uuid

from daily.domain.models import ScheduleEvent


def _to_event(row: sqlite3.Row) -> ScheduleEvent:
    return ScheduleEvent(
        event_id=row["event_id"],
        title=row["title"],
        start=row["start"],
        end=row["end"],
        derived_from_event_id=row["derived_from_event_id"],
        derived_from_rule_id=row["derived_from_rule_id"],
    )


class ScheduleRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert_event(
        self,
        *,
        title: str,
        start: str,
        end: str,
        derived_from_event_id: str | None = None,
        derived_from_rule_id: str | None = None,
        event_id: str | None = None,
    ) -> str:
        """Explicit `event_id` only to restore a deleted event (undo): otherwise a new one is
        generated."""
        event_id = event_id or str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO schedule (event_id, title, start, end,"
            " derived_from_event_id, derived_from_rule_id) VALUES (?, ?, ?, ?, ?, ?)",
            (event_id, title, start, end, derived_from_event_id, derived_from_rule_id),
        )
        return event_id

    def get_by_id(self, event_id: str) -> ScheduleEvent | None:
        row = self._conn.execute(
            "SELECT * FROM schedule WHERE event_id = ?", (event_id,)
        ).fetchone()
        return _to_event(row) if row else None

    def events_for_date(self, date: str) -> list[ScheduleEvent]:
        """Events of day `date` (`YYYY-MM-DD`), ordered by start.

        Range scan on `start` (a sortable ISO string).
        """
        rows = self._conn.execute(
            "SELECT * FROM schedule WHERE start >= ? AND start < ? ORDER BY start ASC",
            (f"{date}T00:00", f"{date}T24:00"),
        ).fetchall()
        return [_to_event(row) for row in rows]

    def find_conflicts(
        self, *, start: str, end: str, exclude_event_id: str | None = None
    ) -> list[ScheduleEvent]:
        """Events overlapping `[start, end)`: a shared function.

        `exclude_event_id` excludes the event itself from an update, otherwise it would
        always conflict with itself. Adjacent events do not conflict.
        """
        sql = "SELECT * FROM schedule WHERE start < ? AND end > ?"
        args: list[str] = [end, start]
        if exclude_event_id is not None:
            sql += " AND event_id != ?"
            args.append(exclude_event_id)
        return [_to_event(row) for row in self._conn.execute(sql, args).fetchall()]

    def events_derived_from(self, parent_event_id: str) -> list[ScheduleEvent]:
        """Events derived from `parent_event_id`: cascade candidates."""
        rows = self._conn.execute(
            "SELECT * FROM schedule WHERE derived_from_event_id = ?", (parent_event_id,)
        ).fetchall()
        return [_to_event(row) for row in rows]

    def update_event(
        self,
        event_id: str,
        *,
        title: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        """Updates only the given fields (`None` = leave unchanged)."""
        values = {"title": title, "start": start, "end": end}
        changes = {column: value for column, value in values.items() if value is not None}
        if not changes:
            return
        assignments = ", ".join(f"{column} = ?" for column in changes)
        self._conn.execute(
            f"UPDATE schedule SET {assignments} WHERE event_id = ?",
            (*changes.values(), event_id),
        )

    def delete_event(self, event_id: str) -> None:
        self._conn.execute("DELETE FROM schedule WHERE event_id = ?", (event_id,))
