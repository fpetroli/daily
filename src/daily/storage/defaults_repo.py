"""Access to `preferences.event_defaults`: usual time and duration of a single event.

Filled in two ways: explicitly (`add_event_rule` without `related_title`) or by habit
detection. This repository covers CRUD only.
"""

import sqlite3

from daily.domain.models import EventDefault

_CLEARABLE = ("typical_start", "typical_duration_minutes")


def _to_default(row: sqlite3.Row) -> EventDefault:
    return EventDefault(
        title=row["title"],
        typical_start=row["typical_start"],
        typical_duration_minutes=row["typical_duration_minutes"],
    )


class EventDefaultsRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_by_title(self, title: str) -> EventDefault | None:
        row = self._conn.execute(
            "SELECT * FROM event_defaults WHERE title = ? COLLATE NOCASE", (title,)
        ).fetchone()
        return _to_default(row) if row else None

    def list_all(self) -> list[EventDefault]:
        return [_to_default(row) for row in self._conn.execute("SELECT * FROM event_defaults")]

    def upsert(
        self,
        title: str,
        *,
        typical_start: str | None = None,
        typical_duration_minutes: int | None = None,
    ) -> None:
        """Writes or updates the row of `title`.

        An omitted field (`None`) does not overwrite the existing value: it stays as it is if
        the row exists, it stays `None` if the row is new. To clear a single field there is
        [`clear_field`].
        """
        existing = self.get_by_title(title)
        if existing is None:
            self._conn.execute(
                "INSERT INTO event_defaults (title, typical_start, typical_duration_minutes)"
                " VALUES (?, ?, ?)",
                (title, typical_start, typical_duration_minutes),
            )
            return
        self._conn.execute(
            "UPDATE event_defaults SET typical_start = ?, typical_duration_minutes = ?"
            " WHERE title = ? COLLATE NOCASE",
            (
                typical_start if typical_start is not None else existing.typical_start,
                typical_duration_minutes
                if typical_duration_minutes is not None
                else existing.typical_duration_minutes,
                title,
            ),
        )

    def clear_field(self, title: str, field: str) -> None:
        """Clears only `field` on the row of `title`, without deleting it."""
        if field not in _CLEARABLE:
            raise ValueError(f"Field cannot be cleared: {field!r}")
        self._conn.execute(
            f"UPDATE event_defaults SET {field} = NULL WHERE title = ? COLLATE NOCASE", (title,)
        )

    def delete_row(self, title: str) -> None:
        """Deletes the whole row of `title` (`remove_event_rule` without `field`)."""
        self._conn.execute("DELETE FROM event_defaults WHERE title = ? COLLATE NOCASE", (title,))
