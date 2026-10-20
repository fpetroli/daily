"""Access to `preferences.event_rules`. CRUD and queries only.

The read "at the two moments" (before and after placing an event), the choice of the
candidate to propose and the materialization of gaps (always dormant until confirmed) live
in the services, not here. No upsert: the caller decides whether `add_event_rule` creates or
updates.
"""

import sqlite3
import uuid

from daily.domain.models import EventRule


def _to_rule(row: sqlite3.Row) -> EventRule:
    return EventRule(
        rule_id=row["rule_id"],
        trigger_title=row["trigger_title"],
        related_title=row["related_title"],
        position=row["position"],
        offset_minutes=row["offset_minutes"],
        gap_minutes=row["gap_minutes"],
    )


class EventRulesRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert_rule(
        self,
        *,
        trigger_title: str,
        related_title: str,
        position: str,
        offset_minutes: int | None = None,
        gap_minutes: int | None = None,
    ) -> str:
        rule_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO event_rules (rule_id, trigger_title, related_title, position,"
            " offset_minutes, gap_minutes) VALUES (?, ?, ?, ?, ?, ?)",
            (rule_id, trigger_title, related_title, position, offset_minutes, gap_minutes),
        )
        return rule_id

    def _query(self, where: str, args: tuple[object, ...]) -> list[EventRule]:
        rows = self._conn.execute(f"SELECT * FROM event_rules WHERE {where}", args).fetchall()
        return [_to_rule(row) for row in rows]

    def find_by_trigger(self, trigger_title: str) -> list[EventRule]:
        """Moment 1: the rules in which `trigger_title` is the dependent event."""
        return self._query("trigger_title = ? COLLATE NOCASE", (trigger_title,))

    def find_by_related(self, related_title: str) -> list[EventRule]:
        """Moment 2: the downstream candidates, the rows in which the title is the anchor."""
        return self._query("related_title = ? COLLATE NOCASE", (related_title,))

    def find_gaps_for(self, related_title: str, position: str) -> list[EventRule]:
        """Gap rows for the same anchor and position."""
        return self._query(
            "related_title = ? COLLATE NOCASE AND position = ? AND gap_minutes IS NOT NULL",
            (related_title, position),
        )

    def find_matching(self, trigger_title: str, related_title: str) -> list[EventRule]:
        """Exact match for `remove_event_rule`: empty = not found, several rows = ambiguous."""
        return self._query(
            "trigger_title = ? COLLATE NOCASE AND related_title = ? COLLATE NOCASE",
            (trigger_title, related_title),
        )

    def get_all(self) -> list[EventRule]:
        return [_to_rule(row) for row in self._conn.execute("SELECT * FROM event_rules")]

    def delete_rule(self, rule_id: str) -> None:
        self._conn.execute("DELETE FROM event_rules WHERE rule_id = ?", (rule_id,))

    def update_rule(
        self,
        rule_id: str,
        *,
        position: str | None = None,
        offset_minutes: int | None = None,
        gap_minutes: int | None = None,
    ) -> None:
        """Updates only the given fields (`None` = leave unchanged, as for event_defaults)."""
        values = {
            "position": position,
            "offset_minutes": offset_minutes,
            "gap_minutes": gap_minutes,
        }
        changes = {column: value for column, value in values.items() if value is not None}
        if not changes:
            return
        assignments = ", ".join(f"{column} = ?" for column in changes)
        self._conn.execute(
            f"UPDATE event_rules SET {assignments} WHERE rule_id = ?",
            (*changes.values(), rule_id),
        )
