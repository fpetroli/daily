"""Access to the 4 `notes.*` tables: behavioural patterns not confirmed yet.

Tracking only (counter and 2-week window). The decision to propose a promotion (threshold
reached, state still `pending`) lives in the services.

Rules common to all 4:
- the pattern's reference (time, duration, offset, gap) is the one of the first occurrence
  or of the last reset: later occurrences within tolerance only increment the counter,
  they do not move it;
- a reset because the window expired does **not** bring a `rejected` or `confirmed`
  pattern back to `pending`: "never propose again" is permanent, separate from the
  counter's freshness.
"""

import sqlite3
import uuid
from datetime import datetime

from daily.domain.models import (
    EventPattern,
    GapPattern,
    SequencePattern,
    TriggerDurationPattern,
)
from daily.domain.patterns import NoteStatus, next_window_state
from daily.domain.timeutil import minutes_of_day


def _first_at(row: sqlite3.Row) -> datetime:
    return datetime.fromisoformat(row["first_occurrence_at"])


def _set_status(conn: sqlite3.Connection, table: str, note_id: str, status: NoteStatus) -> None:
    conn.execute(f"UPDATE {table} SET status = ? WHERE note_id = ?", (status.value, note_id))


class EventPatternsRepository:
    """Standalone event repeated at a similar time and duration (match on both)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @staticmethod
    def _to_pattern(row: sqlite3.Row) -> EventPattern:
        return EventPattern(
            note_id=row["note_id"],
            title=row["title"],
            typical_start=row["typical_start"],
            typical_duration_minutes=row["typical_duration_minutes"],
            occurrence_count=row["occurrence_count"],
            first_occurrence_at=_first_at(row),
            status=NoteStatus(row["status"]),
        )

    def record_occurrence(
        self,
        *,
        title: str,
        observed_start: str,
        observed_duration_minutes: int,
        time_tolerance_minutes: int,
        duration_tolerance_minutes: int,
        now: datetime | None = None,
    ) -> EventPattern:
        effective_now = now or datetime.now()
        rows = self._conn.execute(
            "SELECT * FROM event_patterns WHERE title = ? COLLATE NOCASE", (title,)
        ).fetchall()
        observed_minutes = minutes_of_day(observed_start)

        def within_tolerance(p: EventPattern) -> bool:
            start_diff = abs(minutes_of_day(p.typical_start) - observed_minutes)
            duration_diff = abs(p.typical_duration_minutes - observed_duration_minutes)
            return (
                start_diff <= time_tolerance_minutes and duration_diff <= duration_tolerance_minutes
            )

        match = next((p for p in map(self._to_pattern, rows) if within_tolerance(p)), None)
        window = next_window_state(
            existing_count=match.occurrence_count if match else None,
            existing_first_occurrence_at=match.first_occurrence_at if match else None,
            now=effective_now,
        )
        is_reset = match is not None and window.occurrence_count == 1
        keep_reference = match is not None and not is_reset
        reference_start = match.typical_start if keep_reference else observed_start
        reference_duration = (
            match.typical_duration_minutes if keep_reference else observed_duration_minutes
        )
        status = match.status if match else NoteStatus.PENDING
        note_id = match.note_id if match else str(uuid.uuid4())

        self._conn.execute(
            "INSERT OR REPLACE INTO event_patterns (note_id, title, typical_start,"
            " typical_duration_minutes, occurrence_count, first_occurrence_at, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                note_id,
                title,
                reference_start,
                reference_duration,
                window.occurrence_count,
                window.first_occurrence_at.isoformat(),
                status.value,
            ),
        )
        return EventPattern(
            note_id=note_id,
            title=title,
            typical_start=reference_start,
            typical_duration_minutes=reference_duration,
            occurrence_count=window.occurrence_count,
            first_occurrence_at=window.first_occurrence_at,
            status=status,
        )

    def list_pending(self) -> list[EventPattern]:
        rows = self._conn.execute("SELECT * FROM event_patterns WHERE status = 'pending'")
        return [self._to_pattern(row) for row in rows]

    def set_status(self, note_id: str, status: NoteStatus) -> None:
        _set_status(self._conn, "event_patterns", note_id, status)


class TriggerDurationPatternsRepository:
    """Duration of an event with a rule but no known duration, asked ad hoc (match on duration)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @staticmethod
    def _to_pattern(row: sqlite3.Row) -> TriggerDurationPattern:
        return TriggerDurationPattern(
            note_id=row["note_id"],
            title=row["title"],
            typical_duration_minutes=row["typical_duration_minutes"],
            occurrence_count=row["occurrence_count"],
            first_occurrence_at=_first_at(row),
            status=NoteStatus(row["status"]),
        )

    def record_occurrence(
        self,
        *,
        title: str,
        observed_duration_minutes: int,
        duration_tolerance_minutes: int,
        now: datetime | None = None,
    ) -> TriggerDurationPattern:
        effective_now = now or datetime.now()
        rows = self._conn.execute(
            "SELECT * FROM trigger_duration_patterns WHERE title = ? COLLATE NOCASE", (title,)
        ).fetchall()
        match = next(
            (
                p
                for p in map(self._to_pattern, rows)
                if abs(p.typical_duration_minutes - observed_duration_minutes)
                <= duration_tolerance_minutes
            ),
            None,
        )
        window = next_window_state(
            existing_count=match.occurrence_count if match else None,
            existing_first_occurrence_at=match.first_occurrence_at if match else None,
            now=effective_now,
        )
        is_reset = match is not None and window.occurrence_count == 1
        reference_duration = (
            match.typical_duration_minutes
            if match is not None and not is_reset
            else observed_duration_minutes
        )
        status = match.status if match else NoteStatus.PENDING
        note_id = match.note_id if match else str(uuid.uuid4())

        self._conn.execute(
            "INSERT OR REPLACE INTO trigger_duration_patterns (note_id, title,"
            " typical_duration_minutes, occurrence_count, first_occurrence_at, status)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                note_id,
                title,
                reference_duration,
                window.occurrence_count,
                window.first_occurrence_at.isoformat(),
                status.value,
            ),
        )
        return TriggerDurationPattern(
            note_id=note_id,
            title=title,
            typical_duration_minutes=reference_duration,
            occurrence_count=window.occurrence_count,
            first_occurrence_at=window.first_occurrence_at,
            status=status,
        )

    def list_pending(self) -> list[TriggerDurationPattern]:
        rows = self._conn.execute(
            "SELECT * FROM trigger_duration_patterns WHERE status = 'pending'"
        )
        return [self._to_pattern(row) for row in rows]

    def set_status(self, note_id: str, status: NoteStatus) -> None:
        _set_status(self._conn, "trigger_duration_patterns", note_id, status)


class SequencePatternsRepository:
    """Relation between two consecutive events, not taught yet (match on the offset).

    The caller must check that no rule exists yet for the pair before recording the
    occurrence: it is not duplicated here.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @staticmethod
    def _to_pattern(row: sqlite3.Row) -> SequencePattern:
        return SequencePattern(
            note_id=row["note_id"],
            trigger_title=row["trigger_title"],
            related_title=row["related_title"],
            position=row["position"],
            offset_minutes=row["offset_minutes"],
            occurrence_count=row["occurrence_count"],
            first_occurrence_at=_first_at(row),
            status=NoteStatus(row["status"]),
        )

    def record_occurrence(
        self,
        *,
        trigger_title: str,
        related_title: str,
        position: str,
        observed_offset_minutes: int,
        time_tolerance_minutes: int,
        now: datetime | None = None,
    ) -> SequencePattern:
        effective_now = now or datetime.now()
        rows = self._conn.execute(
            "SELECT * FROM sequence_patterns WHERE trigger_title = ? COLLATE NOCASE"
            " AND related_title = ? COLLATE NOCASE AND position = ?",
            (trigger_title, related_title, position),
        ).fetchall()
        match = next(
            (
                p
                for p in map(self._to_pattern, rows)
                if abs(p.offset_minutes - observed_offset_minutes) <= time_tolerance_minutes
            ),
            None,
        )
        window = next_window_state(
            existing_count=match.occurrence_count if match else None,
            existing_first_occurrence_at=match.first_occurrence_at if match else None,
            now=effective_now,
        )
        is_reset = match is not None and window.occurrence_count == 1
        reference_offset = (
            match.offset_minutes if match is not None and not is_reset else observed_offset_minutes
        )
        status = match.status if match else NoteStatus.PENDING
        note_id = match.note_id if match else str(uuid.uuid4())

        self._conn.execute(
            "INSERT OR REPLACE INTO sequence_patterns (note_id, trigger_title, related_title,"
            " position, offset_minutes, occurrence_count, first_occurrence_at, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                note_id,
                trigger_title,
                related_title,
                position,
                reference_offset,
                window.occurrence_count,
                window.first_occurrence_at.isoformat(),
                status.value,
            ),
        )
        return SequencePattern(
            note_id=note_id,
            trigger_title=trigger_title,
            related_title=related_title,
            position=position,
            offset_minutes=reference_offset,
            occurrence_count=window.occurrence_count,
            first_occurrence_at=window.first_occurrence_at,
            status=status,
        )

    def list_pending(self) -> list[SequencePattern]:
        rows = self._conn.execute("SELECT * FROM sequence_patterns WHERE status = 'pending'")
        return [self._to_pattern(row) for row in rows]

    def set_status(self, note_id: str, status: NoteStatus) -> None:
        _set_status(self._conn, "sequence_patterns", note_id, status)


class GapPatternsRepository:
    """Recurring empty slot tied to an anchor (match on the gap minutes).

    Promotion differs from the other 3: it also needs a name (`trigger_title`) the system
    cannot invent, so this table holds no title.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @staticmethod
    def _to_pattern(row: sqlite3.Row) -> GapPattern:
        return GapPattern(
            note_id=row["note_id"],
            related_title=row["related_title"],
            position=row["position"],
            gap_minutes=row["gap_minutes"],
            occurrence_count=row["occurrence_count"],
            first_occurrence_at=_first_at(row),
            status=NoteStatus(row["status"]),
        )

    def record_occurrence(
        self,
        *,
        related_title: str,
        position: str,
        observed_gap_minutes: int,
        time_tolerance_minutes: int,
        now: datetime | None = None,
    ) -> GapPattern:
        effective_now = now or datetime.now()
        rows = self._conn.execute(
            "SELECT * FROM gap_patterns WHERE related_title = ? COLLATE NOCASE AND position = ?",
            (related_title, position),
        ).fetchall()
        match = next(
            (
                p
                for p in map(self._to_pattern, rows)
                if abs(p.gap_minutes - observed_gap_minutes) <= time_tolerance_minutes
            ),
            None,
        )
        window = next_window_state(
            existing_count=match.occurrence_count if match else None,
            existing_first_occurrence_at=match.first_occurrence_at if match else None,
            now=effective_now,
        )
        is_reset = match is not None and window.occurrence_count == 1
        reference_gap = (
            match.gap_minutes if match is not None and not is_reset else observed_gap_minutes
        )
        status = match.status if match else NoteStatus.PENDING
        note_id = match.note_id if match else str(uuid.uuid4())

        self._conn.execute(
            "INSERT OR REPLACE INTO gap_patterns (note_id, related_title, position,"
            " gap_minutes, occurrence_count, first_occurrence_at, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                note_id,
                related_title,
                position,
                reference_gap,
                window.occurrence_count,
                window.first_occurrence_at.isoformat(),
                status.value,
            ),
        )
        return GapPattern(
            note_id=note_id,
            related_title=related_title,
            position=position,
            gap_minutes=reference_gap,
            occurrence_count=window.occurrence_count,
            first_occurrence_at=window.first_occurrence_at,
            status=status,
        )

    def list_pending(self) -> list[GapPattern]:
        rows = self._conn.execute("SELECT * FROM gap_patterns WHERE status = 'pending'")
        return [self._to_pattern(row) for row in rows]

    def set_status(self, note_id: str, status: NoteStatus) -> None:
        _set_status(self._conn, "gap_patterns", note_id, status)
