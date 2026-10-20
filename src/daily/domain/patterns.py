"""Counter and time window shared by the 4 `notes.*` tables.

Pure (no database access): each repository uses it to compute
`occurrence_count`/`first_occurrence_at`, then writes the row itself.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

RESET_AFTER = timedelta(days=14)


class NoteStatus(StrEnum):
    """State of a pattern proposal in the 4 `notes.*` tables."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class WindowState:
    occurrence_count: int
    first_occurrence_at: datetime


def next_window_state(
    *,
    existing_count: int | None,
    existing_first_occurrence_at: datetime | None,
    now: datetime,
) -> WindowState:
    """New (counter, first occurrence) given the pattern's existing state.

    Starts from scratch if it did not exist, resets if the 2-week window has expired
    (strictly more than 14 days), otherwise increments.
    """
    if existing_count is None or existing_first_occurrence_at is None:
        return WindowState(1, now)
    if now - existing_first_occurrence_at > RESET_AFTER:
        return WindowState(1, now)
    return WindowState(existing_count + 1, existing_first_occurrence_at)
