"""Time bands: the filter is code, the band is chosen by the model.

The model passes `time_band` as an optional field of `get_schedule` and
`get_free_slots`; the code filters the result after execution. There is no keyword
detection on the message any more.
"""

from collections.abc import Iterable
from enum import StrEnum
from typing import Protocol

from daily.domain.timeutil import add_minutes_to_iso, combine_date_and_time, parse_iso


class TimeBand(StrEnum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


def time_band_range(band: TimeBand) -> tuple[int, int]:
    """Range ``[start_hour, end_hour)``. "Night" does not cross midnight in v1."""
    return {
        TimeBand.MORNING: (6, 12),
        TimeBand.AFTERNOON: (12, 19),
        TimeBand.EVENING: (19, 23),
        TimeBand.NIGHT: (23, 24),
    }[band]


class HasStart(Protocol):
    @property
    def start(self) -> str: ...


def filter_by_band[T: HasStart](items: Iterable[T], band: TimeBand) -> list[T]:
    """Keeps only the items whose `start` falls inside the band."""
    start_hour, end_hour = time_band_range(band)
    return [item for item in items if start_hour <= parse_iso(item.start).hour < end_hour]


def band_bounds(date: str, band: TimeBand) -> tuple[str, str]:
    """Start and end (ISO) of the band on day `date`; for the night the end is midnight."""
    start_hour, end_hour = time_band_range(band)
    midnight = combine_date_and_time(date, "00:00")
    start = add_minutes_to_iso(midnight, start_hour * 60)
    return start, add_minutes_to_iso(midnight, end_hour * 60)
