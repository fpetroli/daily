"""Arithmetic on ISO datetimes ``YYYY-MM-DDTHH:MM``.

They are the user's local times, with no time zone and no seconds: the same format as the
values exchanged by tools and model. The code does the arithmetic, never the model.
"""

from datetime import datetime, timedelta

ISO_MINUTE_FORMAT = "%Y-%m-%dT%H:%M"


def parse_iso(iso_datetime: str) -> datetime:
    return datetime.fromisoformat(iso_datetime)


def _format_iso(moment: datetime) -> str:
    return moment.strftime(ISO_MINUTE_FORMAT)


def add_minutes_to_iso(iso_datetime: str, minutes: int) -> str:
    """Adds ``minutes`` (signed) to an ISO datetime, returning the same format."""
    return _format_iso(parse_iso(iso_datetime) + timedelta(minutes=minutes))


def combine_date_and_time(date: str, hhmm: str) -> str:
    """Combines a ``YYYY-MM-DD`` date with an ``HH:MM`` time."""
    return f"{date}T{hhmm}"


def today_date_string(now: datetime | None = None) -> str:
    """Today's date ``YYYY-MM-DD``: ``now`` always comes from the caller."""
    return (now or datetime.now()).strftime("%Y-%m-%d")


def now_iso_string(now: datetime | None = None) -> str:
    """Current instant rounded to the minute."""
    return _format_iso(now or datetime.now())


def time_of_day_from_iso(iso_datetime: str) -> str:
    """Only the ``HH:MM`` part of a full ISO datetime."""
    return iso_datetime.split("T")[1]


def minutes_between_iso(start_iso: str, end_iso: str) -> int:
    """Minutes between two ISO datetimes (``end - start``), truncated toward zero."""
    return int((parse_iso(end_iso) - parse_iso(start_iso)).total_seconds() / 60)


def minutes_of_day(hhmm: str) -> int:
    """Minutes since midnight of an ``HH:MM`` time."""
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)
