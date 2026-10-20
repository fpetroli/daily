"""Value types shared by everything that receives data from outside: the API and the model's
tool calls.

Dates and times are local ISO strings, as in the services: no conversion. A regular
expression checks the shape and a real check verifies that the date or time exists (no
February 30, no hour 25): the code **validates** a value and never reinterprets it.
"""

import re
from datetime import datetime
from typing import Annotated

from pydantic import AfterValidator, Field, StringConstraints

# The patterns are also the grammar imposed on llama-server: its converter does not
# understand `\d` (it ignores it, and the model would write "13:30" where an ISO date is due)
# but it does apply `[0-9]`, groups and alternatives. That is why year (20xx), month, day, hour
# and minutes are spelled out in the pattern: without it, the model once wrote the year 2126.
_DAY = "20[0-9]{2}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"
_HOUR_MINUTE = "([01][0-9]|2[0-3]):[0-5][0-9]"
_ISO_MINUTE = re.compile(f"^{_DAY}T{_HOUR_MINUTE}$")
_ISO_DATE = re.compile(f"^{_DAY}$")
_TIME_OF_DAY = re.compile(f"^{_HOUR_MINUTE}$")


def _validate(pattern: re.Pattern[str], fmt: str, expected: str):
    def check(value: str) -> str:
        if not pattern.match(value):
            raise ValueError(f"expected {expected}")
        try:
            datetime.strptime(value, fmt)
        except ValueError:
            raise ValueError("no such date or time") from None
        return value

    return check


IsoMinute = Annotated[
    str,
    Field(pattern=_ISO_MINUTE.pattern, examples=["2026-09-21T18:00"]),
    AfterValidator(_validate(_ISO_MINUTE, "%Y-%m-%dT%H:%M", "YYYY-MM-DDTHH:MM")),
]
IsoDate = Annotated[
    str,
    Field(pattern=_ISO_DATE.pattern, examples=["2026-09-21"]),
    AfterValidator(_validate(_ISO_DATE, "%Y-%m-%d", "YYYY-MM-DD")),
]
TimeOfDay = Annotated[
    str,
    Field(pattern=_TIME_OF_DAY.pattern, examples=["18:00"]),
    AfterValidator(_validate(_TIME_OF_DAY, "%H:%M", "HH:MM")),
]

# Outputs are not validated again: an odd value in the database must not turn into a 500.
IsoMinuteOut = Annotated[
    str, Field(description="Local time `YYYY-MM-DDTHH:MM`.", examples=["2026-09-21T18:00"])
]

Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
DurationMinutes = Annotated[int, Field(gt=0, le=24 * 60)]
ShiftMinutes = Annotated[int, Field(ge=-7 * 24 * 60, le=7 * 24 * 60)]
