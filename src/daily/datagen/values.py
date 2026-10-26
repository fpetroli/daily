"""Values and the ways a user says them: days, times, durations, titles.

Every function returns the words **and** the value they mean, computed the way the prompt
tells the model to read them (a weekday is its first occurrence from tomorrow, an absolute
date has the current year, today is an omitted date).
"""

import random
from datetime import date, datetime, timedelta

from daily.agent.prompts import WEEKDAYS

MONTHS = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
]  # fmt: skip
NUMBER_WORDS = {
    2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve",
}  # fmt: skip


def iso(day: date) -> str:
    return f"{day:%Y-%m-%d}"


def hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def weekday_name(rng: random.Random, weekday: int) -> str:
    name = WEEKDAYS[weekday]
    return name if rng.random() < 0.5 else name.lower()


# --- days --------------------------------------------------------------------------------


def say_day(rng: random.Random, now: datetime, *, today: float = 0.3) -> tuple[str, str | None]:
    """A day the user may name, as "on Friday", "tomorrow", "in a week", "on October 12".

    Today is the empty string or "today" and its value is `None`: the call omits the date."""
    roll = rng.random()
    if roll < today:
        return rng.choice(["", "", "today"]), None
    offset_roll = rng.random()
    if offset_roll < 0.3:
        return "tomorrow", iso(now + timedelta(days=1))
    if offset_roll < 0.38:
        return "the day after tomorrow", iso(now + timedelta(days=2))
    if offset_roll < 0.7:
        offset = rng.randint(1, 7)
        day = now + timedelta(days=offset)
        name = weekday_name(rng, day.weekday())
        return rng.choice([f"on {name}", name, f"next {name}"]), iso(day)
    if offset_roll < 0.82:
        offset = rng.choice([3, 4, 5, 6, 7, 7, 8, 10, 14])
        if offset == 7:
            text = "in a week"
        elif offset == 14:
            text = rng.choice(["in two weeks", "in 14 days"])
        else:
            text = f"in {rng.choice([str(offset), NUMBER_WORDS.get(offset, str(offset))])} days"
        return text, iso(now + timedelta(days=offset))
    return say_absolute_date(rng, now)


def say_absolute_date(rng: random.Random, now: datetime) -> tuple[str, str | None]:
    """A date with the month, always in the current year and at least two days ahead."""
    for _ in range(20):
        day = (now + timedelta(days=rng.randint(2, 60))).date()
        if day.year == now.year:
            break
    else:  # end of December: nothing ahead in the same year
        return "tomorrow", iso(now + timedelta(days=1))
    month = MONTHS[day.month - 1]
    suffix = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd", 31: "st"}.get(day.day, "th")
    text = rng.choice(
        [
            f"on {month} {day.day}",
            f"on {day.day} {month}",
            f"on {month} {day.day}{suffix}",
            f"on the {day.day}{suffix} of {month}",
        ]
    )
    return text, iso(day)


def say_day_possessive(rng: random.Random, now: datetime) -> tuple[str, str | None]:
    """The day an event is on now, as a possessive: "tomorrow's", "Friday's", "today's"."""
    roll = rng.random()
    if roll < 0.25:
        return rng.choice(["today's", "tonight's"]), None
    if roll < 0.6:
        return "tomorrow's", iso(now + timedelta(days=1))
    day = now + timedelta(days=rng.randint(2, 7))
    return f"{weekday_name(rng, day.weekday())}'s", iso(day)


# --- times -------------------------------------------------------------------------------

_MINUTES = [0, 0, 0, 0, 0, 15, 30, 30, 30, 45, 10, 20, 40, 50, 5]


def pick_time(rng: random.Random, first: int = 7, last: int = 22) -> int:
    """A start time in minutes from midnight."""
    return rng.randint(first, last) * 60 + rng.choice(_MINUTES)


def _twelve(hour: int) -> int:
    return hour - 12 if hour > 12 else hour


def say_time(rng: random.Random, minutes: int) -> str:
    """A time with "at": "at 18:30", "at 6:30 pm", "at 9", "at noon", "at half past 8 in the
    evening". Never an hour that could be morning or afternoon (no bare "at 6")."""
    hour, minute = divmod(minutes, 60)
    forms = [f"at {hour:02d}:{minute:02d}" if hour < 10 else f"at {hour}:{minute:02d}"]
    if minute == 0 and hour >= 13:
        forms.append(f"at {hour}")
    am_pm = "pm" if hour >= 12 else "am"
    clock = f"{_twelve(hour)}" if minute == 0 else f"{_twelve(hour)}:{minute:02d}"
    forms += [f"at {clock} {am_pm}", f"at {clock}{am_pm}"]
    if 7 <= hour <= 11:  # a bare hour from 7 to 11 is the morning
        forms.append(f"at {clock}")
        if minute == 0:
            forms.append(f"at {hour} in the morning")
        if minute == 30:
            forms.append(f"at half past {hour}")
        if minute == 15:
            forms.append(f"at a quarter past {hour}")
    if hour == 12 and minute == 0:
        forms += ["at noon", "at midday", "at 12"]
    if 13 <= hour <= 17 and minute == 0:
        forms.append(f"at {_twelve(hour)} in the afternoon")
    if 18 <= hour <= 22:
        if minute == 0:
            forms.append(f"at {_twelve(hour)} in the evening")
        if minute == 30:
            forms.append(f"at half past {_twelve(hour)} in the evening")
    if minute == 45 and 17 <= hour <= 21:
        forms.append(f"at a quarter to {_twelve(hour + 1)} in the evening")
    return rng.choice(forms)


def say_clock(rng: random.Random, minutes: int) -> str:
    """A time without "at", for ranges and "until": "18:30", "6:30 pm", "noon"."""
    return say_time(rng, minutes).removeprefix("at ")


def say_range(rng: random.Random, start: int, end: int) -> str:
    """A range, "from 16:00 to 17:30", "from 4 to 5 pm", "between 9 and 11"."""
    style = rng.random()
    if style < 0.35:
        return f"from {hhmm(start)} to {hhmm(end)}"
    if style < 0.6 and start >= 13 * 60 and end <= 23 * 60:
        return f"from {_short(start)} to {_short(end)} pm"
    if style < 0.75 and 7 * 60 <= start and end <= 11 * 60 + 59:
        return f"from {_short(start)} to {_short(end)}"
    lead = rng.choice(["from", "between"])
    joint = "to" if lead == "from" else "and"
    return f"{lead} {say_clock(rng, start)} {joint} {say_clock(rng, end)}"


def _short(minutes: int) -> str:
    hour, minute = divmod(minutes, 60)
    return f"{_twelve(hour)}" if minute == 0 else f"{_twelve(hour)}:{minute:02d}"


# --- durations ---------------------------------------------------------------------------

DURATION_WORDS: dict[int, list[str]] = {
    5: ["5 minutes", "five minutes"],
    10: ["10 minutes", "ten minutes", "10 min"],
    15: ["15 minutes", "a quarter of an hour", "15 min"],
    20: ["20 minutes", "twenty minutes", "20 min"],
    25: ["25 minutes"],
    30: ["half an hour", "30 minutes", "30 min", "half hour"],
    40: ["40 minutes", "forty minutes"],
    45: ["45 minutes", "three quarters of an hour", "45 min"],
    50: ["50 minutes"],
    60: ["an hour", "one hour", "1 hour", "60 minutes", "1h"],
    75: ["an hour and a quarter", "75 minutes", "1 hour and 15 minutes"],
    90: ["an hour and a half", "90 minutes", "1 hour and 30 minutes", "1.5 hours"],
    120: ["two hours", "2 hours", "2h"],
    150: ["two and a half hours", "2 hours and 30 minutes", "150 minutes"],
    180: ["three hours", "3 hours"],
    240: ["four hours", "4 hours"],
}
EVENT_DURATIONS = [15, 20, 30, 30, 45, 45, 60, 60, 60, 75, 90, 90, 120, 120, 150, 180, 240]
SHIFTS = [5, 10, 15, 15, 20, 30, 30, 45, 60, 60, 90, 120]


def say_duration(rng: random.Random, minutes: int) -> str:
    return rng.choice(DURATION_WORDS[minutes])


# --- titles ------------------------------------------------------------------------------

# None of these may match a title of the eval cases: `titles()` removes any that does.
TITLES = [
    "piano lesson", "haircut", "swim", "laundry", "book club", "physio", "grocery run",
    "coffee with Anna", "vet appointment", "car service", "Spanish class", "standup",
    "budget review", "parents' evening", "football practice", "bike ride", "pottery class",
    "tax appointment", "lunch with Giulia", "school pickup", "pharmacy", "dog walk",
    "meditation", "climbing", "tennis", "choir rehearsal", "job interview", "date night",
    "board game night", "bank appointment", "dry cleaning", "study session", "code review",
    "client demo", "pilates", "boxing class", "violin practice", "farmers market",
    "brunch with Tom", "eye exam", "hairdresser", "plumber visit", "museum visit", "cinema",
    "concert", "reading", "nap", "language exchange", "design review", "one-on-one with Elena",
    "volunteering", "massage", "gardening", "cooking class", "post office", "sprint planning",
    "retrospective", "wine tasting", "birthday party", "train to Turin", "flight to Paris",
    "physics homework", "guitar lesson", "swimming lesson", "parent-teacher meeting",
    "stretching", "jogging", "skype with grandma", "laundromat", "car wash", "oil change",
    "chess club", "drawing class", "podcast recording", "photo shoot", "sales pitch",
    "quarterly review", "onboarding session", "webinar", "hackathon", "wedding rehearsal",
    "baby shower", "housewarming", "barbecue", "picnic", "hiking", "spin class", "zumba",
    "padel", "squash", "volleyball", "basketball", "rowing", "ironing", "vacuuming",
    "bill payments", "vaccination",
    "blood test", "orthodontist", "therapy session", "nail salon", "barber", "library",
    "French lesson", "math tutoring", "thesis writing", "lab meeting", "office hours",
    "conference call", "interview prep", "product launch", "marketing sync", "invoice review",
    "inventory check", "warehouse visit", "site inspection", "dance class", "karate",
    "aperitivo with Chiara", "tea with Omar", "movie night", "game night", "family lunch",
]  # fmt: skip


def titles(excluded: set[str]) -> list[str]:
    """The title pool without the eval titles, nor any that contains one as a word or is
    contained in one: "dinner with Anna" goes if the cases have "dinner"."""
    lowered = {title.lower() for title in excluded}

    def clashes(title: str) -> bool:
        low = f" {title.lower()} "
        return any(f" {other} " in low or f" {low.strip()} " in f" {other} " for other in lowered)

    return [title for title in TITLES if not clashes(title)]


def with_article(rng: random.Random, title: str, *, indefinite: bool = True) -> str:
    """The title as the user writes it: bare, "the …", sometimes "a …"."""
    if title[0].isupper() or "'" in title.split()[0]:
        return title
    choices = [title, f"the {title}"]
    if indefinite:
        choices.append(f"{'an' if title[0] in 'aeiou' else 'a'} {title}")
    return rng.choice(choices)
