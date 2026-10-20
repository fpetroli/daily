from dataclasses import dataclass

from daily.domain.timeband import TimeBand, band_bounds, filter_by_band, time_band_range


@dataclass(frozen=True)
class Item:
    title: str
    start: str


def test_band_ranges():
    assert time_band_range(TimeBand.MORNING) == (6, 12)
    assert time_band_range(TimeBand.AFTERNOON) == (12, 19)
    assert time_band_range(TimeBand.EVENING) == (19, 23)
    assert time_band_range(TimeBand.NIGHT) == (23, 24)


def test_values_are_the_words_used_by_the_model():
    assert {band.value for band in TimeBand} == {"morning", "afternoon", "evening", "night"}


def test_filter_keeps_only_items_inside_the_band():
    items = [
        Item("breakfast", "2026-08-05T08:00"),
        Item("lunch", "2026-08-05T13:00"),
        Item("dinner", "2026-08-05T20:00"),
    ]

    afternoon = filter_by_band(items, TimeBand.AFTERNOON)

    assert [i.title for i in afternoon] == ["lunch"]


def test_a_slot_counts_from_its_start_time():
    slots = [Item("free", "2026-08-05T00:00"), Item("free", "2026-08-05T14:00")]

    assert filter_by_band(slots, TimeBand.MORNING) == []  # 00:00 is outside 06-12
    assert len(filter_by_band(slots, TimeBand.AFTERNOON)) == 1


def test_bounds_are_start_included_end_excluded():
    items = [Item("a", "2026-08-05T12:00"), Item("b", "2026-08-05T19:00")]

    assert [i.title for i in filter_by_band(items, TimeBand.AFTERNOON)] == ["a"]
    assert [i.title for i in filter_by_band(items, TimeBand.EVENING)] == ["b"]


def test_band_bounds_on_a_day():
    assert band_bounds("2026-08-05", TimeBand.MORNING) == ("2026-08-05T06:00", "2026-08-05T12:00")
    assert band_bounds("2026-08-05", TimeBand.EVENING) == ("2026-08-05T19:00", "2026-08-05T23:00")


def test_night_ends_at_midnight_of_the_next_day():
    assert band_bounds("2026-08-05", TimeBand.NIGHT) == ("2026-08-05T23:00", "2026-08-06T00:00")
