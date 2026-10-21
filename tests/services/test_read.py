"""Reads: the day's schedule, free slots, overlap check."""

from datetime import datetime

from daily.domain.timeband import TimeBand
from daily.services.schedule import check_overlap, get_free_slots, get_schedule
from daily.services.types import FreeSlot

from .conftest import DAY, NOW, add_event

# `now` fixed far from the test date: these tests check the gap computation in general, not
# the cut on "today" (tested separately): without an explicit `now`, if the test date matched
# the real one the cut would kick in unexpectedly.
FAR_PAST = datetime(2020, 1, 1)


def slots(repos, *, date=DAY, min_duration_minutes=None, time_band=None, now=FAR_PAST):
    return get_free_slots(
        repos,
        default_date=DAY,
        date=date,
        min_duration_minutes=min_duration_minutes,
        time_band=time_band,
        now=now,
    )


# --- get_free_slots --------------------------------------------------------------------


def test_empty_day_a_single_slot_from_start_to_end(repos):
    assert slots(repos) == [FreeSlot(f"{DAY}T00:00", "2026-08-06T00:00")]


def test_an_event_in_the_middle_gives_two_slots(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")

    assert slots(repos) == [
        FreeSlot(f"{DAY}T00:00", f"{DAY}T18:00"),
        FreeSlot(f"{DAY}T19:00", "2026-08-06T00:00"),
    ]


def test_overlapping_events_are_merged_before_computing_gaps(repos):
    add_event(repos, "a", f"{DAY}T10:00", f"{DAY}T12:00")
    add_event(repos, "b", f"{DAY}T11:00", f"{DAY}T13:00")

    result = slots(repos)

    assert [(s.start, s.end) for s in result] == [
        (f"{DAY}T00:00", f"{DAY}T10:00"),
        (f"{DAY}T13:00", "2026-08-06T00:00"),
    ]


def test_event_contained_in_another_does_not_widen_the_block(repos):
    add_event(repos, "long", f"{DAY}T10:00", f"{DAY}T14:00")
    add_event(repos, "short", f"{DAY}T11:00", f"{DAY}T12:00")

    result = slots(repos)

    assert [(s.start, s.end) for s in result] == [
        (f"{DAY}T00:00", f"{DAY}T10:00"),
        (f"{DAY}T14:00", "2026-08-06T00:00"),
    ]


def test_adjacent_events_leave_no_gaps(repos):
    add_event(repos, "a", f"{DAY}T10:00", f"{DAY}T11:00")
    add_event(repos, "b", f"{DAY}T11:00", f"{DAY}T12:00")

    assert len(slots(repos)) == 2


def test_minimum_duration_filters_too_short_slots(repos):
    add_event(repos, "a", f"{DAY}T09:00", f"{DAY}T09:50")
    add_event(repos, "b", f"{DAY}T10:00", f"{DAY}T23:00")

    result = slots(repos, min_duration_minutes=30)

    assert all(s.start != f"{DAY}T09:50" for s in result)


def test_missing_minimum_duration_is_zero(repos):
    add_event(repos, "a", f"{DAY}T09:00", f"{DAY}T09:50")
    add_event(repos, "b", f"{DAY}T10:00", f"{DAY}T23:00")

    assert any(s.start == f"{DAY}T09:50" for s in slots(repos, min_duration_minutes=None))


def test_if_the_date_is_today_the_first_slot_starts_at_current_time(repos):
    result = slots(repos, now=datetime(2026, 8, 5, 14, 0))

    assert result == [FreeSlot(f"{DAY}T14:00", "2026-08-06T00:00")]


def test_ongoing_event_the_slot_resumes_from_its_end(repos):
    add_event(repos, "meeting", f"{DAY}T13:30", f"{DAY}T15:00")

    result = slots(repos, now=datetime(2026, 8, 5, 14, 0))

    assert result == [FreeSlot(f"{DAY}T15:00", "2026-08-06T00:00")]


def test_date_other_than_today_no_cut_at_current_time(repos):
    result = get_free_slots(
        repos, default_date="2026-08-06", date=None, now=datetime(2026, 8, 5, 14, 0)
    )

    assert result[0].start == "2026-08-06T00:00"


def test_time_band_clips_slots_to_the_band(repos):
    add_event(repos, "work", f"{DAY}T09:00", f"{DAY}T17:00")  # free: 00-09 and 17-24

    def band(name, **kwargs):
        return [(s.start, s.end) for s in slots(repos, time_band=name, **kwargs)]

    assert band(TimeBand.MORNING) == [(f"{DAY}T06:00", f"{DAY}T09:00")]
    assert band(TimeBand.AFTERNOON) == [(f"{DAY}T17:00", f"{DAY}T19:00")]
    assert band(TimeBand.EVENING) == [(f"{DAY}T19:00", f"{DAY}T23:00")]
    assert band(TimeBand.NIGHT) == [(f"{DAY}T23:00", "2026-08-06T00:00")]


def test_time_band_without_gaps_in_the_band_empty_list(repos):
    add_event(repos, "work", f"{DAY}T06:00", f"{DAY}T12:00")

    assert slots(repos, time_band=TimeBand.MORNING) == []


def test_minimum_duration_applies_after_clipping_to_the_band(repos):
    # free from 17:00 on: 4 hours in the "evening" band (19-23), but 7 hours in all
    add_event(repos, "work", f"{DAY}T09:00", f"{DAY}T17:00")

    assert len(slots(repos, time_band=TimeBand.EVENING, min_duration_minutes=240)) == 1
    assert slots(repos, time_band=TimeBand.EVENING, min_duration_minutes=300) == []


def test_time_band_and_today_cut_combine(repos):
    result = slots(repos, time_band=TimeBand.EVENING, now=datetime(2026, 8, 5, 20, 30))

    assert result == [FreeSlot(f"{DAY}T20:30", f"{DAY}T23:00")]


# --- get_schedule ----------------------------------------------------------------------


def test_day_schedule_shows_title_times_and_state(repos):
    event = add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")

    [entry] = get_schedule(repos, default_date=DAY, date=DAY, now=NOW)

    # the ID stays in the service; the model's adapters remove it
    assert (entry.event_id, entry.title, entry.start, entry.end) == (
        event.event_id,
        "gym",
        f"{DAY}T18:00",
        f"{DAY}T19:00",
    )


def test_without_date_uses_the_default_date(repos):
    add_event(repos, "x", f"{DAY}T10:00", f"{DAY}T11:00")

    assert len(get_schedule(repos, default_date=DAY, date=None, now=NOW)) == 1


def test_empty_day_empty_list(repos):
    assert get_schedule(repos, default_date=DAY, date=DAY, now=NOW) == []


def test_done_is_derived_from_the_current_time(repos):
    add_event(repos, "morning", f"{DAY}T09:00", f"{DAY}T10:00")
    add_event(repos, "in corso", f"{DAY}T11:30", f"{DAY}T12:30")
    add_event(repos, "evening", f"{DAY}T20:00", f"{DAY}T21:00")

    entries = get_schedule(repos, default_date=DAY, now=NOW)  # NOW = 12:00

    assert [(e.title, e.done, e.started) for e in entries] == [
        ("morning", True, True),
        ("in corso", False, True),
        ("evening", False, False),
    ]


def test_an_event_ending_exactly_now_is_done(repos):
    add_event(repos, "just finished", f"{DAY}T11:00", f"{DAY}T12:00")

    [entry] = get_schedule(repos, default_date=DAY, now=NOW)

    assert entry.done


def test_schedule_is_ordered_by_start(repos):
    add_event(repos, "b", f"{DAY}T15:00", f"{DAY}T16:00")
    add_event(repos, "a", f"{DAY}T08:00", f"{DAY}T09:00")

    assert [e.title for e in get_schedule(repos, default_date=DAY, now=NOW)] == ["a", "b"]


def test_schedule_filters_by_time_band(repos):
    add_event(repos, "breakfast", f"{DAY}T08:00", f"{DAY}T08:30")
    add_event(repos, "meeting", f"{DAY}T15:00", f"{DAY}T16:00")
    add_event(repos, "dinner", f"{DAY}T20:00", f"{DAY}T21:00")

    def titles(band):
        return [e.title for e in get_schedule(repos, default_date=DAY, time_band=band, now=NOW)]

    assert titles(TimeBand.MORNING) == ["breakfast"]
    assert titles(TimeBand.AFTERNOON) == ["meeting"]
    assert titles(TimeBand.EVENING) == ["dinner"]
    assert titles(TimeBand.NIGHT) == []


# --- check_overlap ---------------------------------------------------------------------


def test_check_overlap_detects_overlaps(repos):
    add_event(repos, "meeting", f"{DAY}T10:00", f"{DAY}T11:00")

    result = check_overlap(repos, start=f"{DAY}T10:30", end=f"{DAY}T11:30")

    assert result.overlap
    assert [c.title for c in result.conflicts] == ["meeting"]


def test_check_overlap_adjacent_events_do_not_conflict(repos):
    add_event(repos, "meeting", f"{DAY}T10:00", f"{DAY}T11:00")

    result = check_overlap(repos, start=f"{DAY}T11:00", end=f"{DAY}T12:00")

    assert not result.overlap
    assert result.conflicts == ()
