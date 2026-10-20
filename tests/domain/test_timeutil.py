from datetime import datetime

from daily.domain.timeutil import (
    add_minutes_to_iso,
    combine_date_and_time,
    minutes_between_iso,
    minutes_of_day,
    now_iso_string,
    time_of_day_from_iso,
    today_date_string,
)


def test_add_minutes_to_iso_simple():
    assert add_minutes_to_iso("2026-08-05T18:00", 90) == "2026-08-05T19:30"


def test_add_minutes_to_iso_crosses_midnight():
    assert add_minutes_to_iso("2026-08-05T23:30", 45) == "2026-08-06T00:15"


def test_add_minutes_to_iso_negative_and_month_change():
    assert add_minutes_to_iso("2026-09-01T00:10", -30) == "2026-08-31T23:40"


def test_combine_date_and_time():
    assert combine_date_and_time("2026-08-05", "18:00") == "2026-08-05T18:00"


def test_today_date_string_uses_the_given_instant():
    assert today_date_string(datetime(2026, 8, 5, 23, 59)) == "2026-08-05"


def test_now_iso_string_rounds_to_the_minute():
    assert now_iso_string(datetime(2026, 8, 5, 9, 7, 59)) == "2026-08-05T09:07"


def test_time_of_day_from_iso():
    assert time_of_day_from_iso("2026-08-05T18:30") == "18:30"


def test_minutes_between_iso_positive_and_negative():
    assert minutes_between_iso("2026-08-05T18:00", "2026-08-05T19:15") == 75
    assert minutes_between_iso("2026-08-05T19:15", "2026-08-05T18:00") == -75


def test_minutes_of_day():
    assert minutes_of_day("00:00") == 0
    assert minutes_of_day("07:05") == 425
    assert minutes_of_day("23:59") == 1439
