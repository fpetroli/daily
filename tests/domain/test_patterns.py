from datetime import datetime, timedelta

from daily.domain.patterns import NoteStatus, WindowState, next_window_state

START = datetime(2026, 8, 1, 9, 0)


def test_no_existing_state_starts_at_one():
    state = next_window_state(existing_count=None, existing_first_occurrence_at=None, now=START)

    assert state == WindowState(1, START)


def test_inside_the_window_increments_and_keeps_first_occurrence():
    state = next_window_state(
        existing_count=2, existing_first_occurrence_at=START, now=START + timedelta(days=5)
    )

    assert state == WindowState(3, START)


def test_exactly_14_days_does_not_reset():
    state = next_window_state(
        existing_count=2, existing_first_occurrence_at=START, now=START + timedelta(days=14)
    )

    assert state.occurrence_count == 3


def test_beyond_14_days_resets_with_the_new_first_occurrence():
    now = START + timedelta(days=14, seconds=1)

    state = next_window_state(existing_count=2, existing_first_occurrence_at=START, now=now)

    assert state == WindowState(1, now)


def test_note_status_has_the_column_values():
    assert [s.value for s in NoteStatus] == ["pending", "confirmed", "rejected"]
    assert NoteStatus("rejected") is NoteStatus.REJECTED
