from datetime import datetime, timedelta

import pytest

from daily.domain.patterns import NoteStatus
from daily.storage.notes_repos import (
    EventPatternsRepository,
    GapPatternsRepository,
    SequencePatternsRepository,
    TriggerDurationPatternsRepository,
)

AUGUST_1 = datetime(2026, 8, 1)


class TestEventPatterns:
    @pytest.fixture
    def repo(self, conn):
        return EventPatternsRepository(conn)

    def record(self, repo, *, title="yoga", start="19:00", duration=45, now=AUGUST_1):
        return repo.record_occurrence(
            title=title,
            observed_start=start,
            observed_duration_minutes=duration,
            time_tolerance_minutes=15,
            duration_tolerance_minutes=10,
            now=now,
        )

    def test_first_occurrence_creates_a_pending_row(self, repo):
        pattern = self.record(repo)

        assert pattern.occurrence_count == 1
        assert pattern.status is NoteStatus.PENDING
        assert pattern.typical_start == "19:00"
        assert pattern.typical_duration_minutes == 45

    def test_occurrence_within_tolerance_increments_without_moving_the_reference(self, repo):
        self.record(repo)

        second = self.record(repo, start="19:10", duration=50, now=datetime(2026, 8, 3))

        assert second.occurrence_count == 2
        assert second.typical_start == "19:00"  # the reference does not move
        assert second.typical_duration_minutes == 45

    def test_occurrence_out_of_tolerance_creates_a_separate_pattern(self, repo):
        self.record(repo)

        far_off = self.record(repo, start="07:00", now=datetime(2026, 8, 2))

        assert far_off.occurrence_count == 1
        assert far_off.typical_start == "07:00"

    def test_expired_window_resets_with_a_new_reference(self, repo):
        self.record(repo)

        after_window = self.record(repo, start="19:05", now=AUGUST_1 + timedelta(days=15))

        assert after_window.occurrence_count == 1
        assert after_window.typical_start == "19:05"

    def test_reset_for_expired_window_does_not_bring_a_rejected_back_to_pending(self, repo):
        first = self.record(repo)
        repo.set_status(first.note_id, NoteStatus.REJECTED)

        after_window = self.record(repo, now=AUGUST_1 + timedelta(days=15))

        assert after_window.status is NoteStatus.REJECTED  # "never propose again" is permanent

    def test_case_insensitive_title_hooks_the_same_pattern(self, repo):
        self.record(repo, title="Yoga")

        second = self.record(repo, title="yoga", now=datetime(2026, 8, 2))

        assert second.occurrence_count == 2

    def test_list_pending_excludes_confirmed_and_rejected(self, repo):
        pending = self.record(repo)
        rejected = self.record(repo, title="reading", start="22:30", duration=20)
        repo.set_status(rejected.note_id, NoteStatus.REJECTED)

        assert [p.note_id for p in repo.list_pending()] == [pending.note_id]


class TestGapPatterns:
    @pytest.fixture
    def repo(self, conn):
        return GapPatternsRepository(conn)

    def record(self, repo, *, related="work", position="before", gap=45, now=AUGUST_1):
        return repo.record_occurrence(
            related_title=related,
            position=position,
            observed_gap_minutes=gap,
            time_tolerance_minutes=15,
            now=now,
        )

    def test_first_occurrence_creates_a_pending_row(self, repo):
        pattern = self.record(repo)

        assert pattern.occurrence_count == 1
        assert pattern.status is NoteStatus.PENDING
        assert pattern.gap_minutes == 45

    def test_occurrence_within_tolerance_increments_without_moving_the_reference(self, repo):
        self.record(repo)

        second = self.record(repo, gap=50, now=datetime(2026, 8, 3))

        assert second.occurrence_count == 2
        assert second.gap_minutes == 45

    def test_different_position_does_not_hook_the_same_pattern(self, repo):
        self.record(repo, position="before")

        after_side = self.record(repo, position="after", now=datetime(2026, 8, 2))

        assert after_side.occurrence_count == 1

    def test_expired_window_resets(self, repo):
        self.record(repo)

        after_window = self.record(repo, gap=40, now=AUGUST_1 + timedelta(days=15))

        assert after_window.occurrence_count == 1
        assert after_window.gap_minutes == 40

    def test_list_pending_excludes_confirmed_and_rejected(self, repo):
        pending = self.record(repo)
        confirmed = self.record(repo, related="gym", position="after", gap=15)
        repo.set_status(confirmed.note_id, NoteStatus.CONFIRMED)

        assert [p.note_id for p in repo.list_pending()] == [pending.note_id]


class TestSequencePatterns:
    @pytest.fixture
    def repo(self, conn):
        return SequencePatternsRepository(conn)

    def record(self, repo, *, trigger, related, offset):
        return repo.record_occurrence(
            trigger_title=trigger,
            related_title=related,
            position="after",
            observed_offset_minutes=offset,
            time_tolerance_minutes=15,
            now=AUGUST_1,
        )

    def test_list_pending_excludes_confirmed_and_rejected(self, repo):
        pending = self.record(repo, trigger="stretching", related="gym", offset=5)
        rejected = self.record(repo, trigger="meditation", related="breakfast", offset=10)
        repo.set_status(rejected.note_id, NoteStatus.REJECTED)

        assert [p.note_id for p in repo.list_pending()] == [pending.note_id]

    def test_increments_within_tolerance_on_the_same_pair(self, repo):
        self.record(repo, trigger="stretching", related="gym", offset=5)

        second = self.record(repo, trigger="stretching", related="gym", offset=12)

        assert second.occurrence_count == 2
        assert second.offset_minutes == 5


class TestTriggerDurationPatterns:
    @pytest.fixture
    def repo(self, conn):
        return TriggerDurationPatternsRepository(conn)

    def record(self, repo, *, title, duration):
        return repo.record_occurrence(
            title=title,
            observed_duration_minutes=duration,
            duration_tolerance_minutes=10,
            now=AUGUST_1,
        )

    def test_list_pending_excludes_confirmed_and_rejected(self, repo):
        pending = self.record(repo, title="call with the client", duration=45)
        confirmed = self.record(repo, title="dentist", duration=30)
        repo.set_status(confirmed.note_id, NoteStatus.CONFIRMED)

        assert [p.note_id for p in repo.list_pending()] == [pending.note_id]

    def test_increments_within_duration_tolerance(self, repo):
        self.record(repo, title="dentist", duration=30)

        second = self.record(repo, title="dentist", duration=38)

        assert second.occurrence_count == 2
        assert second.typical_duration_minutes == 30
