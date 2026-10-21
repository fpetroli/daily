from datetime import datetime, timedelta

from daily.domain.patterns import NoteStatus
from daily.services.habits import (
    EventPatternProposal,
    GapPatternProposal,
    SequencePatternProposal,
    dismiss_habit,
    mark_proposed,
    next_habit_proposal,
    record_habit_signals,
)

from .conftest import DAY, NOW, add_event


def track(repos, event, *, derived_from_rule_id=None, now=NOW):
    record_habit_signals(
        repos, added_event=event, derived_from_rule_id=derived_from_rule_id, now=now
    )


# --- record_habit_signals --------------------------------------------------------------


def test_two_adjacent_events_without_rule_track_sequence_with_previous(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    shower = add_event(repos, "shower", f"{DAY}T19:00", f"{DAY}T19:15")

    track(repos, shower)

    [pattern] = repos.sequence_patterns.list_pending()
    assert (pattern.trigger_title, pattern.related_title) == ("shower", "gym")
    assert (pattern.position, pattern.offset_minutes, pattern.occurrence_count) == ("after", 0, 1)


def test_pair_already_ruled_does_not_track_sequence(repos):
    add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )
    shower = add_event(repos, "shower", f"{DAY}T19:00", f"{DAY}T19:15")

    track(repos, shower)

    assert repos.sequence_patterns.list_pending() == []


def test_always_tracks_gap_on_neighbour_regardless_of_rules(repos):
    add_event(repos, "work", f"{DAY}T09:00", f"{DAY}T17:00")
    dinner = add_event(repos, "dinner", f"{DAY}T19:00", f"{DAY}T20:00")

    track(repos, dinner)

    [pattern] = repos.gap_patterns.list_pending()
    assert (pattern.related_title, pattern.position, pattern.gap_minutes) == (
        "work",
        "after",
        120,  # 17:00 -> 19:00
    )


def test_next_neighbour_records_gap_and_sequence_with_added_event_as_anchor(repos):
    gym = add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    add_event(repos, "shower", f"{DAY}T19:10", f"{DAY}T19:25")

    track(repos, gym)

    [gap] = repos.gap_patterns.list_pending()
    assert (gap.related_title, gap.gap_minutes) == ("gym", 10)
    [sequence] = repos.sequence_patterns.list_pending()
    assert (sequence.trigger_title, sequence.related_title, sequence.offset_minutes) == (
        "shower",
        "gym",
        10,
    )


def test_title_with_complete_usual_time_does_not_track_repeated_event(repos):
    repos.event_defaults.upsert("gym", typical_start="18:00", typical_duration_minutes=60)
    event = add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")

    track(repos, event)

    assert repos.event_patterns.list_pending() == []


def test_title_with_partial_usual_time_tracks_anyway(repos):
    repos.event_defaults.upsert("gym", typical_start="18:00")  # missing duration
    event = add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")

    track(repos, event)

    [pattern] = repos.event_patterns.list_pending()
    assert (pattern.typical_start, pattern.typical_duration_minutes) == ("18:00", 60)


def test_no_neighbour_on_the_same_day_tracks_no_adjacency(repos):
    event = add_event(repos, "standalone", f"{DAY}T10:00", f"{DAY}T11:00")

    track(repos, event)

    assert repos.sequence_patterns.list_pending() == []
    assert repos.gap_patterns.list_pending() == []


def test_event_born_from_a_rule_does_not_track_repeated_event(repos):
    event = add_event(repos, "shower", f"{DAY}T19:00", f"{DAY}T19:15")

    track(repos, event, derived_from_rule_id="a-rule")

    assert repos.event_patterns.list_pending() == []


def test_overlapping_events_are_not_adjacent(repos):
    add_event(repos, "meeting", f"{DAY}T10:00", f"{DAY}T11:00")
    event = add_event(repos, "call", f"{DAY}T10:30", f"{DAY}T11:30")

    track(repos, event)

    assert repos.gap_patterns.list_pending() == []


def test_only_the_immediate_neighbour_counts(repos):
    add_event(repos, "work", f"{DAY}T09:00", f"{DAY}T12:00")
    add_event(repos, "lunch", f"{DAY}T12:00", f"{DAY}T13:00")
    rest = add_event(repos, "rest", f"{DAY}T14:00", f"{DAY}T15:00")

    track(repos, rest)

    [gap] = repos.gap_patterns.list_pending()
    assert gap.related_title == "lunch"  # not "work"


def test_second_occurrence_within_tolerance_increments_the_counter(repos):
    gym = add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    track(repos, gym)
    gym2 = add_event(repos, "gym", "2026-08-06T18:05", "2026-08-06T19:05")

    track(repos, gym2, now=NOW + timedelta(days=1))

    [pattern] = repos.event_patterns.list_pending()
    assert pattern.occurrence_count == 2


# --- next_habit_proposal / dismiss_habit ---------------------------------------------


def seed_event_pattern(repos, *, occurrences, title="gym"):
    for day in range(occurrences):
        date = (datetime(2026, 8, 1) + timedelta(days=day)).strftime("%Y-%m-%d")
        repos.event_patterns.record_occurrence(
            title=title,
            observed_start="18:00",
            observed_duration_minutes=60,
            time_tolerance_minutes=15,
            duration_tolerance_minutes=10,
            now=datetime.fromisoformat(f"{date}T12:00"),
        )


def test_no_proposal_below_threshold(repos):
    seed_event_pattern(repos, occurrences=2)  # default threshold: 3

    assert next_habit_proposal(repos) is None


def test_repeated_event_proposal_at_threshold(repos):
    seed_event_pattern(repos, occurrences=3)

    proposal = next_habit_proposal(repos)

    assert isinstance(proposal, EventPatternProposal)
    assert (proposal.title, proposal.typical_start, proposal.typical_duration_minutes) == (
        "gym",
        "18:00",
        60,
    )


def test_threshold_is_configurable(repos):
    repos.app_settings.set_int("repetition_threshold", 2)
    seed_event_pattern(repos, occurrences=2)

    assert isinstance(next_habit_proposal(repos), EventPatternProposal)


def record_sequence(repos, times):
    for day in range(times):
        repos.sequence_patterns.record_occurrence(
            trigger_title="shower",
            related_title="gym",
            position="after",
            observed_offset_minutes=0,
            time_tolerance_minutes=15,
            now=datetime(2026, 8, 1 + day, 12, 0),
        )


def record_gap(repos, times):
    for day in range(times):
        repos.gap_patterns.record_occurrence(
            related_title="work",
            position="after",
            observed_gap_minutes=30,
            time_tolerance_minutes=15,
            now=datetime(2026, 8, 1 + day, 12, 0),
        )


def test_proposal_order_event_then_sequence_then_gap(repos):
    record_gap(repos, 3)
    record_sequence(repos, 3)
    seed_event_pattern(repos, occurrences=3)

    first = next_habit_proposal(repos)
    assert isinstance(first, EventPatternProposal)
    dismiss_habit(repos, first)

    second = next_habit_proposal(repos)
    assert isinstance(second, SequencePatternProposal)
    assert (second.trigger_title, second.related_title, second.position) == (
        "shower",
        "gym",
        "after",
    )
    dismiss_habit(repos, second)

    third = next_habit_proposal(repos)
    assert isinstance(third, GapPatternProposal)
    assert (third.related_title, third.gap_minutes) == ("work", 30)
    dismiss_habit(repos, third)

    assert next_habit_proposal(repos) is None


def test_dismiss_rejects_the_current_proposal_and_it_is_not_proposed_again(repos):
    seed_event_pattern(repos, occurrences=3)
    shown = next_habit_proposal(repos)

    dismissed = dismiss_habit(repos, shown)

    assert dismissed == shown
    assert next_habit_proposal(repos) is None
    # a new similar event does not bring the pattern back to "pending"
    seed_event_pattern(repos, occurrences=1)
    assert next_habit_proposal(repos) is None


def test_dismiss_without_a_shown_proposal_does_nothing(repos):
    seed_event_pattern(repos, occurrences=3)

    assert dismiss_habit(repos, None) is None
    assert isinstance(next_habit_proposal(repos), EventPatternProposal)


def test_a_shown_proposal_comes_back_only_after_the_threshold_again(repos):
    seed_event_pattern(repos, occurrences=3)
    mark_proposed(repos, next_habit_proposal(repos))

    assert next_habit_proposal(repos) is None
    seed_event_pattern(repos, occurrences=2)
    assert next_habit_proposal(repos) is None
    seed_event_pattern(repos, occurrences=1)
    assert isinstance(next_habit_proposal(repos), EventPatternProposal)


def test_mark_proposed_restarts_the_counter_of_each_kind(repos):
    record_sequence(repos, 3)
    record_gap(repos, 3)

    for _ in range(2):
        mark_proposed(repos, next_habit_proposal(repos))

    assert next_habit_proposal(repos) is None
    assert len(repos.sequence_patterns.list_pending()) == 1  # still pending, not rejected
    assert len(repos.gap_patterns.list_pending()) == 1


def test_a_rejected_pattern_has_state_rejected(repos):
    seed_event_pattern(repos, occurrences=3)
    [pending] = repos.event_patterns.list_pending()

    dismiss_habit(repos, next_habit_proposal(repos))

    assert repos.event_patterns.list_pending() == []
    assert pending.status == NoteStatus.PENDING  # it was pending before the rejection
