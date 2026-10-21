from datetime import datetime

from daily.domain.patterns import NoteStatus
from daily.services.rules import (
    AddEventRuleInvalidFormat,
    AddEventRuleNeedsInfo,
    AddEventRuleNeedsPosition,
    AddEventRuleSuccess,
    RemoveEventRuleAmbiguous,
    RemoveEventRuleNotFound,
    RemoveEventRuleSuccess,
    SavedGap,
    UsualTimes,
    add_event_rule,
    remove_event_rule,
    remove_rule_target,
)


def add_rule(repos, trigger_title, **kwargs):
    return add_event_rule(repos, trigger_title=trigger_title, **kwargs)


def remove_rule(repos, trigger_title, **kwargs):
    return remove_event_rule(repos, trigger_title=trigger_title, **kwargs)


# --- add_event_rule --------------------------------------------------------------------


def test_without_related_title_writes_the_usual_time(repos):
    outcome = add_rule(repos, "meeting", typical_start="09:00", typical_duration_minutes=30)

    assert outcome == AddEventRuleSuccess()
    saved = repos.event_defaults.get_by_title("meeting")
    assert (saved.typical_start, saved.typical_duration_minutes) == ("09:00", 30)


def test_with_related_title_writes_a_relational_rule(repos):
    outcome = add_rule(repos, "shower", related_title="gym", position="after", offset_minutes=0)

    assert outcome == AddEventRuleSuccess()
    [rule] = repos.event_rules.find_by_trigger("shower")
    assert rule.related_title == "gym"


def test_gap_minutes_writes_a_gap_row(repos):
    add_rule(repos, "commute", related_title="work", position="before", gap_minutes=45)

    [rule] = repos.event_rules.find_by_trigger("commute")
    assert rule.is_gap
    assert rule.gap_minutes == 45


def test_related_title_without_position_asks_position_without_writing(repos):
    outcome = add_rule(repos, "shower", related_title="gym", offset_minutes=5)

    assert outcome == AddEventRuleNeedsPosition("shower", "gym", offset_minutes=5)
    assert repos.event_rules.find_by_trigger("shower") == []


def test_called_again_for_the_same_pair_updates_instead_of_duplicating(repos):
    add_rule(repos, "shower", related_title="gym", position="after", offset_minutes=0)
    add_rule(repos, "shower", related_title="gym", position="after", offset_minutes=10)

    [rule] = repos.event_rules.find_by_trigger("shower")
    assert rule.offset_minutes == 10


def test_update_with_omitted_field_does_not_clear_it(repos):
    add_rule(repos, "shower", related_title="gym", position="after", offset_minutes=5)
    add_rule(repos, "shower", related_title="gym", position="before")

    [rule] = repos.event_rules.find_by_trigger("shower")
    assert (rule.position, rule.offset_minutes) == ("before", 5)


def test_typical_start_iso_datetime_is_normalized_to_hhmm(repos):
    outcome = add_rule(repos, "breakfast", typical_start="2026-08-21T11:00")

    assert outcome == AddEventRuleSuccess()
    assert repos.event_defaults.get_by_title("breakfast").typical_start == "11:00"


def test_unrecognizable_typical_start_is_invalid_format_and_writes_nothing(repos):
    outcome = add_rule(repos, "breakfast", typical_start="garbage")

    assert outcome == AddEventRuleInvalidFormat()
    assert repos.event_defaults.get_by_title("breakfast") is None


def test_out_of_range_typical_start_is_invalid_format(repos):
    assert add_rule(repos, "breakfast", typical_start="25:00") == AddEventRuleInvalidFormat()
    assert add_rule(repos, "breakfast", typical_start="9:00") == AddEventRuleInvalidFormat()


def test_duration_alone_is_valid(repos):
    outcome = add_rule(repos, "breakfast", typical_duration_minutes=30)

    assert outcome == AddEventRuleSuccess()
    assert repos.event_defaults.get_by_title("breakfast").typical_duration_minutes == 30


def test_typical_start_at_the_valid_limit(repos):
    assert add_rule(repos, "breakfast", typical_start="23:59") == AddEventRuleSuccess()


def test_usual_time_update_does_not_clear_the_other_field(repos):
    add_rule(repos, "meeting", typical_start="09:00", typical_duration_minutes=30)
    add_rule(repos, "meeting", typical_start="10:00")

    saved = repos.event_defaults.get_by_title("meeting")
    assert (saved.typical_start, saved.typical_duration_minutes) == ("10:00", 30)


def test_without_related_title_time_or_duration_there_is_nothing_to_store(repos):
    outcome = add_rule(repos, "meeting")

    assert outcome == AddEventRuleNeedsInfo("meeting")
    assert repos.event_defaults.get_by_title("meeting") is None


# --- remove_event_rule -----------------------------------------------------------------


def test_remove_with_related_title_deletes_the_rule(repos):
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )

    outcome = remove_rule(repos, "shower", related_title="gym")

    assert isinstance(outcome, RemoveEventRuleSuccess)
    assert outcome.removed.trigger_title == "shower"
    assert repos.event_rules.find_by_trigger("shower") == []


def test_remove_finds_the_rule_said_the_other_way_round(repos):
    repos.event_rules.insert_rule(
        trigger_title="gym", related_title="breakfast", position="before", offset_minutes=0
    )

    outcome = remove_rule(repos, "breakfast", related_title="gym")

    assert isinstance(outcome, RemoveEventRuleSuccess)
    assert repos.event_rules.get_all() == []


def test_remove_with_related_title_without_match_not_found(repos):
    assert remove_rule(repos, "shower", related_title="gym") == RemoveEventRuleNotFound()


def test_remove_without_related_title_deletes_the_whole_defaults_row(repos):
    repos.event_defaults.upsert("meeting", typical_start="09:00", typical_duration_minutes=30)

    outcome = remove_rule(repos, "meeting")

    assert outcome == RemoveEventRuleSuccess(UsualTimes("meeting"))
    assert repos.event_defaults.get_by_title("meeting") is None


def test_remove_with_one_title_deletes_its_only_rule_on_either_side(repos):
    repos.event_rules.insert_rule(
        trigger_title="breakfast", related_title="gym", position="after", offset_minutes=0
    )

    outcome = remove_rule(repos, "gym")

    assert isinstance(outcome, RemoveEventRuleSuccess)
    assert repos.event_rules.get_all() == []


def test_remove_with_one_title_and_several_things_asks_which(repos):
    repos.event_defaults.upsert("gym", typical_start="07:00", typical_duration_minutes=90)
    repos.event_rules.insert_rule(
        trigger_title="gym", related_title="breakfast", position="before", offset_minutes=0
    )

    outcome = remove_rule(repos, "gym")

    assert isinstance(outcome, RemoveEventRuleAmbiguous)
    assert outcome.candidates[0] == UsualTimes("gym")
    assert outcome.candidates[1].related_title == "breakfast"
    assert repos.event_defaults.get_by_title("gym") is not None  # nothing removed yet
    assert len(repos.event_rules.get_all()) == 1


def test_remove_rule_target_deletes_exactly_the_choice(repos):
    repos.event_defaults.upsert("gym", typical_start="07:00", typical_duration_minutes=90)
    rule_id = repos.event_rules.insert_rule(
        trigger_title="gym", related_title="breakfast", position="before", offset_minutes=0
    )
    rule = repos.event_rules.get_by_id(rule_id)

    assert remove_rule_target(repos, rule) == RemoveEventRuleSuccess(rule)
    assert repos.event_rules.get_all() == []
    assert repos.event_defaults.get_by_title("gym") is not None
    assert remove_rule_target(repos, rule) == RemoveEventRuleNotFound()


def test_remove_with_field_clears_only_that_field(repos):
    repos.event_defaults.upsert("meeting", typical_start="09:00", typical_duration_minutes=30)

    outcome = remove_rule(repos, "meeting", field="typical_duration_minutes")

    assert outcome == RemoveEventRuleSuccess(UsualTimes("meeting", "typical_duration_minutes"))
    saved = repos.event_defaults.get_by_title("meeting")
    assert (saved.typical_start, saved.typical_duration_minutes) == ("09:00", None)


def test_remove_without_defaults_row_not_found(repos):
    assert remove_rule(repos, "missing") == RemoveEventRuleNotFound()


def test_remove_several_rows_for_the_same_pair_is_ambiguous(repos):
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="before", offset_minutes=5
    )

    outcome = remove_rule(repos, "shower", related_title="gym")

    assert isinstance(outcome, RemoveEventRuleAmbiguous)
    assert len(outcome.candidates) == 2
    assert {c.trigger_title for c in outcome.candidates} == {"shower"}


# --- automatic confirmation of patterns (no pending state) -----------------------------


def observe_event_pattern(repos, title="gym"):
    return repos.event_patterns.record_occurrence(
        title=title,
        observed_start="18:00",
        observed_duration_minutes=60,
        time_tolerance_minutes=15,
        duration_tolerance_minutes=10,
        now=datetime(2026, 8, 1, 12, 0),
    )


def observe_sequence(repos):
    return repos.sequence_patterns.record_occurrence(
        trigger_title="shower",
        related_title="gym",
        position="after",
        observed_offset_minutes=0,
        time_tolerance_minutes=15,
        now=datetime(2026, 8, 1, 12, 0),
    )


def observe_gap(repos, gap=30):
    return repos.gap_patterns.record_occurrence(
        related_title="work",
        position="after",
        observed_gap_minutes=gap,
        time_tolerance_minutes=15,
        now=datetime(2026, 8, 1, 12, 0),
    )


def status_of(pending_list):
    return [p.status for p in pending_list]


def test_complete_usual_time_closes_the_pattern_with_the_same_title(repos):
    observe_event_pattern(repos)

    add_rule(repos, "Gym", typical_start="18:00", typical_duration_minutes=60)

    assert repos.event_patterns.list_pending() == []


def test_partial_usual_time_leaves_the_pattern_pending(repos):
    observe_event_pattern(repos)

    add_rule(repos, "gym", typical_start="18:00")

    assert len(repos.event_patterns.list_pending()) == 1


def test_usual_time_of_another_title_does_not_close_the_pattern(repos):
    observe_event_pattern(repos)

    add_rule(repos, "yoga", typical_start="19:00", typical_duration_minutes=45)

    assert len(repos.event_patterns.list_pending()) == 1


def test_relational_rule_closes_the_pairs_sequence_pattern(repos):
    observe_sequence(repos)

    add_rule(repos, "shower", related_title="gym", position="after", offset_minutes=0)

    assert repos.sequence_patterns.list_pending() == []


def test_rule_of_another_pair_does_not_close_the_sequence_pattern(repos):
    observe_sequence(repos)

    add_rule(repos, "dinner", related_title="gym", position="after", offset_minutes=0)

    assert len(repos.sequence_patterns.list_pending()) == 1


def test_a_position_and_minutes_with_no_related_event_are_free_time_around_it(repos):
    outcome = add_rule(repos, "work", position="before", offset_minutes=30)

    assert outcome == AddEventRuleSuccess(SavedGap("work", "before", 30))
    (rule,) = repos.event_rules.get_all()
    assert (rule.trigger_title, rule.related_title, rule.position) == (
        "free time",
        "work",
        "before",
    )
    assert (rule.offset_minutes, rule.gap_minutes) == (None, 30)
    assert repos.event_defaults.get_by_title("work") is None


def test_gap_rule_closes_the_gap_pattern_within_tolerance(repos):
    observe_gap(repos, gap=30)

    add_rule(repos, "break", related_title="work", position="after", gap_minutes=30)

    assert repos.gap_patterns.list_pending() == []


def test_gap_rule_with_very_different_minutes_does_not_close_the_pattern(repos):
    observe_gap(repos, gap=30)

    add_rule(repos, "break", related_title="work", position="after", gap_minutes=120)

    assert len(repos.gap_patterns.list_pending()) == 1


def test_closed_patterns_state_is_confirmed(repos):
    pattern = observe_event_pattern(repos)

    add_rule(repos, "gym", typical_start="18:00", typical_duration_minutes=60)

    [row] = repos.conn.execute("SELECT status FROM event_patterns").fetchall()
    assert row["status"] == NoteStatus.CONFIRMED.value
    assert pattern.status == NoteStatus.PENDING  # it was at the time of the observation
