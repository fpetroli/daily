from daily.agent.execute import execute
from daily.agent.results import (
    Executed,
    FreeSlotsView,
    HabitDismissed,
    MissingField,
    NoAction,
    ScheduleView,
)
from daily.agent.schemas import parse_tool_call
from daily.domain.timeband import TimeBand
from daily.services.habits import EventPatternProposal
from daily.services.rules import (
    AddEventRuleNeedsPosition,
    AddEventRuleSuccess,
    RemoveEventRuleNotFound,
    RemoveEventRuleSuccess,
)
from daily.services.schedule import (
    EventAmbiguous,
    EventCreated,
    EventNotFound,
    EventRemoved,
    EventUpdated,
    InsufficientInfo,
    NeedsGapConfirmation,
)
from daily.services.types import CheckOverlapResult

from .conftest import NOW, TODAY, TOMORROW, add_event


def run(repos, **payload):
    return execute(repos, parse_tool_call(payload), now=NOW)


def outcome_of(result):
    assert isinstance(result, Executed), result
    return result.outcome


# --- add_event -----------------------------------------------------------------------------


def test_add_event_creates_the_event(repos):
    outcome = outcome_of(
        run(
            repos,
            tool="add_event",
            title="gym",
            date=TOMORROW,
            time="18:00",
            duration_minutes=60,
        )
    )

    assert isinstance(outcome, EventCreated)
    assert (outcome.start, outcome.end) == (f"{TOMORROW}T18:00", f"{TOMORROW}T19:00")
    assert [e.title for e in repos.schedule.events_for_date(TOMORROW)] == ["gym"]


def test_add_event_without_title_asks_for_the_title(repos):
    result = run(repos, tool="add_event", date=TOMORROW, time="18:00", duration_minutes=60)

    assert isinstance(result, MissingField) and result.field == "title"
    assert repos.schedule.events_for_date(TOMORROW) == []


def test_add_event_without_time_asks_for_the_time(repos):
    outcome = outcome_of(run(repos, tool="add_event", title="yoga"))

    assert outcome == InsufficientInfo(missing="start")


def test_add_event_uses_today_as_default_day_for_rules(repos):
    repos.event_defaults.upsert("yoga", typical_start="19:00", typical_duration_minutes=45)

    outcome = outcome_of(run(repos, tool="add_event", title="yoga"))

    assert outcome.start == f"{TODAY}T19:00"


def test_add_event_date_moves_the_day_of_the_rules(repos):
    repos.event_defaults.upsert("yoga", typical_start="19:00", typical_duration_minutes=45)

    outcome = outcome_of(run(repos, tool="add_event", title="yoga", date=TOMORROW))

    assert outcome.start == f"{TOMORROW}T19:00"


def test_add_event_with_gap_answers_the_gap_question(repos):
    add_event(repos, "work", f"{TODAY}T09:00", f"{TODAY}T17:00")
    repos.event_rules.insert_rule(
        trigger_title="commute", related_title="work", position="before", gap_minutes=45
    )
    repos.event_rules.insert_rule(
        trigger_title="dinner", related_title="work", position="before", offset_minutes=0
    )
    repos.event_defaults.upsert("dinner", typical_duration_minutes=60)

    asked = outcome_of(run(repos, tool="add_event", title="dinner"))
    with_gap = outcome_of(run(repos, tool="add_event", title="dinner", with_gap=True))

    assert isinstance(asked, NeedsGapConfirmation)
    assert with_gap.start == f"{TODAY}T07:15"


def test_add_event_anchor_time_places_relative_to_an_anchor_not_in_schedule(repos):
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )
    repos.event_defaults.upsert("shower", typical_duration_minutes=15)
    repos.event_defaults.upsert("gym", typical_duration_minutes=60)

    outcome = outcome_of(run(repos, tool="add_event", title="shower", anchor_time="18:00"))

    assert (outcome.start, outcome.end) == (f"{TODAY}T19:00", f"{TODAY}T19:15")


def test_add_event_passes_the_time_to_habit_tracking(repos):
    run(repos, tool="add_event", title="yoga", date=TOMORROW, time="19:00", duration_minutes=45)

    [pattern] = repos.event_patterns.list_pending()
    assert pattern.first_occurrence_at == NOW


# --- add_event_relative --------------------------------------------------------------------


def test_add_event_relative_creates_after_the_anchor(repos):
    add_event(repos, "gym", f"{TODAY}T18:00", f"{TODAY}T19:00")

    outcome = outcome_of(
        run(
            repos,
            tool="add_event_relative",
            title="shower",
            anchor_description="gym",
            position="after",
            duration_minutes=15,
        )
    )

    assert (outcome.start, outcome.end) == (f"{TODAY}T19:00", f"{TODAY}T19:15")


def test_add_event_relative_each_missing_value_has_its_question(repos):
    base = {"tool": "add_event_relative", "title": "shower"}
    full = {"anchor_description": "gym", "position": "after", "duration_minutes": 15}

    for missing in full:
        payload = {**base, **{k: v for k, v in full.items() if k != missing}}
        result = run(repos, **payload)
        assert isinstance(result, MissingField) and result.field == missing, missing
        assert result.subject == "shower"
    assert isinstance(run(repos, tool="add_event_relative", **full), MissingField)  # no title


def test_add_event_relative_uses_date_and_anchor_time(repos):
    add_event(repos, "call", f"{TOMORROW}T10:00", f"{TOMORROW}T10:30")
    add_event(repos, "call", f"{TOMORROW}T15:00", f"{TOMORROW}T15:30")

    outcome = outcome_of(
        run(
            repos,
            tool="add_event_relative",
            title="follow-up",
            anchor_description="call",
            anchor_time="15:00",
            date=TOMORROW,
            position="after",
            duration_minutes=15,
        )
    )

    assert outcome.start == f"{TOMORROW}T15:30"


# --- update_event and remove_event ---------------------------------------------------------


def test_update_event_moves_with_shift(repos):
    add_event(repos, "meeting", f"{TODAY}T16:00", f"{TODAY}T17:00")

    outcome = outcome_of(
        run(repos, tool="update_event", event_description="meeting", shift_minutes=30)
    )

    assert isinstance(outcome, EventUpdated)
    assert outcome.start == f"{TODAY}T16:30"


def test_update_event_searches_the_given_day_and_disambiguates(repos):
    add_event(repos, "call", f"{TOMORROW}T10:00", f"{TOMORROW}T10:30")
    add_event(repos, "call", f"{TOMORROW}T15:00", f"{TOMORROW}T15:30")

    ambiguous = outcome_of(
        run(
            repos,
            tool="update_event",
            event_description="call",
            event_date=TOMORROW,
            shift_minutes=5,
        )
    )
    chosen = outcome_of(
        run(
            repos,
            tool="update_event",
            event_description="call",
            event_date=TOMORROW,
            event_time="15:00",
            shift_minutes=5,
        )
    )

    assert isinstance(ambiguous, EventAmbiguous)
    assert isinstance(chosen, EventUpdated) and chosen.start == f"{TOMORROW}T15:05"


def test_update_event_not_found_today(repos):
    add_event(repos, "meeting", f"{TOMORROW}T16:00", f"{TOMORROW}T17:00")  # tomorrow, not today

    outcome = outcome_of(
        run(repos, tool="update_event", event_description="meeting", shift_minutes=5)
    )

    assert outcome == EventNotFound()


def test_update_event_without_description_or_changes_asks(repos):
    add_event(repos, "meeting", f"{TODAY}T16:00", f"{TODAY}T17:00")

    no_description = run(repos, tool="update_event", shift_minutes=30)
    no_change = run(repos, tool="update_event", event_description="meeting")

    assert isinstance(no_description, MissingField) and no_description.field == "event_description"
    assert isinstance(no_change, MissingField)
    assert (no_change.field, no_change.subject) == ("change", "meeting")
    assert repos.schedule.events_for_date(TODAY)[0].start == f"{TODAY}T16:00"


def test_remove_event_without_description_asks_which_event(repos):
    add_event(repos, "meeting", f"{TODAY}T16:00", f"{TODAY}T17:00")

    missing = run(repos, tool="remove_event")

    assert isinstance(missing, MissingField) and missing.field == "event_description"
    assert len(repos.schedule.events_for_date(TODAY)) == 1


def test_remove_event_really_deletes(repos):
    add_event(repos, "meeting", f"{TODAY}T16:00", f"{TODAY}T17:00")

    outcome = outcome_of(run(repos, tool="remove_event", event_description="meeting"))

    assert isinstance(outcome, EventRemoved)
    assert repos.schedule.events_for_date(TODAY) == []


# --- reads ---------------------------------------------------------------------------------


def test_get_schedule_is_a_view_with_day_and_state(repos):
    add_event(repos, "breakfast", f"{TODAY}T08:00", f"{TODAY}T08:30")
    add_event(repos, "dinner", f"{TODAY}T20:00", f"{TODAY}T21:00")

    view = outcome_of(run(repos, tool="get_schedule"))

    assert isinstance(view, ScheduleView) and view.date == TODAY
    assert [(e.title, e.done) for e in view.entries] == [("breakfast", True), ("dinner", False)]


def test_get_schedule_filters_by_band_and_day(repos):
    add_event(repos, "breakfast", f"{TOMORROW}T08:00", f"{TOMORROW}T08:30")
    add_event(repos, "dinner", f"{TOMORROW}T20:00", f"{TOMORROW}T21:00")

    view = outcome_of(run(repos, tool="get_schedule", date=TOMORROW, time_band="evening"))

    assert view.time_band == TimeBand.EVENING
    assert [e.title for e in view.entries] == ["dinner"]


def test_a_whole_day_also_brings_the_first_event_of_the_next_day(repos):
    add_event(repos, "dinner", f"{TODAY}T20:00", f"{TODAY}T21:00")
    add_event(repos, "gym", f"{TOMORROW}T18:00", f"{TOMORROW}T19:00")
    add_event(repos, "breakfast", f"{TOMORROW}T08:00", f"{TOMORROW}T08:30")
    add_event(repos, "later", "2026-09-23T07:00", "2026-09-23T08:00")

    view = outcome_of(run(repos, tool="get_schedule"))

    assert view.following is not None and view.following.title == "breakfast"


def test_no_next_day_event_for_a_band_an_empty_day_or_an_empty_next_day(repos):
    add_event(repos, "breakfast", f"{TOMORROW}T08:00", f"{TOMORROW}T08:30")

    assert outcome_of(run(repos, tool="get_schedule")).following is None  # empty today
    add_event(repos, "dinner", f"{TODAY}T20:00", f"{TODAY}T21:00")
    banded = outcome_of(run(repos, tool="get_schedule", time_band="evening"))
    assert banded.entries and banded.following is None
    assert outcome_of(run(repos, tool="get_schedule", date=TOMORROW)).following is None


def test_get_free_slots_starts_from_current_time_if_today(repos):
    view = outcome_of(run(repos, tool="get_free_slots"))

    assert isinstance(view, FreeSlotsView)
    assert view.slots[0].start == f"{TODAY}T15:30"


def test_get_free_slots_with_band_and_minimum_duration(repos):
    add_event(repos, "work", f"{TOMORROW}T09:00", f"{TOMORROW}T17:00")

    view = outcome_of(
        run(
            repos,
            tool="get_free_slots",
            date=TOMORROW,
            time_band="evening",
            min_duration_minutes=60,
        )
    )

    assert [(s.start, s.end) for s in view.slots] == [(f"{TOMORROW}T19:00", f"{TOMORROW}T23:00")]
    assert view.min_duration_minutes == 60


def test_check_overlap(repos):
    add_event(repos, "meeting", f"{TODAY}T16:00", f"{TODAY}T17:00")

    busy = outcome_of(run(repos, tool="check_overlap", time="16:30", end_time="17:30"))
    free = outcome_of(run(repos, tool="check_overlap", time="17:00", end_time="18:00"))

    assert isinstance(busy, CheckOverlapResult) and busy.overlap
    assert not free.overlap


def test_check_overlap_asks_for_the_missing_bound(repos):
    assert run(repos, tool="check_overlap", end_time="17:00").field == "time"
    assert run(repos, tool="check_overlap", time="17:00").field == "end_time"


# --- rules and habits ----------------------------------------------------------------------


def test_add_event_rule_relational_and_usual_time(repos):
    relational = outcome_of(
        run(
            repos,
            tool="add_event_rule",
            trigger_title="shower",
            related_title="gym",
            position="after",
            offset_minutes=0,
        )
    )
    default = outcome_of(
        run(
            repos,
            tool="add_event_rule",
            trigger_title="meeting",
            typical_start="09:00",
            typical_duration_minutes=30,
        )
    )

    assert relational == AddEventRuleSuccess() == default
    assert repos.event_defaults.get_by_title("meeting").typical_start == "09:00"


def test_add_event_rule_without_position_asks_for_position(repos):
    outcome = outcome_of(
        run(repos, tool="add_event_rule", trigger_title="shower", related_title="gym")
    )

    assert isinstance(outcome, AddEventRuleNeedsPosition)


def test_add_event_rule_without_title_asks_for_name_and_says_so_for_a_gap(repos):
    plain = run(repos, tool="add_event_rule", related_title="gym", position="after")
    gap = run(repos, tool="add_event_rule", related_title="work", position="after", gap_minutes=30)

    assert (plain.field, plain.subject) == ("trigger_title", None)
    assert (gap.field, gap.subject) == ("trigger_title", "gap")


def test_remove_event_rule(repos):
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )

    removed = outcome_of(
        run(repos, tool="remove_event_rule", trigger_title="shower", related_title="gym")
    )
    again = outcome_of(
        run(repos, tool="remove_event_rule", trigger_title="shower", related_title="gym")
    )

    assert removed == RemoveEventRuleSuccess()
    assert again == RemoveEventRuleNotFound()
    assert run(repos, tool="remove_event_rule").field == "trigger_title"


def test_none_is_no_action(repos):
    assert run(repos, tool="none") == NoAction()


# --- habit proposals -----------------------------------------------------------------------


def create_yoga_on(repos, day):
    return execute(
        repos,
        parse_tool_call(
            {
                "tool": "add_event",
                "title": "yoga",
                "date": day,
                "time": "19:00",
                "duration_minutes": 45,
            }
        ),
        now=NOW,
    )


def test_habit_proposal_fires_at_threshold_after_a_creation(repos):
    first = create_yoga_on(repos, "2026-09-22")
    second = create_yoga_on(repos, "2026-09-23")
    third = create_yoga_on(repos, "2026-09-24")

    assert first.habit is None and second.habit is None
    assert third.habit == EventPatternProposal(third.habit.note_id, "yoga", "19:00", 45)


def test_proposal_is_not_appended_to_a_downstream_question(repos):
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="yoga", position="after", offset_minutes=0
    )
    repos.event_defaults.upsert("shower", typical_duration_minutes=15)
    create_yoga_on(repos, "2026-09-22")
    create_yoga_on(repos, "2026-09-23")

    third = create_yoga_on(repos, "2026-09-24")

    assert third.outcome.rule_proposals and third.habit is None


def test_proposal_is_appended_only_after_a_creation(repos):
    for day in ("2026-09-22", "2026-09-23", "2026-09-24"):
        create_yoga_on(repos, day)

    read = run(repos, tool="get_schedule")

    assert read.habit is None


def test_dismiss_habit_rejects_and_appends_the_next_one(repos):
    for day in ("2026-09-22", "2026-09-23", "2026-09-24"):
        create_yoga_on(repos, day)
    add_event(repos, "dinner", f"{TODAY}T20:00", f"{TODAY}T21:00")
    repos.event_defaults.upsert("other", typical_start="10:00", typical_duration_minutes=30)

    result = run(repos, tool="dismiss_habit")

    assert isinstance(result.outcome, HabitDismissed)
    assert result.outcome.proposal.title == "yoga"
    assert repos.event_patterns.list_pending() == []


def test_dismiss_without_proposals_does_nothing(repos):
    assert outcome_of(run(repos, tool="dismiss_habit")) == HabitDismissed(None)


def test_add_event_rule_accepts_the_proposal_and_appends_another(repos):
    for day in ("2026-09-22", "2026-09-23", "2026-09-24"):
        create_yoga_on(repos, day)
        execute(
            repos,
            parse_tool_call(
                {
                    "tool": "add_event",
                    "title": "pilates",
                    "date": day,
                    "time": "10:00",
                    "duration_minutes": 60,
                }
            ),
            now=NOW,
        )

    result = run(
        repos,
        tool="add_event_rule",
        trigger_title="yoga",
        typical_start="19:00",
        typical_duration_minutes=45,
    )

    assert result.outcome == AddEventRuleSuccess()
    assert result.habit is not None and result.habit.title == "pilates"  # the next one


def test_update_event_extend_counts_as_change_and_lengthens(repos):
    add_event(repos, "meeting", f"{TODAY}T16:00", f"{TODAY}T17:00")

    outcome = outcome_of(
        run(repos, tool="update_event", event_description="meeting", extend_minutes=30)
    )

    assert isinstance(outcome, EventUpdated) and outcome.end == f"{TODAY}T17:30"


def test_remove_and_update_accept_only_the_time_without_title(repos):
    add_event(repos, "gym", f"{TODAY}T18:00", f"{TODAY}T19:00")
    add_event(repos, "dinner", f"{TODAY}T20:00", f"{TODAY}T21:00")

    moved = outcome_of(run(repos, tool="update_event", event_time="20:00", shift_minutes=30))
    removed = outcome_of(run(repos, tool="remove_event", event_time="18:00"))

    assert isinstance(moved, EventUpdated) and moved.title == "dinner"
    assert isinstance(removed, EventRemoved) and removed.removed_title == "gym"


def test_without_title_and_time_asks_which_event(repos):
    assert run(repos, tool="remove_event").field == "event_description"
    assert run(repos, tool="update_event", shift_minutes=5).field == "event_description"


# --- action_log (undo) -----------------------------------------------------------------


def test_add_event_logs_the_action(repos):
    outcome = outcome_of(
        run(
            repos,
            tool="add_event",
            title="gym",
            date=TOMORROW,
            time="18:00",
            duration_minutes=60,
        )
    )

    entry = repos.action_log.latest()
    assert entry is not None
    assert entry.kind == "add_event"
    assert entry.undo_data == {
        "event_id": outcome.event_id,
        "title": "gym",
        "start": f"{TOMORROW}T18:00",
        "end": f"{TOMORROW}T19:00",
    }


def test_update_event_logs_the_previous_values(repos):
    add_event(repos, "meeting", f"{TODAY}T16:00", f"{TODAY}T17:00")

    outcome = outcome_of(
        run(repos, tool="update_event", event_description="meeting", shift_minutes=30)
    )

    entry = repos.action_log.latest()
    assert entry.kind == "update_event"
    assert entry.undo_data == {
        "event_id": outcome.event_id,
        "title": "meeting",
        "start": f"{TODAY}T16:00",
        "end": f"{TODAY}T17:00",
    }


def test_remove_event_logs_the_removed_event(repos):
    add_event(repos, "meeting", f"{TODAY}T16:00", f"{TODAY}T17:00")

    outcome = outcome_of(run(repos, tool="remove_event", event_description="meeting"))

    entry = repos.action_log.latest()
    assert entry.kind == "remove_event"
    assert entry.undo_data == {
        "event_id": outcome.event_id,
        "title": "meeting",
        "start": f"{TODAY}T16:00",
        "end": f"{TODAY}T17:00",
    }


def test_a_read_logs_nothing(repos):
    run(repos, tool="get_schedule")

    assert repos.action_log.latest() is None


def test_a_rule_logs_nothing(repos):
    run(repos, tool="add_event_rule", trigger_title="meeting", typical_start="09:00")

    assert repos.action_log.latest() is None


def test_a_missing_field_logs_nothing(repos):
    run(repos, tool="add_event", title="yoga")

    assert repos.action_log.latest() is None


def test_each_write_replaces_the_previous_as_latest(repos):
    add_event(repos, "meeting", f"{TODAY}T16:00", f"{TODAY}T17:00")
    run(repos, tool="add_event", title="gym", date=TOMORROW, time="18:00", duration_minutes=60)

    run(repos, tool="update_event", event_description="meeting", shift_minutes=30)

    assert repos.action_log.latest().kind == "update_event"
