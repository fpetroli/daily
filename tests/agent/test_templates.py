from datetime import date

import pytest

from daily.agent.parser import ParseFailure
from daily.agent.prompts import INSTRUCTIONS
from daily.agent.results import (
    Executed,
    ExecutionError,
    FreeSlotsView,
    HabitDismissed,
    MissingField,
    NoAction,
    ParseFailed,
    ScheduleView,
)
from daily.agent.schemas import parse_tool_call
from daily.agent.templates import Reply, day_label, render, rules_reply, undone_text
from daily.domain.models import EventDefault, EventRule
from daily.domain.timeband import TimeBand
from daily.services.habits import (
    EventPatternProposal,
    GapPatternProposal,
    SequencePatternProposal,
)
from daily.services.placement import (
    AnchorDurationMissing,
    AnchorGapConfirmationNeeded,
    AnchorMissing,
    AnchorNeedsDuration,
)
from daily.services.relative import AnchorAmbiguous, AnchorNotFound
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
)
from daily.services.schedule import (
    EventAlreadyThere,
    EventAmbiguous,
    EventCreated,
    EventNotFound,
    EventRemoved,
    EventUpdated,
    InsufficientInfo,
    InvalidRange,
    NeedsAnchorConfirmation,
    NeedsAnchorDuration,
    NeedsDurationConfirmation,
    NeedsGapConfirmation,
)
from daily.services.types import (
    CheckOverlapResult,
    EventInfo,
    FreeSlot,
    RuleProposal,
    ScheduleEntry,
)
from daily.services.undo import Undone

from .conftest import NOW, TODAY, TOMORROW

YES_NO = ("Yes", "No")


def call(**payload):
    return parse_tool_call(payload)


def show(payload: dict, outcome, habit=None) -> Reply:
    return render(Executed(call(**payload), outcome, habit), NOW)


def info(title, start, end, event_id="e"):
    return EventInfo(event_id, title, f"{TODAY}T{start}", f"{TODAY}T{end}")


# --- days ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        ("2026-09-21", "today"),
        ("2026-09-22", "tomorrow"),
        ("2026-09-23", "the day after tomorrow"),
        ("2026-09-20", "yesterday"),
        ("2026-09-26", "Saturday, September 26"),
        ("2026-12-01", "Tuesday, December 1"),
        ("2027-01-04", "Monday, January 4, 2027"),
    ],
)
def test_day_labels(day, expected):
    assert day_label(day, date(2026, 9, 21)) == expected


# --- creation --------------------------------------------------------------------------


def created(title="gym", start="18:00", end="19:00", **extra):
    return EventCreated("e1", title, f"{TOMORROW}T{start}", f"{TOMORROW}T{end}", **extra)


def test_creation_is_self_contained_title_day_and_times():
    reply = show({"tool": "add_event"}, created())

    assert reply == Reply("Done: gym tomorrow from 18:00 to 19:00.")


def test_creation_reports_conflicts_with_title_and_time():
    conflicts = (info("meeting", "10:00", "11:00"),)

    reply = show({"tool": "add_event"}, created(conflicts=conflicts))

    assert "Warning: it overlaps with meeting (10:00-11:00)." in reply.text
    assert not reply.expects_reply


def test_creation_with_downstream_proposal_asks_confirmation_with_full_times():
    proposals = (RuleProposal("shower", f"{TOMORROW}T19:00", f"{TOMORROW}T19:15"),)

    reply = show({"tool": "add_event"}, created(rule_proposals=proposals))

    assert reply.text.endswith('Shall I also add "shower" from 19:00 to 19:15?')
    assert (reply.quick_replies, reply.expects_reply) == (YES_NO, True)


def test_creation_at_24_writes_24_00():
    reply = show(
        {"tool": "add_event"},
        EventCreated("e", "party", f"{TODAY}T23:00", "2026-09-22T00:00"),
    )

    assert "from 23:00 to 24:00" in reply.text


# --- questions on missing data ---------------------------------------------------------


def test_missing_time_and_missing_duration_quote_the_title():
    ask_start = show({"tool": "add_event", "title": "yoga"}, InsufficientInfo("start"))
    ask_duration = show({"tool": "add_event", "title": "yoga"}, InsufficientInfo("duration"))

    assert ask_start == Reply('What time do you want "yoga"?', expects_reply=True)
    assert ask_duration == Reply('How long does "yoga" last?', expects_reply=True)


def test_an_exact_copy_says_it_was_not_added_again():
    outcome = EventAlreadyThere("gym", f"{TODAY}T18:00", f"{TODAY}T19:00")

    reply = show({"tool": "add_event"}, outcome)

    assert reply == Reply(
        '"gym" is already in your schedule today from 18:00 to 19:00: I didn\'t add it again.'
    )


def test_invalid_range_asks_to_correct():
    reply = show({"tool": "add_event"}, InvalidRange())

    assert "the end must come after the start" in reply.text
    assert reply.expects_reply


def test_missing_anchor_asks_the_time_and_quotes_the_suggestion():
    missing = AnchorMissing("shower", "gym", suggested_related_start="18:00")

    reply = show({"tool": "add_event", "title": "shower"}, NeedsAnchorConfirmation(missing))

    assert reply.text == (
        'I can\'t find "gym" in the schedule today: what time is it, so I can place "shower" '
        "accordingly? (it is usually at 18:00)"
    )
    assert reply.expects_reply and reply.quick_replies == ()


def test_missing_anchor_without_suggestion_and_for_the_requested_day():
    missing = AnchorMissing("shower", "gym")

    reply = show(
        {"tool": "add_event", "title": "shower", "date": TOMORROW}, NeedsAnchorConfirmation(missing)
    )

    assert "in the schedule tomorrow" in reply.text and "(it is usually" not in reply.text


def test_missing_duration():
    reply = show(
        {"tool": "add_event", "title": "breakfast"},
        NeedsDurationConfirmation(AnchorDurationMissing("breakfast")),
    )

    assert reply == Reply('How long does "breakfast" usually last?', expects_reply=True)


def test_missing_anchor_duration_explains_what_to_do():
    reply = show(
        {"tool": "add_event", "title": "shower", "anchor_time": "18:00"},
        NeedsAnchorDuration(AnchorNeedsDuration("shower", "gym")),
    )

    assert 'I don\'t know how long "gym" lasts' in reply.text
    assert '"gym from 18 to 19"' in reply.text


def gap_confirmation(position="after"):
    rule = EventRule("r", "commute", "gym", position, None, 20)
    return AnchorGapConfirmationNeeded(
        trigger_title="shower",
        anchor_title="gym",
        anchor_boundary=f"{TOMORROW}T19:00",
        start_with_gap=f"{TOMORROW}T19:20",
        end_with_gap=f"{TOMORROW}T19:35",
        start_without_gap=f"{TOMORROW}T19:00",
        end_without_gap=f"{TOMORROW}T19:15",
        gaps=(rule,),
    )


def test_usual_gap_is_a_yes_no_question_with_both_times():
    reply = show({"tool": "add_event", "title": "shower"}, NeedsGapConfirmation(gap_confirmation()))

    assert reply.text == (
        'After "gym" there is usually a 20-minute gap: "shower" would be tomorrow at 19:20 '
        "with the gap, at 19:00 without it. Shall I keep the gap?"
    )
    assert (reply.quick_replies, reply.expects_reply) == (YES_NO, True)


def test_usual_gap_before_the_anchor():
    reply = show(
        {"tool": "add_event", "title": "shower"},
        NeedsGapConfirmation(gap_confirmation("before")),
    )

    assert reply.text.startswith('Before "gym" there is usually a')


# --- finding an event ------------------------------------------------------------------


def test_event_not_found_quotes_description_and_day():
    today = show({"tool": "remove_event", "event_description": "dentist"}, EventNotFound())
    other = show(
        {"tool": "update_event", "event_description": "dentist", "event_date": TOMORROW},
        EventNotFound(),
    )

    assert today == Reply(
        'I can\'t find any event "dentist" in the schedule today.', expects_reply=True
    )
    assert other.text == 'I can\'t find any event "dentist" in the schedule tomorrow.'


def test_same_name_events_are_numbered_with_time_and_each_is_a_quick_reply():
    candidates = (
        info("call with Marco", "10:00", "10:30", "a"),
        info("call with the client", "15:00", "15:30", "b"),
    )

    reply = show({"tool": "remove_event", "event_description": "call"}, EventAmbiguous(candidates))

    assert reply.text == (
        "I found several events today: 1) call with Marco (10:00-10:30), 2) call with the "
        "client (15:00-15:30). Which one do you mean?"
    )
    assert reply.quick_replies == ("call with Marco at 10:00", "call with the client at 15:00")
    assert reply.expects_reply


def test_relative_anchor_not_found_and_ambiguous():
    payload = {"tool": "add_event_relative", "title": "shower", "anchor_description": "gym"}

    not_found = show(payload, AnchorNotFound())
    ambiguous = show(
        payload,
        AnchorAmbiguous((info("gym", "07:00", "08:00"), info("gym", "18:00", "19:00"))),
    )

    assert not_found.text == 'I can\'t find any event "gym" in the schedule today.'
    assert ambiguous.quick_replies == ("gym at 07:00", "gym at 18:00")


# --- change and deletion ---------------------------------------------------------------


def updated(**kwargs):
    base = {
        "event_id": "e1",
        "title": "gym",
        "start": f"{TODAY}T18:30",
        "end": f"{TODAY}T19:30",
        "original_start": f"{TODAY}T18:00",
        "original_end": f"{TODAY}T19:00",
        "original_title": "gym",
    }
    return EventUpdated(**{**base, **kwargs})


def test_move_says_where_it_ended_up():
    reply = show({"tool": "update_event"}, updated())

    assert reply == Reply('Done: I moved "gym" to today from 18:30 to 19:30.')


def test_title_only_change():
    reply = show(
        {"tool": "update_event"},
        updated(title="intense gym", start=f"{TODAY}T18:00", end=f"{TODAY}T19:00"),
    )

    assert reply.text == 'Done: "gym" is now called "intense gym".'


def test_move_reports_conflicts():
    reply = show({"tool": "update_event"}, updated(conflicts=(info("lunch", "19:00", "20:00"),)))

    assert "Warning: it overlaps with lunch (19:00-20:00)." in reply.text


def test_move_proposes_moving_the_derived_event_by_the_same_interval():
    outcome = updated(cascade_candidates=(info("shower", "19:00", "19:15"),))

    reply = show({"tool": "update_event"}, outcome)

    assert reply.text.endswith('Shall I also move "shower" from 19:30 to 19:45?')
    assert (reply.quick_replies, reply.expects_reply) == (YES_NO, True)


def test_move_backwards_moves_the_derived_event_backwards():
    outcome = updated(
        start=f"{TODAY}T17:00",
        end=f"{TODAY}T18:00",
        cascade_candidates=(info("shower", "19:00", "19:15"),),
    )

    assert (
        'Shall I also move "shower" from 18:00 to 18:15?'
        in show({"tool": "update_event"}, outcome).text
    )


def test_deletion_says_what_was_deleted():
    reply = show(
        {"tool": "remove_event"},
        EventRemoved("e1", "gym", f"{TODAY}T18:00", f"{TODAY}T19:00"),
    )

    assert reply == Reply('Done: I deleted "gym" (today 18:00-19:00).')


def test_deletion_proposes_deleting_the_derived_event():
    outcome = EventRemoved(
        "e1",
        "gym",
        f"{TODAY}T18:00",
        f"{TODAY}T19:00",
        cascade_candidates=(info("shower", "19:00", "19:15"),),
    )

    reply = show({"tool": "remove_event"}, outcome)

    assert reply.text.endswith('Shall I also delete "shower" (19:00-19:15)?')
    assert (reply.quick_replies, reply.expects_reply) == (YES_NO, True)


def test_overlap():
    payload = {"tool": "check_overlap", "time": "16:30", "end_time": "17:30"}
    busy = show(payload, CheckOverlapResult(True, (info("meeting", "16:00", "17:00"),)))
    free = show(payload, CheckOverlapResult(False, ()))

    assert busy.text == "Yes: today from 16:30 to 17:30 overlaps with meeting (16:00-17:00)."
    assert free.text == "No: today from 16:30 to 17:30 you are free."


# --- reads -----------------------------------------------------------------------------


def entry(title, start, end, done=False, started=None):
    started = done if started is None else started
    return ScheduleEntry("e", title, f"{TODAY}T{start}", f"{TODAY}T{end}", done, started)


def schedule(entries, day=TODAY, band=None):
    return show({"tool": "get_schedule"}, ScheduleView(day, band, tuple(entries)))


def styled(reply: Reply) -> list[tuple[str, str]]:
    return [(s.style, reply.text[s.start : s.end]) for s in reply.styles]


def test_empty_schedule():
    assert schedule([]).text == "Today · Monday, September 21\n\nNothing planned."
    assert schedule([], TOMORROW, TimeBand.EVENING).text == (
        "Tomorrow in the evening · Tuesday, September 22\n\nNothing planned."
    )
    assert schedule([]).styles == ()


def test_schedule_is_a_card_per_event_with_start_end_and_duration():
    reply = schedule(
        [
            entry("dinner", "20:00", "21:00"),
            entry("film", "21:30", "23:15"),
            entry("tea", "23:30", "23:45"),
        ]
    )

    assert reply.text == (
        "Today · Monday, September 21\n\n"
        "dinner\n20:00 → 21:00 · 1 h\n\n"
        "┄┄ 30 min ┄┄\n\n"
        "film\n21:30 → 23:15 · 1 h 45 min\n\n"
        "┄┄ 15 min ┄┄\n\n"
        "tea\n23:30 → 23:45 · 15 min"
    )
    assert styled(reply) == [("bold", "dinner"), ("bold", "film"), ("bold", "tea")]


def test_schedule_strikes_through_what_is_already_done():
    reply = schedule(
        [entry("breakfast", "08:00", "08:30", done=True), entry("dinner", "20:00", "21:00")]
    )

    assert reply.text == (
        "Today · Monday, September 21\n\n"
        "breakfast\n08:00 → 08:30 · 30 min\n\n"
        "┄┄ 11 h 30 min ┄┄\n\n"
        "dinner\n20:00 → 21:00 · 1 h"
    )
    assert styled(reply) == [
        ("bold", "breakfast"),
        ("strikethrough", "breakfast"),
        ("strikethrough", "08:00 → 08:30 · 30 min"),
        ("bold", "dinner"),
    ]


def test_the_free_time_is_struck_through_once_the_next_event_began():
    reply = schedule(
        [
            entry("work", "09:00", "18:00", done=True),
            entry("gym", "18:30", "20:00", started=True),
            entry("dinner", "21:00", "22:00"),
        ]
    )

    assert "┄┄ 30 min ┄┄" in reply.text
    assert "┄┄ 1 h ┄┄" in reply.text
    assert ("strikethrough", "30 min") in styled(reply)
    assert ("strikethrough", "1 h") not in styled(reply)


def test_schedule_until_midnight_ends_at_24():
    late = ScheduleEntry("e", "party", f"{TODAY}T22:00", f"{TOMORROW}T00:00")

    assert schedule([late]).text.endswith("party\n22:00 → 24:00 · 2 h")


def test_no_gap_line_between_touching_or_overlapping_events():
    text = schedule(
        [
            entry("work", "09:00", "13:00"),
            entry("lunch", "13:00", "14:00"),
            entry("call", "13:30", "13:45"),  # inside lunch: the gap runs from 14:00
            entry("walk", "14:01", "14:30"),
        ]
    ).text

    assert text.count("┄┄") == 2 and "lunch\n13:00 → 14:00 · 1 h\n\ncall" in text
    assert "call\n13:30 → 13:45 · 15 min\n\n┄┄ 1 min ┄┄\n\nwalk" in text


def test_schedule_ends_with_the_gap_and_the_first_event_of_the_next_day():
    first = ScheduleEntry("e", "breakfast", f"{TOMORROW}T07:30", f"{TOMORROW}T08:00")
    view = ScheduleView(
        TODAY,
        None,
        (entry("gym", "18:00", "19:00"), entry("call", "18:30", "19:30")),
        first,
    )

    reply = show({"tool": "get_schedule"}, view)

    assert reply.text == (
        "Today · Monday, September 21\n\n"
        "gym\n18:00 → 19:00 · 1 h\n\n"
        "call\n18:30 → 19:30 · 1 h\n\n"
        "──── 12 h ────\n\n"  # from the latest end, not from the last start
        "Tomorrow · Tuesday, September 22\n\n"
        "breakfast\n07:30 → 08:00 · 30 min"
    )
    assert styled(reply)[-1] == ("bold", "breakfast")


def test_no_gap_when_the_day_runs_into_the_next_event():
    late = ScheduleEntry("e", "party", f"{TODAY}T23:00", f"{TOMORROW}T01:00")
    first = ScheduleEntry("e", "sleep", f"{TOMORROW}T00:30", f"{TOMORROW}T07:00")

    text = show({"tool": "get_schedule"}, ScheduleView(TODAY, None, (late,), first)).text

    assert "\n\n────────────\n\nTomorrow · Tuesday, September 22\n\nsleep\n" in text


def test_schedule_of_another_day_with_band():
    reply = schedule([entry("meeting", "15:00", "16:00")], "2026-09-24", TimeBand.AFTERNOON)

    assert reply.text == ("Thursday, September 24 in the afternoon\n\nmeeting\n15:00 → 16:00 · 1 h")


def slots(items, band=None, minimum=None, day=TODAY):
    view = FreeSlotsView(day, band, minimum, tuple(FreeSlot(*item) for item in items))
    return show({"tool": "get_free_slots"}, view).text


def test_free_slots():
    assert slots([(f"{TODAY}T16:00", f"{TODAY}T18:00")]) == "Today you are free: 16:00-18:00."
    assert slots([(f"{TODAY}T20:00", "2026-09-22T00:00")]) == "Today you are free: 20:00-24:00."


def test_no_free_slot_quotes_the_minimum_duration():
    assert slots([], TimeBand.EVENING, 90) == (
        "Today in the evening I found no free slots of at least 90 minutes."
    )
    assert slots([]) == "Today I found no free slots."


# --- rules -----------------------------------------------------------------------------


def rule_added(**payload):
    return show({"tool": "add_event_rule", **payload}, AddEventRuleSuccess()).text


def test_relational_rule_gap_and_usual_time():
    assert (
        rule_added(trigger_title="shower", related_title="gym", position="after", offset_minutes=0)
        == 'Done: "shower" always goes after "gym", right after.'
    )
    assert (
        rule_added(
            trigger_title="breakfast", related_title="work", position="before", offset_minutes=30
        )
        == 'Done: "breakfast" always goes before "work", 30 minutes before.'
    )
    assert (
        rule_added(trigger_title="break", related_title="work", position="after", gap_minutes=30)
        == 'Done: after "work" I will always leave a 30-minute gap.'
    )
    assert (
        rule_added(trigger_title="meeting", typical_start="09:00", typical_duration_minutes=30)
        == 'Done: "meeting" usually is at 09:00 and lasts 30 minutes.'
    )
    assert rule_added(trigger_title="meeting", typical_duration_minutes=30) == (
        'Done: "meeting" usually lasts 30 minutes.'
    )


def test_free_time_read_from_a_position_tells_what_was_saved():
    outcome = AddEventRuleSuccess(SavedGap("work", "before", 30))
    payload = {"trigger_title": "work", "position": "before", "offset_minutes": 30}

    reply = show({"tool": "add_event_rule", **payload}, outcome)

    assert reply.text == 'Done: before "work" I will always leave a 30-minute gap.'


def test_rule_removed_tells_what_was_removed():
    def removed(target):
        return show({"tool": "remove_event_rule"}, RemoveEventRuleSuccess(target)).text

    assert removed(EventRule("r", "shower", "gym", "after")) == (
        'Done: I removed the rule "shower right after gym".'
    )
    assert removed(EventRule("r", "breakfast", "gym", "before", offset_minutes=15)) == (
        'Done: I removed the rule "breakfast 15 min before gym".'
    )
    assert removed(EventRule("r", "free time", "work", "before", gap_minutes=30)) == (
        'Done: I removed the rule "30 min free before work".'
    )
    assert removed(UsualTimes("meeting")) == (
        'Done: I removed the usual time and duration of "meeting".'
    )
    assert removed(UsualTimes("meeting", "typical_start")) == (
        'Done: I removed the usual time of "meeting".'
    )


def test_rule_without_position_asks_before_or_after():
    reply = show({"tool": "add_event_rule"}, AddEventRuleNeedsPosition("shower", "gym"))

    assert reply == Reply('Does "shower" go before or after "gym"?', ("Before", "After"), True)


def test_other_questions_about_rules():
    payload = {"tool": "add_event_rule"}

    assert "HH:MM" in show(payload, AddEventRuleInvalidFormat()).text
    assert show(payload, AddEventRuleNeedsInfo("meeting")).text.startswith(
        'What do you want me to remember for "meeting"'
    )
    assert show({"tool": "remove_event_rule"}, RemoveEventRuleNotFound()).text == (
        "I can't find any matching rule."
    )


def test_ambiguous_rules_are_buttons_that_remove_them():
    candidates = (UsualTimes("gym"), EventRule("r2", "shower", "gym", "after"))

    reply = show({"tool": "remove_event_rule"}, RemoveEventRuleAmbiguous(candidates))

    assert reply.text == (
        'I found several: 1) the usual time and duration of "gym", 2) the rule "shower right '
        'after gym". Which one do you want to remove?'
    )
    assert reply.quick_replies == (
        'The usual time and duration of "gym"',
        'The rule "shower right after gym"',
    )
    assert reply.choices == candidates


# --- habits ----------------------------------------------------------------------------


def test_usual_time_proposal():
    proposal = EventPatternProposal("n", "gym", "18:00", 60)

    reply = show({"tool": "add_event"}, created(), proposal)

    assert reply.text == (
        'Done: gym tomorrow from 18:00 to 19:00. I noticed you often do "gym" around 18:00 '
        "(60 minutes): shall I save it as a usual time?"
    )
    assert (reply.quick_replies, reply.expects_reply) == (YES_NO, True)


def test_sequence_rule_proposal():
    after_gap = SequencePatternProposal("n", "shower", "gym", "after", 10)
    right_after = SequencePatternProposal("n", "shower", "gym", "after", 0)

    assert (
        '"shower" 10 minutes after "gym": shall I save it as a rule?'
        in show({"tool": "add_event"}, created(), after_gap).text
    )
    assert '"shower" right after "gym"' in show({"tool": "add_event"}, created(), right_after).text


def test_recurring_gap_proposal_is_a_yes_or_no():
    proposal = GapPatternProposal("n", "work", "after", 30)

    reply = show({"tool": "add_event"}, created(), proposal)

    assert reply.text.endswith(
        'You often leave about 30 minutes free after "work": shall I save it as a rule?'
    )
    assert (reply.quick_replies, reply.expects_reply) == (YES_NO, True)


@pytest.mark.parametrize(
    ("proposal", "accept"),
    [
        (
            EventPatternProposal("n", "gym", "18:00", 60),
            {"trigger_title": "gym", "typical_start": "18:00", "typical_duration_minutes": 60},
        ),
        (
            SequencePatternProposal("n", "shower", "gym", "after", 0),
            {"trigger_title": "shower", "related_title": "gym", "position": "after"},
        ),
        (
            GapPatternProposal("n", "work", "after", 30),
            {
                "trigger_title": "free time",
                "related_title": "work",
                "position": "after",
                "gap_minutes": 30,
            },
        ),
    ],
)
def test_proposal_buttons_save_the_rule_or_reject_it(proposal, accept):
    reply = show({"tool": "add_event"}, created(), proposal)

    assert reply.choices == (call(tool="add_event_rule", **accept), call(tool="dismiss_habit"))


def test_rejection_of_a_suggestion():
    yes = show(
        {"tool": "dismiss_habit"}, HabitDismissed(EventPatternProposal("n", "p", "18:00", 60))
    )
    nothing = show({"tool": "dismiss_habit"}, HabitDismissed(None))

    assert yes.text == "Ok, I won't suggest it again."
    assert nothing.text == "There was no suggestion to reject."


def test_after_a_rejection_the_next_suggestion_is_appended():
    reply = show(
        {"tool": "dismiss_habit"},
        HabitDismissed(EventPatternProposal("n", "p", "18:00", 60)),
        SequencePatternProposal("m", "shower", "gym", "after", 0),
    )

    assert reply.text.startswith('Ok, I won\'t suggest it again. You often add "shower"')
    assert reply.expects_reply


# --- missing data and errors -----------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "subject", "expected"),
    [
        ("title", None, "What is the event called?"),
        ("event_description", None, "Which event do you mean?"),
        ("trigger_title", None, "Which event is it about?"),
        ("time", None, "At what time?"),
        ("end_time", None, "Until what time?"),
        ("anchor_description", "shower", 'Relative to which event do you want to place "shower"?'),
        ("duration_minutes", "shower", 'How long does "shower" last?'),
        (
            "change",
            "meeting",
            'What do you want to change about "meeting": the time, the duration or the name?',
        ),
        ("something", None, "Can you give me more details?"),
    ],
)
def test_questions_on_missing_data(field, subject, expected):
    reply = render(MissingField(call(tool="none"), field, subject), NOW)

    assert reply == Reply(expected, expects_reply=True)


def test_missing_position_has_quick_replies():
    reply = render(MissingField(call(tool="add_event_relative"), "position", "shower"), NOW)

    assert reply == Reply('Does "shower" go before or after?', ("Before", "After"), True)


def test_no_action_and_errors_are_fixed_messages_without_question():
    assert "I can add, move or delete events" in render(NoAction(), NOW).text
    assert render(ParseFailed(ParseFailure("invalid_output")), NOW) == Reply(
        "I didn't understand, can you rephrase?"
    )
    for kind in ("unavailable", "timeout", "error"):
        assert render(ParseFailed(ParseFailure(kind)), NOW) == Reply(
            "The model is not answering: try again shortly."
        )
    assert render(ExecutionError(), NOW) == Reply("Something went wrong: try again.")
    for reply in (render(NoAction(), NOW), render(ExecutionError(), NOW)):
        assert not reply.expects_reply and reply.quick_replies == ()


def test_unknown_outcome_or_broken_template_give_a_generic_message():
    assert render(object(), NOW) == Reply("Done.")
    assert render(Executed(call(tool="none"), object()), NOW) == Reply("Done.")
    # outcome and tool call inconsistent: the template raises, the message stays generic
    broken = Executed(call(tool="none"), CheckOverlapResult(True, ()))
    assert render(broken, NOW) == Reply("Done.")


# --- the prompt and the templates say the same things ------------------------------------


def test_phrases_the_prompt_teaches_to_recognize_are_the_template_ones():
    texts = " ".join(
        [
            show(
                {"tool": "add_event"},
                created(
                    rule_proposals=(
                        RuleProposal("shower", f"{TOMORROW}T19:00", f"{TOMORROW}T19:15"),
                    )
                ),
            ).text,
            show(
                {"tool": "update_event"},
                updated(cascade_candidates=(info("shower", "19:00", "19:15"),)),
            ).text,
            show(
                {"tool": "remove_event"},
                EventRemoved(
                    "e",
                    "p",
                    f"{TODAY}T18:00",
                    f"{TODAY}T19:00",
                    (info("shower", "19:00", "19:15"),),
                ),
            ).text,
            show({"tool": "add_event"}, NeedsGapConfirmation(gap_confirmation())).text,
            show(
                {"tool": "add_event"},
                NeedsAnchorConfirmation(AnchorMissing("shower", "gym")),
            ).text,
            show(
                {"tool": "add_event"}, NeedsDurationConfirmation(AnchorDurationMissing("shower"))
            ).text,
            show(
                {"tool": "remove_event", "event_description": "c"},
                EventAmbiguous((info("c", "10:00", "11:00"), info("c", "12:00", "13:00"))),
            ).text,
        ]
    )

    for phrase in (
        "Shall I also add",
        "Shall I also move",
        "Shall I also delete",
        "Shall I keep the gap",
        "what time is it",
        "How long does",
        "Which one do you mean?",
    ):
        assert phrase in texts, phrase
        assert phrase in INSTRUCTIONS, phrase


def test_event_not_found_by_time_quotes_the_time():
    reply = show({"tool": "remove_event", "event_time": "18:00"}, EventNotFound())

    assert reply.text == "I can't find any event at 18:00 in the schedule today."


def test_a_no_after_a_question_says_nothing_changed():
    from daily.agent.results import NoAction

    assert render(NoAction(after_question=True), NOW).text.startswith(
        "Ok, I changed nothing. If you meant something else, write the whole request"
    )
    assert render(NoAction(), NOW).text.startswith("I didn't understand what to do")


# --- rules and undo (the bot's commands) ----------------------------------------------


def test_rules_without_rules_explains_how_to_teach_one():
    reply = rules_reply([], [])

    assert reply.text.startswith("There are no rules yet.") and reply.styles == ()


def test_rules_are_a_card_per_event_with_what_is_known_about_it():
    rules = [
        EventRule("r1", "shower", "gym", "after"),
        EventRule("r2", "stretching", "run", "before", offset_minutes=10),
        EventRule("r3", "break", "lunch", "after", gap_minutes=30),
        EventRule("r4", "stretching", "Yoga", "before", offset_minutes=90),
    ]
    defaults = [EventDefault("yoga", "19:00", 45), EventDefault("meeting", None, 60)]

    reply = rules_reply(rules, defaults)

    assert reply.text == (
        "Rules and usual times\n\n"
        "gym\nshower right after\n\n"
        "lunch\n30 min free after\n\n"
        "meeting\nusually 1 h\n\n"
        "run\nstretching 10 min before\n\n"
        "yoga\nusually at 19:00 · 45 min\nstretching 1 h 30 min before"
    )
    assert styled(reply) == [
        ("bold", title) for title in ("gym", "lunch", "meeting", "run", "yoga")
    ]


@pytest.mark.parametrize(
    ("undone", "expected"),
    [
        (None, "There is nothing to undo."),
        (
            Undone("add_event", "gym", f"{TOMORROW}T18:00", f"{TOMORROW}T19:00"),
            'Undone: I deleted "gym" tomorrow from 18:00 to 19:00.',
        ),
        (
            Undone("update_event", "gym", f"{TODAY}T18:00", f"{TODAY}T19:00"),
            'Undone: "gym" is back today from 18:00 to 19:00.',
        ),
        (
            Undone("remove_event", "gym", "2026-09-28T18:00", "2026-09-28T19:00"),
            'Undone: "gym" is back on Monday, September 28 from 18:00 to 19:00.',
        ),
    ],
)
def test_undone_text(undone, expected):
    assert undone_text(undone, NOW.date()) == expected
