"""Multi-turn scenarios with a scripted parse: the workflow, the history and the messages
together.

The scripted parser plays the model: when it answers a "yes" it builds the tool call **by
reading only the text of the last exchange**. If a message did not contain everything needed
(title, day, time), the test would fail: it proves that the messages are self-contained.
"""

import asyncio
import logging
import re

import pytest

from daily.agent.parser import Exchange, ParseFailure
from daily.agent.results import Executed, ExecutionError
from daily.agent.templates import Reply
from daily.agent.turn import TurnService
from daily.domain.patterns import NoteStatus
from daily.services.schedule import EventCreated
from scripted import ScriptedParser
from turn_log import read_turn

from .conftest import NOW, TODAY, TOMORROW, add_event, say

TIMES = re.compile(r"\d{2}:\d{2}")


def last_reply(request) -> str:
    return request.history[-1].reply if request.history else ""


def titles_on(repos, day):
    return [e.title for e in repos.schedule.events_for_date(day)]


def by_title(repos, day, title):
    return next(e for e in repos.schedule.events_for_date(day) if e.title == title)


# --- one turn --------------------------------------------------------------------------


async def test_a_turn_executes_and_replies(repos, parser, turn):
    parser.then(
        {
            "tool": "add_event",
            "title": "gym",
            "date": TOMORROW,
            "time": "18:00",
            "duration_minutes": 60,
        }
    )

    result = await say(turn, "add gym tomorrow at 6 pm for an hour")

    assert result.reply == Reply("Done: gym tomorrow from 18:00 to 19:00.")
    assert titles_on(repos, TOMORROW) == ["gym"]
    assert isinstance(result.execution, Executed)
    assert isinstance(result.execution.outcome, EventCreated)


async def test_parser_receives_message_time_and_time_zone(parser, turn):
    parser.then({"tool": "none"})

    await turn.handle("hi", now=NOW, timezone="Europe/Rome")

    [request] = parser.requests
    assert (request.message, request.now, request.timezone) == ("hi", NOW, "Europe/Rome")
    assert request.history == ()


# --- the history -----------------------------------------------------------------------


async def test_missing_value_then_completion_rereads_the_last_exchange(repos, parser, turn):
    def model(request):
        if not request.history:
            return {"tool": "add_event", "title": "yoga"}
        assert request.history == (Exchange("add yoga", 'What time do you want "yoga"?'),)
        return {
            "tool": "add_event",
            "title": "yoga",
            "time": "19:00",
            "duration_minutes": 45,
        }

    parser.answer_with(model)

    first = await say(turn, "add yoga")
    second = await say(turn, "at 7 pm for 45 minutes")

    assert first.reply == Reply('What time do you want "yoga"?', expects_reply=True)
    assert second.reply.text == "Done: yoga today from 19:00 to 19:45."
    assert len(parser.requests) == 2 and parser.requests[1].history


async def test_history_always_carries_up_to_the_three_latest_exchanges(parser, turn):
    parser.then(
        {"tool": "add_event", "title": "one", "time": "20:00", "duration_minutes": 30},
        {"tool": "add_event", "title": "two"},  # question
        {"tool": "add_event", "title": "three"},  # question
        {"tool": "add_event", "title": "three", "time": "21:00", "duration_minutes": 30},
        {"tool": "none"},
    )

    for message in ("one", "two", "three", "at 9 pm", "thanks"):
        await say(turn, message)

    histories = [[(e.user, e.reply) for e in r.history] for r in parser.requests]
    one_done = "Done: one today from 20:00 to 20:30."
    two_asked = 'What time do you want "two"?'
    three_asked = 'What time do you want "three"?'
    three_done = "Done: three today from 21:00 to 21:30."
    assert histories[0] == []  # first turn: no history
    assert histories[1] == [("one", one_done)]
    assert histories[2] == [("one", one_done), ("two", two_asked)]
    assert histories[3] == [("one", one_done), ("two", two_asked), ("three", three_asked)]
    # fifth turn: "one" has left the window of the last 3
    assert histories[4] == [("two", two_asked), ("three", three_asked), ("at 9 pm", three_done)]


async def test_a_cold_yes_finds_nothing_to_confirm(repos, parser, turn):
    parser.then({"tool": "none"})

    result = await say(turn, "yes")

    assert "I can add, move or delete events" in result.reply.text
    assert parser.requests[0].history == ()
    assert repos.schedule.events_for_date(TODAY) == []


async def test_after_a_restart_the_history_is_empty(repos, parser):
    first = TurnService(repos, parser)
    parser.then({"tool": "add_event", "title": "yoga"}, {"tool": "none"})
    await say(first, "add yoga")

    restarted = TurnService(repos, parser)  # the service restarts: sessions are in memory
    await say(restarted, "at 7 pm")

    assert parser.requests[1].history == ()


async def test_a_failed_turn_now_enters_the_history_too(parser, turn):
    """The history no longer depends on `expects_reply`: even a turn that ended in an error now
    appears in the window of the last 3."""
    parser.then(ParseFailure("invalid_output"), {"tool": "none"})

    await say(turn, "asdf")
    await say(turn, "hi")

    assert parser.requests[1].history == (
        Exchange(user="asdf", reply="I didn't understand, can you rephrase?"),
    )


# --- proposals and confirmations -------------------------------------------------------


async def test_downstream_proposal_then_created_with_the_message_times(repos, parser, turn):
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )
    repos.event_defaults.upsert("shower", typical_duration_minutes=15)

    def model(request):
        if request.message == "yes" and 'Shall I also add "shower"' in last_reply(request):
            start, end = TIMES.findall(last_reply(request))[-2:]
            return {
                "tool": "add_event",
                "title": "shower",
                "date": TOMORROW,
                "time": start,
                "end_time": end,
            }
        return {
            "tool": "add_event",
            "title": "gym",
            "date": TOMORROW,
            "time": "18:00",
            "end_time": "19:00",
        }

    parser.answer_with(model)

    proposal = await say(turn, "add gym tomorrow at 6 pm")
    done = await say(turn, "yes")

    assert proposal.reply.quick_replies == ("Yes", "No") and proposal.reply.expects_reply
    assert done.reply.text == "Done: shower tomorrow from 19:00 to 19:15."
    shower, gym = by_title(repos, TOMORROW, "shower"), by_title(repos, TOMORROW, "gym")
    assert shower.derived_from_event_id == gym.event_id  # inferred provenance
    assert shower.derived_from_rule_id is not None


async def test_moving_the_event_proposes_moving_the_derived_one(repos, parser, turn):
    gym = add_event(repos, "gym", f"{TODAY}T18:00", f"{TODAY}T19:00")
    add_event(
        repos, "shower", f"{TODAY}T19:00", f"{TODAY}T19:15", derived_from_event_id=gym.event_id
    )

    def model(request):
        if request.message == "yes" and 'Shall I also move "shower"' in last_reply(request):
            start, end = TIMES.findall(last_reply(request))[-2:]
            return {
                "tool": "update_event",
                "event_description": "shower",
                "new_time": start,
                "new_end_time": end,
            }
        return {"tool": "update_event", "event_description": "gym", "shift_minutes": 30}

    parser.answer_with(model)

    proposal = await say(turn, "move the gym by half an hour")
    await say(turn, "yes")

    assert proposal.reply.text.endswith('Shall I also move "shower" from 19:30 to 19:45?')
    shower = by_title(repos, TODAY, "shower")
    assert (shower.start, shower.end) == (f"{TODAY}T19:30", f"{TODAY}T19:45")


async def test_deleting_the_event_proposes_deleting_the_derived_one(repos, parser, turn):
    gym = add_event(repos, "gym", f"{TODAY}T18:00", f"{TODAY}T19:00")
    add_event(
        repos, "shower", f"{TODAY}T19:00", f"{TODAY}T19:15", derived_from_event_id=gym.event_id
    )

    def model(request):
        if request.message == "yes" and 'Shall I also delete "shower"' in last_reply(request):
            start = TIMES.findall(last_reply(request))[-2]
            return {"tool": "remove_event", "event_description": "shower", "event_time": start}
        return {"tool": "remove_event", "event_description": "gym"}

    parser.answer_with(model)

    await say(turn, "delete the gym")
    done = await say(turn, "yes")

    assert done.reply.text == 'Done: I deleted "shower" (today 19:00-19:15).'
    assert repos.schedule.events_for_date(TODAY) == []


@pytest.mark.parametrize(("answer", "expected_start"), [("yes", "07:15"), ("no", "08:00")])
async def test_usual_gap_yes_or_no(repos, parser, turn, answer, expected_start):
    add_event(repos, "work", f"{TODAY}T09:00", f"{TODAY}T17:00")
    repos.event_rules.insert_rule(
        trigger_title="commute", related_title="work", position="before", gap_minutes=45
    )
    repos.event_rules.insert_rule(
        trigger_title="dinner", related_title="work", position="before", offset_minutes=0
    )
    repos.event_defaults.upsert("dinner", typical_duration_minutes=60)

    def model(request):
        if "Shall I keep the gap?" in last_reply(request):
            return {"tool": "add_event", "title": "dinner", "with_gap": request.message == "yes"}
        return {"tool": "add_event", "title": "dinner"}

    parser.answer_with(model)

    question = await say(turn, "add dinner")
    await say(turn, answer)

    assert "with the gap, at 08:00 without it. Shall I keep the gap?" in question.reply.text
    assert "at 07:15 with the gap" in question.reply.text
    assert by_title(repos, TODAY, "dinner").start == f"{TODAY}T{expected_start}"


async def test_anchor_not_in_schedule_answered_with_the_time(repos, parser, turn):
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )
    repos.event_defaults.upsert("shower", typical_duration_minutes=15)
    repos.event_defaults.upsert("gym", typical_duration_minutes=60)

    def model(request):
        if request.message == "at 6 pm" and 'so I can place "shower"' in last_reply(request):
            return {"tool": "add_event", "title": "shower", "anchor_time": "18:00"}
        return {"tool": "add_event", "title": "shower"}

    parser.answer_with(model)

    question = await say(turn, "add shower")
    done = await say(turn, "at 6 pm")

    assert question.reply.text.startswith('I can\'t find "gym" in the schedule today')
    assert done.reply.text == "Done: shower today from 19:00 to 19:15."
    assert titles_on(repos, TODAY) == ["shower"]  # the gym did not enter the schedule


async def test_missing_duration_answered_with_minutes(repos, parser, turn):
    add_event(repos, "gym", f"{TODAY}T18:00", f"{TODAY}T19:00")
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )

    def model(request):
        if request.message == "15 minutes" and 'How long does "shower"' in last_reply(request):
            return {"tool": "add_event", "title": "shower", "duration_minutes": 15}
        return {"tool": "add_event", "title": "shower"}

    parser.answer_with(model)

    question = await say(turn, "add shower")
    done = await say(turn, "15 minutes")

    assert question.reply.text == 'How long does "shower" usually last?'
    assert done.reply.text == "Done: shower today from 19:00 to 19:15."


async def test_same_name_events_chosen_with_a_quick_reply(repos, parser, turn):
    add_event(repos, "call with Marco", f"{TODAY}T16:00", f"{TODAY}T16:30")
    add_event(repos, "call with the client", f"{TODAY}T17:00", f"{TODAY}T17:30")

    def model(request):
        chosen = re.fullmatch(r"(.+) at (\d{2}:\d{2})", request.message)
        if chosen and "Which one do you mean?" in last_reply(request):
            return {
                "tool": "update_event",
                "event_description": chosen[1],
                "event_time": chosen[2],
                "shift_minutes": 5,
            }
        return {"tool": "update_event", "event_description": "call", "shift_minutes": 5}

    parser.answer_with(model)

    asked = await say(turn, "move the call by 5 minutes")
    pick = asked.reply.quick_replies[1]  # the user taps the second quick reply
    done = await say(turn, pick)

    assert pick == "call with the client at 17:00"
    assert done.reply.text == 'Done: I moved "call with the client" to today from 17:05 to 17:35.'
    assert by_title(repos, TODAY, "call with Marco").start == f"{TODAY}T16:00"  # untouched


async def test_a_conflict_is_reported_and_the_event_stays(repos, parser, turn):
    add_event(repos, "meeting", f"{TODAY}T16:00", f"{TODAY}T17:00")
    parser.then({"tool": "add_event", "title": "call", "time": "16:30", "duration_minutes": 60})

    result = await say(turn, "add call at 16:30")

    assert "Warning: it overlaps with meeting (16:00-17:00)." in result.reply.text
    assert sorted(titles_on(repos, TODAY)) == ["call", "meeting"]


# --- habits ----------------------------------------------------------------------------


def create_yoga(day):
    return {
        "tool": "add_event",
        "title": "yoga",
        "date": day,
        "time": "19:00",
        "duration_minutes": 45,
    }


async def three_yoga(turn, parser):
    parser.then(create_yoga("2026-09-22"), create_yoga("2026-09-23"), create_yoga("2026-09-24"))
    replies = [await say(turn, "add yoga at 7 pm") for _ in range(3)]
    return replies


async def test_third_time_the_usual_time_is_proposed_and_yes_saves_it(repos, parser, turn):
    first, second, third = await three_yoga(turn, parser)

    def model(request):
        proposal = re.search(r"around (\d{2}:\d{2}) \((\d+) minutes\)", last_reply(request))
        if request.message == "yes" and proposal:
            return {
                "tool": "add_event_rule",
                "trigger_title": "yoga",
                "typical_start": proposal[1],
                "typical_duration_minutes": int(proposal[2]),
            }
        return {"tool": "none"}

    parser.answer_with(model)
    accepted = await say(turn, "yes")

    assert not first.reply.expects_reply and not second.reply.expects_reply
    assert third.reply.text.endswith(
        'I noticed you often do "yoga" around 19:00 (45 minutes): shall I save it as a usual time?'
    )
    assert accepted.reply.text == 'Done: "yoga" usually is at 19:00 and lasts 45 minutes.'
    assert repos.event_defaults.get_by_title("yoga").typical_start == "19:00"
    assert repos.event_patterns.list_pending() == []  # closed by itself


async def test_no_to_a_suggestion_rejects_it_forever(repos, parser, turn):
    await three_yoga(turn, parser)
    parser.then({"tool": "dismiss_habit"}, create_yoga("2026-09-25"))

    refused = await say(turn, "no")
    later = await say(turn, "add yoga at 7 pm")

    assert refused.reply.text == "Ok, I won't suggest it again."
    assert not later.reply.expects_reply and "I noticed" not in later.reply.text
    rows = repos.conn.execute("SELECT status FROM event_patterns").fetchall()
    assert [row["status"] for row in rows] == [NoteStatus.REJECTED.value]


async def test_recurring_gap_is_saved_by_saying_the_name(repos, parser, turn):
    for day in (1, 2, 3):
        repos.gap_patterns.record_occurrence(
            related_title="work",
            position="after",
            observed_gap_minutes=30,
            time_tolerance_minutes=15,
            now=NOW.replace(day=day),
        )
    parser.then({"tool": "get_schedule"})  # a read does not append the proposal
    parser.then({"tool": "add_event", "title": "reading", "time": "21:00", "duration_minutes": 30})
    assert "I noticed" not in (await say(turn, "what do I have today?")).reply.text

    proposal = await say(turn, "add reading at 9 pm")

    def model(request):
        gap = re.search(r'about (\d+) minutes free after "(.+?)"', last_reply(request))
        assert gap and request.message == "break"
        return {
            "tool": "add_event_rule",
            "trigger_title": request.message,
            "related_title": gap[2],
            "position": "after",
            "gap_minutes": int(gap[1]),
        }

    parser.answer_with(model)
    saved = await say(turn, "break")

    assert proposal.reply.quick_replies == ("No",)
    assert saved.reply.text == 'Done: after "work" I will always leave a 30-minute gap.'
    [rule] = repos.event_rules.get_all()
    assert (rule.trigger_title, rule.related_title, rule.gap_minutes) == ("break", "work", 30)
    assert repos.gap_patterns.list_pending() == []


# --- errors: no exception crosses the steps --------------------------------------------


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (ParseFailure("timeout"), "The model is not answering: try again shortly."),
        (ParseFailure("unavailable"), "The model is not answering: try again shortly."),
        (ParseFailure("invalid_output", "broken json"), "I didn't understand, can you rephrase?"),
    ],
)
async def test_a_model_that_does_not_answer_gives_a_fixed_message(
    repos, parser, turn, failure, message
):
    parser.then(failure, {"tool": "none"})

    failed = await say(turn, "add yoga at 7 pm")
    after = await say(turn, "hi")

    assert failed.reply == Reply(message)
    assert repos.schedule.events_for_date(TODAY) == []
    assert "I can add" in after.reply.text  # the next turn works


async def test_invalid_model_output_is_a_request_to_rephrase(parser, turn):
    parser.then({"tool": "add_event", "time": "tomorrow at 6 pm"})

    result = await say(turn, "add yoga tomorrow at 6 pm")

    assert result.reply.text == "I didn't understand, can you rephrase?"


async def test_a_parser_that_raises_does_not_stop_the_turn(parser, turn, caplog):
    def broken(request):
        raise RuntimeError("internal secret")

    parser.answer_with(broken)
    with caplog.at_level(logging.ERROR, logger="daily.agent"):
        result = await say(turn, "add yoga")

    assert result.reply.text == "The model is not answering: try again shortly."
    assert "internal secret" not in result.reply.text
    assert "The parse step raised" in caplog.text


async def test_an_execution_error_writes_nothing_and_does_not_block_later_turns(
    repos, parser, turn, monkeypatch, caplog
):
    from daily.agent import turn as turn_module

    calls = {"n": 0}
    real = turn_module.execute

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("broken")
        return real(*args, **kwargs)

    monkeypatch.setattr(turn_module, "execute", flaky)
    parser.then(create_yoga(TODAY), create_yoga(TODAY))

    with caplog.at_level(logging.ERROR, logger="daily.agent"):
        failed = await say(turn, "add yoga")
        recovered = await say(turn, "add yoga")

    assert failed.reply == Reply("Something went wrong: try again.")
    assert isinstance(failed.execution, ExecutionError)
    assert recovered.reply.text == "Done: yoga today from 19:00 to 19:45."
    assert titles_on(repos, TODAY) == ["yoga"]


async def test_if_agno_itself_fails_the_turn_still_replies(repos, parser, monkeypatch, caplog):
    turn = TurnService(repos, parser)

    async def boom(*args, **kwargs):
        raise RuntimeError("Agno went down")

    monkeypatch.setattr(turn._workflow, "arun", boom)
    with caplog.at_level(logging.ERROR, logger="daily.agent"):
        result = await say(turn, "hi")

    assert result.reply == Reply("Something went wrong: try again.")
    assert "The turn workflow raised" in caplog.text


async def test_a_parser_returning_nothing_becomes_a_handled_error(parser, turn):
    async def nothing(request):
        return None  # it breaks the contract: execution must not collapse

    parser.parse = nothing

    result = await say(turn, "hi")

    assert result.reply == Reply("Something went wrong: try again.")
    assert isinstance(result.execution, ExecutionError)


# --- concurrency and isolation ---------------------------------------------------------


async def test_turns_are_serialized(parser, turn):
    events: list[str] = []

    def slow(request):
        events.append(f"start {request.message}")
        return {"tool": "none"}

    async def parse(request):
        result = slow(request)
        await asyncio.sleep(0.02)
        events.append(f"end {request.message}")
        from scripted import _as_result

        return _as_result(result)

    parser.parse = parse
    await asyncio.gather(say(turn, "a"), say(turn, "b"), say(turn, "c"))

    # never two turns at once: every "start" is followed by its own "end"
    assert events == ["start a", "end a", "start b", "end b", "start c", "end c"]


async def test_two_services_do_not_share_history(repos):
    parser_a = ScriptedParser([{"tool": "add_event", "title": "yoga"}])
    parser_b = ScriptedParser([{"tool": "none"}])
    service_a, service_b = TurnService(repos, parser_a), TurnService(repos, parser_b)

    await say(service_a, "add yoga")
    await say(service_b, "at 7 pm")

    assert parser_b.requests[0].history == ()


# --- turn log and history cap ---------------------------------------------------------


async def test_every_turn_is_recorded_in_turn_log(repos, parser, turn):
    parser.then({"tool": "add_event", "title": "yoga", "time": "19:00", "duration_minutes": 45})

    result = await say(turn, "add yoga at 7 pm for 45 minutes")

    assert result.turn_id is not None
    entry = read_turn(repos.conn, result.turn_id)
    assert entry is not None
    assert entry.message == "add yoga at 7 pm for 45 minutes"
    assert entry.tool == "add_event"
    assert entry.call_json["title"] == "yoga"
    assert entry.kind == "event_created"
    assert entry.reply_text == result.reply.text
    assert entry.flag_note is None


async def test_every_turn_records_its_duration(repos, parser, turn):
    parser.then({"tool": "none"})

    result = await say(turn, "hi")

    entry = read_turn(repos.conn, result.turn_id)
    assert entry.duration_ms is not None
    assert entry.duration_ms >= 0


async def test_a_turn_without_valid_tool_call_is_recorded_anyway(parser, turn, repos):
    parser.then({"tool": "none"})

    result = await say(turn, "hi")

    entry = read_turn(repos.conn, result.turn_id)
    assert entry.tool is None
    assert entry.call_json is None
    assert entry.kind == "no_action"


async def test_a_parse_failure_is_recorded(parser, turn, repos):
    parser.then(ParseFailure("timeout"))

    result = await say(turn, "asdf")

    entry = read_turn(repos.conn, result.turn_id)
    assert entry.tool is None
    assert entry.kind == "parse_failed:timeout"


async def test_a_log_failure_does_not_stop_the_turn(repos, parser, turn, monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("broken")

    monkeypatch.setattr(repos.turn_log, "record", boom)
    parser.then({"tool": "none"})

    with caplog.at_level(logging.ERROR, logger="daily.agent"):
        result = await say(turn, "hi")

    assert result.reply.text != ""
    assert result.turn_id is None
    assert "The turn log raised" in caplog.text


async def test_history_stays_at_n_runs_oldest_disappear(parser, turn):
    parser.then({"tool": "none"}, {"tool": "none"}, {"tool": "none"}, {"tool": "none"})

    for message in ("one", "two", "three", "four"):
        await say(turn, message)

    runs = turn._workflow.db.get_runs(session_id=turn._session_id)
    assert len(runs) == 3
    assert [run.input for run in runs] == ["two", "three", "four"]


async def test_history_pruning_that_raises_does_not_stop_the_turn(
    parser, turn, monkeypatch, caplog
):
    def boom(*args, **kwargs):
        raise RuntimeError("broken")

    monkeypatch.setattr(turn._workflow.db, "get_runs", boom)
    parser.then({"tool": "none"})

    with caplog.at_level(logging.ERROR, logger="daily.agent"):
        result = await say(turn, "hi")

    assert result.reply.text != ""
    assert "The history pruning raised" in caplog.text


async def test_a_no_to_a_proposal_gets_ok_but_not_when_cold(repos, parser, turn):
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )
    repos.event_defaults.upsert("shower", typical_duration_minutes=15)
    parser.then(
        {
            "tool": "add_event",
            "title": "gym",
            "time": "18:00",
            "duration_minutes": 60,
        },
        {"tool": "none"},
        {"tool": "none"},
    )

    await say(turn, "add gym at 6 pm")
    declined = await say(turn, "no thanks")
    cold = await say(turn, "no thanks")

    assert declined.reply == Reply("Ok.")
    assert cold.reply.text.startswith("I didn't understand what to do")
    assert titles_on(repos, TODAY) == ["gym"]
