from datetime import datetime

import pytest

from daily.storage.turn_log_repo import AmbiguousTurnId, TurnLogRepository
from turn_log import read_turn

NOW = datetime(2026, 9, 22, 10, 0)


@pytest.fixture
def repo(conn):
    return TurnLogRepository(conn)


def test_record_writes_every_field(repo, conn):
    turn_id = repo.record(
        message="add gym at 6 pm",
        tool="add_event",
        call_json={"tool": "add_event", "title": "gym"},
        kind="event_created",
        reply_text='Added "gym" at 18:00.',
        now=NOW,
    )

    entry = read_turn(conn, turn_id)

    assert entry is not None
    assert entry.message == "add gym at 6 pm"
    assert entry.tool == "add_event"
    assert entry.call_json == {"tool": "add_event", "title": "gym"}
    assert entry.kind == "event_created"
    assert entry.reply_text == 'Added "gym" at 18:00.'
    assert entry.flag_note is None


def test_record_without_tool_call(repo, conn):
    turn_id = repo.record(
        message="hi",
        tool=None,
        call_json=None,
        kind="no_action",
        reply_text="Ok.",
        now=NOW,
    )

    entry = read_turn(conn, turn_id)

    assert entry.tool is None
    assert entry.call_json is None


def test_record_saves_the_duration(repo, conn):
    turn_id = repo.record(
        message="hi",
        tool=None,
        call_json=None,
        kind="no_action",
        reply_text="Ok.",
        duration_ms=1234,
        now=NOW,
    )

    assert read_turn(conn, turn_id).duration_ms == 1234


def test_record_without_duration_stays_none(repo, conn):
    turn_id = repo.record(
        message="hi", tool=None, call_json=None, kind="no_action", reply_text="Ok.", now=NOW
    )

    assert read_turn(conn, turn_id).duration_ms is None


def test_flag_sets_the_note(repo, conn):
    turn_id = repo.record(
        message="x", tool=None, call_json=None, kind="no_action", reply_text="Ok.", now=NOW
    )

    updated = repo.flag(turn_id, "it misunderstood the time")

    assert updated is True
    assert read_turn(conn, turn_id).flag_note == "it misunderstood the time"


def test_flag_of_unknown_turn_id_is_false(repo):
    assert repo.flag("never-existed", "note") is False


def test_flag_saves_context_of_up_to_three_turns_including_the_flagged_one(repo, conn):
    ids = [
        repo.record(
            message=m,
            tool=None,
            call_json=None,
            kind="no_action",
            reply_text=f"reply {m}",
            now=NOW,
        )
        for m in ("one", "two", "three", "four")
    ]

    repo.flag(ids[3], "note")

    context = read_turn(conn, ids[3]).flag_context
    assert [c["turn_id"] for c in context] == ids[1:4]  # "one" is outside the window
    assert [c["message"] for c in context] == ["two", "three", "four"]
    assert context[-1]["reply_text"] == "reply four"


def test_flag_of_the_first_turn_has_a_single_item_context(repo, conn):
    turn_id = repo.record(
        message="hi", tool=None, call_json=None, kind="no_action", reply_text="Ok.", now=NOW
    )

    repo.flag(turn_id, "note")

    context = read_turn(conn, turn_id).flag_context
    assert [c["turn_id"] for c in context] == [turn_id]


def test_flag_of_a_non_last_turn_does_not_include_later_ones(repo, conn):
    ids = [
        repo.record(
            message=m, tool=None, call_json=None, kind="no_action", reply_text="ok", now=NOW
        )
        for m in ("one", "two", "three")
    ]

    repo.flag(ids[0], "note")

    context = read_turn(conn, ids[0]).flag_context
    assert [c["turn_id"] for c in context] == [ids[0]]


def test_flag_of_unknown_turn_id_writes_no_context(repo):
    assert repo.flag("never-existed", "note") is False


# --- handling the reports ---------------------------------------------------------------


def flagged(repo, message, note="wrong"):
    turn_id = repo.record(
        message=message, tool=None, call_json=None, kind="no_action", reply_text="Ok.", now=NOW
    )
    repo.flag(turn_id, note)
    return turn_id


def test_reports_are_the_open_flagged_turns_oldest_first(repo):
    first = flagged(repo, "a")
    repo.record(message="b", tool=None, call_json=None, kind="no_action", reply_text="Ok.")
    second = flagged(repo, "c", note="")  # a report without a note is still a report

    assert [r.turn_id for r in repo.reports()] == [first, second]
    assert repo.reports()[0].note == "wrong"


def test_a_handled_report_is_no_longer_open(repo):
    first = flagged(repo, "a")
    second = flagged(repo, "b")

    assert repo.handle(first[:8], "code", now=NOW) == first

    assert [r.turn_id for r in repo.reports()] == [second]
    [handled, _] = repo.reports(include_handled=True)
    assert (handled.handled_at, handled.outcome) == ("2026-09-22T10:00:00", "code")


def test_an_assigned_report_stays_open_with_where_its_fix_goes(repo):
    first = flagged(repo, "a")
    second = flagged(repo, "b")
    repo.handle(second, "code", now=NOW)

    assert repo.assign(first, "model") == first
    assert repo.assign(second, "model") == second  # a closed one opens again

    assert [(r.turn_id, r.outcome, r.handled_at) for r in repo.reports()] == [
        (first, "model", None),
        (second, "model", None),
    ]


def test_reports_can_be_filtered_by_where_their_fix_goes(repo):
    first = flagged(repo, "a")
    second = flagged(repo, "b")
    flagged(repo, "c")
    repo.assign(first, "model")
    repo.handle(second, "model", now=NOW)

    assert [r.turn_id for r in repo.reports(outcome="model")] == [first]
    assert [r.turn_id for r in repo.reports(include_handled=True, outcome="model")] == [
        first,
        second,
    ]


def test_handle_of_an_unknown_or_unflagged_turn_is_none(repo):
    plain = repo.record(message="a", tool=None, call_json=None, kind="no_action", reply_text="Ok.")

    assert repo.handle("never", "code") is None
    assert repo.handle(plain, "code") is None


def test_handle_with_an_ambiguous_prefix_raises(repo):
    flagged(repo, "a")
    flagged(repo, "b")

    with pytest.raises(AmbiguousTurnId):
        repo.handle("", "model")


def test_a_new_report_on_a_handled_turn_opens_it_again(repo):
    turn_id = flagged(repo, "a")
    repo.handle(turn_id, "prompt")

    repo.flag(turn_id, "still wrong")

    [report] = repo.reports()
    assert (report.note, report.handled_at, report.outcome) == ("still wrong", None, None)


def test_the_outcome_must_be_a_known_one(repo, conn):
    import sqlite3

    turn_id = flagged(repo, "a")

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE turn_log SET flag_outcome = 'other' WHERE turn_id = ?", (turn_id,))
