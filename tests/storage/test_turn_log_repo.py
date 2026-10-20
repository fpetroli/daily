from datetime import datetime

import pytest

from daily.storage.turn_log_repo import TurnLogRepository
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
