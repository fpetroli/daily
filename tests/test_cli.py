import io
import subprocess
import sys
from datetime import datetime

from daily.cli import main, run_chat

NOW = datetime(2026, 9, 21, 15, 30)


async def chat(repos, *lines) -> str:
    out = io.StringIO()
    await run_chat(lines, out, repos=repos, now=NOW)
    return out.getvalue()


async def test_a_json_tool_call_goes_through_the_whole_turn(repos):
    output = await chat(
        repos,
        '{"tool": "add_event", "title": "yoga", "date": "2026-09-22", "time": "19:00", '
        '"duration_minutes": 45}',
        '{"tool": "get_schedule", "date": "2026-09-22"}',
    )

    assert output.splitlines() == [
        "> Done: yoga tomorrow from 19:00 to 19:45.",
        "> Tomorrow · Tuesday, September 22",
        "",
        "  yoga",
        "  19:00 → 19:45 · 45 min",
    ]


async def test_quick_replies_and_history_work_as_in_the_real_turn(repos):
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )
    repos.event_defaults.upsert("shower", typical_duration_minutes=15)

    output = await chat(
        repos,
        '{"tool": "add_event", "title": "gym", "date": "2026-09-22", "time": "18:00", '
        '"end_time": "19:00"}',
    )

    assert 'Shall I also add "shower" from 19:00 to 19:15?' in output
    assert output.splitlines()[-1] == "  [Yes | No]"


async def test_a_non_json_line_warns_and_does_not_run(repos):
    output = await chat(repos, "add yoga tomorrow at 7 pm")

    assert output.startswith("! No model connected")
    assert repos.schedule.events_for_date("2026-09-22") == []


async def test_invalid_json_asks_to_rephrase(repos):
    output = await chat(repos, '{"tool": "invented"}')

    assert output == "> I didn't understand, can you rephrase?\n"


async def test_empty_lines_are_ignored(repos):
    assert await chat(repos, "", "   ") == ""


def test_command_runs_from_the_terminal_with_an_in_memory_database():
    result = subprocess.run(
        [sys.executable, "-m", "daily.cli", "chat", "--now", "2026-09-21T15:30"],
        input='{"tool": "get_schedule"}\n',
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )

    assert result.stdout == "> Today · Monday, September 21\n\n  Nothing planned.\n"
    assert "Simulated time: 2026-09-21 15:30" in result.stderr


def test_main_rejects_an_unknown_command(capsys):
    import pytest

    with pytest.raises(SystemExit) as exit_info:
        main(["other"])

    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_reports_are_listed_and_handled_from_the_terminal(tmp_path, capsys):
    from daily.services.context import Repositories
    from daily.storage.db import open_database

    path = tmp_path / "daily.db"
    repos = Repositories.from_connection(open_database(path))
    turn_id = repos.turn_log.record(
        message="add gym", tool=None, call_json=None, kind="no_action", reply_text="Ok."
    )
    repos.turn_log.flag(turn_id, "it did nothing")
    repos.conn.close()

    assert main(["reports", "--db", str(path)]) == 0
    listed = capsys.readouterr().out
    assert main(["assign", "--db", str(path), turn_id[:8], "model"]) == 0
    assigned = capsys.readouterr().out
    assert main(["reports", "--db", str(path), "--outcome", "model"]) == 0
    to_fix = capsys.readouterr().out
    assert main(["handle", "--db", str(path), turn_id[:8], "model"]) == 0
    handled = capsys.readouterr().out
    assert main(["reports", "--db", str(path)]) == 0
    after = capsys.readouterr().out
    assert main(["handle", "--db", str(path), "zzzz", "code"]) == 1

    assert (
        turn_id[:8] in listed and "  You:  add gym" in listed and "  Note: it did nothing" in listed
    )
    assert assigned == f"{turn_id}: to fix → model.\n"
    assert "[to fix → model]" in to_fix
    assert handled == f"{turn_id}: fixed → model.\n"
    assert after == "No open reports.\n"
