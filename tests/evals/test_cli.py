import json
import textwrap

from daily.evals.__main__ import main

GOOD = """
- id: good
  say: "add yoga at 7 pm for 45 minutes"
  call: {tool: add_event, title: yoga, time: "19:00", duration_minutes: 45}
  added: [[yoga, "{today}T19:00", "{today}T19:45"]]
"""
BAD = """
- id: wrong
  say: "x"
  call: {tool: none}
  added: [[ghost, "{today}T19:00", "{today}T19:45"]]
"""


def folder(tmp_path, *files):
    for name, text in files:
        (tmp_path / name).write_text(textwrap.dedent(text), encoding="utf-8")
    return tmp_path


def test_runs_and_prints_the_report(tmp_path, capsys):
    cases = folder(tmp_path, ("creation.yaml", GOOD))

    code = main(["run", "--parser", "oracle", "--cases", str(cases)])

    out = capsys.readouterr().out
    assert code == 0
    assert "Cases: 1 · correct: 1 (100.0%)" in out


def test_fail_under_exits_with_one_below_threshold(tmp_path, capsys):
    cases = folder(tmp_path, ("creation.yaml", GOOD + BAD))

    assert main(["run", "--cases", str(cases), "--fail-under", "100"]) == 1
    assert main(["run", "--cases", str(cases), "--fail-under", "50"]) == 0
    assert "wrong" in capsys.readouterr().out


def test_holdout_set_runs_only_when_requested(tmp_path, capsys):
    holdout = GOOD.replace("id: good", "id: held\n  split: holdout")
    cases = folder(tmp_path, ("creation.yaml", GOOD), ("held.yaml", holdout))

    main(["run", "--cases", str(cases)])
    assert "Cases: 1 ·" in capsys.readouterr().out

    main(["run", "--cases", str(cases), "--split", "holdout"])
    assert "Cases: 1 ·" in capsys.readouterr().out

    main(["run", "--cases", str(cases), "--split", "all"])
    captured = capsys.readouterr()
    assert "Cases: 2 ·" in captured.out
    assert "includes the held-out set" in captured.err


def test_filters_by_section_and_writes_json(tmp_path, capsys):
    cases = folder(tmp_path, ("creation.yaml", GOOD), ("other.yaml", BAD.replace("wrong", "b2")))
    out = tmp_path / "result.json"

    code = main(["run", "--cases", str(cases), "--section", "creation", "--json", str(out)])

    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [c["id"] for c in data["cases"]] == ["good"]


def test_invalid_cases_or_no_case_selected_give_2(tmp_path, capsys):
    broken = folder(tmp_path, ("broken.yaml", '- {id: x, say: "y", other: 1}'))
    assert main(["run", "--cases", str(broken)]) == 2
    assert "Invalid cases" in capsys.readouterr().err

    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["run", "--cases", str(empty)]) == 2
    assert main(["run", "--cases", str(tmp_path / "missing")]) == 2


def test_with_the_model_off_every_turn_is_a_failed_parse_and_the_case_fails(tmp_path, capsys):
    cases = folder(tmp_path, ("creation.yaml", GOOD))
    dead = "http://127.0.0.1:1/v1"  # no server listening

    code = main(["run", "--parser", "llama", "--llm-url", dead, "--cases", str(cases),
                 "--fail-under", "100"])  # fmt: skip

    captured = capsys.readouterr()
    assert code == 1
    assert "Cases: 1 · correct: 0 (0.0%)" in captured.out
    assert "[1/1] good WRONG" in captured.err


def test_a_single_case_can_be_chosen(tmp_path, capsys):
    cases = folder(tmp_path, ("creation.yaml", GOOD + BAD))

    main(["run", "--cases", str(cases), "--case", "good"])

    assert "Cases: 1 · correct: 1" in capsys.readouterr().out
