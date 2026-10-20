import textwrap
from datetime import datetime

import pytest

from daily.evals.cases import Case, CaseError, load_cases


def write(tmp_path, text, name="sample.yaml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def load_one(tmp_path, text) -> Case:
    [case] = load_cases(write(tmp_path, text))
    return case


def test_a_single_message_case(tmp_path):
    case = load_one(
        tmp_path,
        """
        - id: c1
          say: "add gym tomorrow at 6 pm"
          call: {tool: add_event, title: gym, date: "{tomorrow}", time: "18:00"}
          added: [[gym, "{tomorrow}T18:00", "{tomorrow}T19:00"]]
          kind: event_created
        """,
    )

    assert (case.id, case.section, case.split) == ("c1", "sample", "dev")
    assert case.now == datetime(2026, 9, 21, 15, 30)
    [turn] = case.turns
    assert turn.say == "add gym tomorrow at 6 pm"  # the text is not expanded
    assert turn.call["date"] == "2026-09-22" and turn.call["time"] == "18:00"
    assert turn.kind == "event_created"
    assert case.expect.added == (("gym", "2026-09-22T18:00", "2026-09-22T19:00"),)


def test_a_multi_message_case(tmp_path):
    case = load_one(
        tmp_path,
        """
        - id: c2
          turns:
            - {say: "one", call: {tool: none}}
            - {say: "two", quick_replies: ["Yes", "No"], reply_has: ["x"]}
        """,
    )

    assert [t.say for t in case.turns] == ["one", "two"]
    assert case.turns[1].quick_replies == ("Yes", "No") and case.turns[1].reply_has == ("x",)
    assert case.turns[0].quick_replies is None


def test_symbolic_dates_are_relative_to_now(tmp_path):
    case = load_one(
        tmp_path,
        """
        - id: c3
          now: "2026-09-23T08:00"
          say: "x"
          seed:
            events:
              - [a, "{today}T09:00", "{yesterday}T10:00"]
              - [b, "{+14}T09:00", "{day_after_tomorrow}T09:30"]
          added:
            - [c, "{mon}T09:00", "{fri}T09:30"]
            - [d, "{sun}T09:00", "{-1}T09:30"]
        """,
    )

    seed = [(e.start, e.end) for e in case.seed.events]
    assert seed == [
        ("2026-09-23T09:00", "2026-09-22T10:00"),
        ("2026-10-07T09:00", "2026-09-25T09:30"),
    ]
    # Wednesday 23rd: next Monday is the 28th, next Friday the 25th, Sunday the 27th
    assert case.expect.added[0][1:] == ("2026-09-28T09:00", "2026-09-25T09:30")
    assert case.expect.added[1][1:] == ("2026-09-27T09:00", "2026-09-22T09:30")


def test_the_same_weekday_is_always_after_today(tmp_path):
    case = load_one(
        tmp_path,
        """
        - id: c4
          say: "x"
          added: [[a, "{mon}T09:00", "{mon}T10:00"]]
        """,
    )  # today is Monday: {mon} is next Monday

    assert case.expect.added[0][1] == "2026-09-28T09:00"


def test_seed_with_parent_rules_and_usual_times(tmp_path):
    case = load_one(
        tmp_path,
        """
        - id: c5
          say: "x"
          seed:
            events:
              - [gym, "{today}T18:00", "{today}T19:00"]
              - {title: shower, start: "{today}T19:00", end: "{today}T19:15", parent: gym}
            rules: [[shower, gym, after, 0], [break, work, after, null, 30]]
            defaults: [[shower, null, 15], [yoga, "19:00"]]
          defaults: {yoga: ["19:00", 45]}
          rules_added: [[a, b]]
        """,
    )

    assert case.seed.events[1].parent == "gym"
    assert (case.seed.rules[0].offset, case.seed.rules[1].gap) == (0, 30)
    assert case.seed.defaults[1] == case.seed.defaults[1].__class__("yoga", "19:00", None)
    assert case.expect.defaults == (("yoga", "19:00", 45),)
    assert case.expect.rules_added == (("a", "b"),)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('- {id: c, say: "x", other: 1}', "unknown keys"),
        ('- {id: c, say: "x", turns: [{say: "y"}]}', "not both"),
        ("- {id: c}", "not both nor neither"),
        ('- {say: "x"}', "missing id"),
        ('- {id: c, say: "x", split: trial}', "invalid split"),
        ('- {id: c, say: "x", added: [[a, "tomorrow", "{today}T10:00"]]}', "invalid time"),
        ('- {id: c, say: "x", call: {tool: invented}}', "c:"),
        ('- {id: c, say: "x", call: {tool: add_event, time: "wrong"}}', "c:"),
        ('- {id: c, say: "x", seed: {events: [[a, "{yesterday", "{today}T10:00"]]}}', "c:"),
        ('- {id: c, say: "x", added: [[a, "{soon}T10:00", "{today}T11:00"]]}', "unknown"),
        ("- {id: c, turns: [{call: {tool: none}}]}", "invalid turn"),
    ],
)
def test_a_badly_written_case_stops_with_its_id(tmp_path, body, message):
    with pytest.raises(CaseError, match=message):
        load_cases(write(tmp_path, body))


def test_duplicate_ids_and_non_list_file(tmp_path):
    write(tmp_path, '- {id: a, say: "x"}', "one.yaml")
    write(tmp_path, '- {id: a, say: "y"}', "two.yaml")
    with pytest.raises(CaseError, match="duplicate id"):
        load_cases(tmp_path)

    with pytest.raises(CaseError, match="list"):
        load_cases(write(tmp_path, "id: a", "three.yaml"))


def test_a_folder_has_one_section_per_file(tmp_path):
    write(tmp_path, '- {id: a, say: "x"}', "creation.yaml")
    write(tmp_path, '- {id: b, say: "y", section: other}', "reads.yaml")

    cases = load_cases(tmp_path)

    assert {c.id: c.section for c in cases} == {"a": "creation", "b": "other"}
