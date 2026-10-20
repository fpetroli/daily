import pytest

from daily.storage.rules_repo import EventRulesRepository


@pytest.fixture
def repo(conn):
    return EventRulesRepository(conn)


def test_find_by_trigger_finds_rules_where_the_title_is_dependent(repo):
    repo.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )

    rules = repo.find_by_trigger("shower")

    assert len(rules) == 1
    assert rules[0].related_title == "gym"
    assert rules[0].is_gap is False


def test_find_by_trigger_is_case_insensitive(repo):
    repo.insert_rule(
        trigger_title="Shower", related_title="gym", position="after", offset_minutes=0
    )

    assert len(repo.find_by_trigger("Shower")) == 1


def test_find_by_related_finds_downstream_candidates(repo):
    repo.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )
    repo.insert_rule(
        trigger_title="stretching", related_title="gym", position="before", offset_minutes=10
    )

    candidates = repo.find_by_related("gym")

    assert {r.trigger_title for r in candidates} == {"shower", "stretching"}


def test_find_gaps_for_only_gap_rows_of_same_anchor_and_position(repo):
    repo.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )
    repo.insert_rule(trigger_title="commute", related_title="gym", position="after", gap_minutes=45)
    repo.insert_rule(
        trigger_title="commute", related_title="gym", position="before", gap_minutes=20
    )

    gaps = repo.find_gaps_for("gym", "after")

    assert len(gaps) == 1
    assert gaps[0].trigger_title == "commute"
    assert gaps[0].gap_minutes == 45
    assert gaps[0].is_gap is True


def test_find_matching_empty_single_multiple(repo):
    assert repo.find_matching("shower", "gym") == []

    repo.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )
    assert len(repo.find_matching("shower", "gym")) == 1

    repo.insert_rule(
        trigger_title="shower", related_title="gym", position="before", offset_minutes=5
    )
    assert len(repo.find_matching("shower", "gym")) == 2


def test_get_all_returns_all_rules(repo):
    repo.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )
    repo.insert_rule(
        trigger_title="commute", related_title="call", position="after", gap_minutes=20
    )

    assert {r.trigger_title for r in repo.get_all()} == {"shower", "commute"}


def test_delete_rule_removes_the_row(repo):
    rule_id = repo.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )

    repo.delete_rule(rule_id)

    assert repo.find_by_trigger("shower") == []


def test_update_rule_changes_only_the_given_fields(repo):
    rule_id = repo.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=5
    )

    repo.update_rule(rule_id, offset_minutes=10)
    rule = repo.find_by_trigger("shower")[0]

    assert rule.offset_minutes == 10
    assert rule.position == "after"  # unchanged
