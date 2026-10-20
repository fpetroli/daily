from daily.services.undo import Undone, undo_last_action

from .conftest import DAY, add_event


def log(repos, kind, event, **values):
    data = {
        "event_id": event.event_id,
        "title": event.title,
        "start": event.start,
        "end": event.end,
    }
    repos.action_log.record(kind=kind, summary="test", undo_data=data | values)


def test_nothing_to_undo(repos):
    assert undo_last_action(repos) is None


def test_undo_of_a_creation_deletes_the_event(repos):
    gym = add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    log(repos, "add_event", gym)

    undone = undo_last_action(repos)

    assert undone == Undone("add_event", "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    assert repos.schedule.get_by_id(gym.event_id) is None


def test_undo_of_a_change_restores_the_previous_values(repos):
    gym = add_event(repos, "gym", f"{DAY}T19:00", f"{DAY}T20:00")
    log(repos, "update_event", gym, start=f"{DAY}T18:00", end=f"{DAY}T19:00")

    undone = undo_last_action(repos)

    assert undone == Undone("update_event", "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    restored = repos.schedule.get_by_id(gym.event_id)
    assert (restored.start, restored.end) == (f"{DAY}T18:00", f"{DAY}T19:00")


def test_undo_of_a_deletion_restores_the_event_with_the_same_id(repos):
    gym = add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    repos.schedule.delete_event(gym.event_id)
    log(repos, "remove_event", gym)

    undo_last_action(repos)

    assert repos.schedule.get_by_id(gym.event_id) == gym


def test_undo_is_a_single_level(repos):
    gym = add_event(repos, "gym", f"{DAY}T18:00", f"{DAY}T19:00")
    log(repos, "add_event", gym)

    undo_last_action(repos)

    assert undo_last_action(repos) is None
