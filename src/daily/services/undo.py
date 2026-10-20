"""Undo of the last write on the schedule.

A single level: it is not a stack, the consumed row is deleted. It acts on the repositories
directly, never through the model (an undo is not text to interpret): the model is a parser,
not an actor. The data to restore is the one `agent/execute.py` recorded in `action_log`.
"""

from dataclasses import dataclass
from typing import Literal

from daily.services.context import Repositories


@dataclass(frozen=True, slots=True)
class Undone:
    """The write that was undone, with the event as it is now restored (or as it was, for an
    undone creation)."""

    kind: Literal["add_event", "update_event", "remove_event"]
    title: str
    start: str
    end: str


def undo_last_action(repos: Repositories) -> Undone | None:
    """Undoes the last recorded write, if any. `None` if there is nothing to undo."""
    entry = repos.action_log.latest()
    if entry is None:
        return None
    data = entry.undo_data

    match entry.kind:
        case "add_event":
            repos.schedule.delete_event(data["event_id"])
        case "update_event":
            repos.schedule.update_event(
                data["event_id"], title=data["title"], start=data["start"], end=data["end"]
            )
        case "remove_event":
            repos.schedule.insert_event(
                event_id=data["event_id"], title=data["title"], start=data["start"], end=data["end"]
            )
        case _:
            raise AssertionError(f"Unknown logged action kind: {entry.kind!r}")

    repos.action_log.delete(entry.log_id)
    return Undone(entry.kind, data["title"], data["start"], data["end"])
