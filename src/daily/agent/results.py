"""The outcomes of the execution step: everything the reply step must be able to tell.

A turn always ends in one of these outcomes; no exception crosses the steps. `Executed`
carries the outcome of a service (or a view) together with the tool call that produced it,
because messages are self-contained and quote titles and times.
"""

import re
from dataclasses import dataclass

from daily.agent.parser import ParseFailure
from daily.agent.schemas import ToolCall
from daily.domain.timeband import TimeBand
from daily.services.habits import HabitProposal
from daily.services.types import FreeSlot, ScheduleEntry


@dataclass(frozen=True, slots=True)
class ScheduleView:
    """Schedule of one day, with the requested band: to describe it ("this afternoon").
    `following` is the first event of the next day, shown after the whole day's events."""

    date: str
    time_band: TimeBand | None
    entries: tuple[ScheduleEntry, ...]
    following: ScheduleEntry | None = None


@dataclass(frozen=True, slots=True)
class FreeSlotsView:
    date: str
    time_band: TimeBand | None
    min_duration_minutes: int | None
    slots: tuple[FreeSlot, ...]


@dataclass(frozen=True, slots=True)
class HabitDismissed:
    """The rejected suggestion, if one was pending."""

    proposal: HabitProposal | None


@dataclass(frozen=True, slots=True)
class Executed:
    """An executed tool call. `outcome` is a services outcome or one of the views above.

    `habit` is a habit proposal to append to the reply, if the threshold was hit and the turn
    does not already have a question of its own.
    """

    call: ToolCall
    outcome: object
    habit: HabitProposal | None = None


@dataclass(frozen=True, slots=True)
class MissingField:
    """A value with no reasonable fallback: the user is asked.

    `field` is the name of the tool call field; `subject` is the title or description of the
    event the question is about, if known.
    """

    call: ToolCall
    field: str
    subject: str | None = None


@dataclass(frozen=True, slots=True)
class NoAction:
    """The model chose "no action": small talk, or an answer with no question.

    `after_question`: the last reply was waiting for an answer, so a "no" or a "never mind"
    ends here and the reply is a plain "Ok." instead of "I did not understand".
    """

    after_question: bool = False


@dataclass(frozen=True, slots=True)
class ParseFailed:
    failure: ParseFailure


@dataclass(frozen=True, slots=True)
class ExecutionError:
    """An unexpected exception during execution: already logged, never shown."""


Execution = Executed | MissingField | NoAction | ParseFailed | ExecutionError


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def kind_of(execution: Execution | None) -> str:
    """The "kind of reply" of a turn, derived from the outcome: independent of the wording.

    In the cases it is written in snake_case: `event_created`, `event_updated`,
    `event_removed`, `schedule_view`, `free_slots_view`, `check_overlap_result`,
    `needs_gap_confirmation`, `missing_field:title`, `insufficient_info`, `invalid_range`,
    `event_not_found`, `event_ambiguous`, `add_event_rule_success`, `habit_dismissed`,
    `no_action`, `parse_failed:timeout`… Reused both by the evals (`evals/`) and by the turn
    log (`turn_log`): the same taxonomy in both places.
    """
    match execution:
        case Executed(outcome=outcome):
            return _snake(type(outcome).__name__)
        case MissingField(field=field):
            return f"missing_field:{field}"
        case NoAction():
            return "no_action"
        case ParseFailed(failure=failure):
            return f"parse_failed:{failure.kind}"
        case ExecutionError():
            return "execution_error"
    return "no_execution"
