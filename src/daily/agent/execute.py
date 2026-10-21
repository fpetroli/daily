"""The adapters: a tool call of the model → an operation of the services.

They are the only place where language meets the domain. They do three things: check that
no value without a fallback is missing (missing field → question), call the services with
their parameter names and return a typed outcome. They never reinterpret a value: the model
has already normalized dates, times and minutes, and the schema has validated them.
"""

from datetime import date, datetime, timedelta

from daily.agent.results import (
    Executed,
    FreeSlotsView,
    HabitDismissed,
    MissingField,
    NoAction,
    ScheduleView,
)
from daily.agent.schemas import (
    AddEventCall,
    AddEventRelativeCall,
    AddEventRuleCall,
    CheckOverlapCall,
    DismissHabitCall,
    GetFreeSlotsCall,
    GetScheduleCall,
    NoActionCall,
    RemoveEventCall,
    RemoveEventRuleCall,
    ToolCall,
    UpdateEventCall,
)
from daily.domain.timeutil import combine_date_and_time, today_date_string
from daily.services.context import Repositories
from daily.services.habits import (
    HabitProposal,
    dismiss_habit,
    mark_proposed,
    next_habit_proposal,
)
from daily.services.relative import RelativeNeedsInfo, create_event_relative
from daily.services.rules import (
    GAP_TITLE,
    AddEventRuleSuccess,
    add_event_rule,
    remove_event_rule,
)
from daily.services.schedule import (
    EventCreated,
    EventRemoved,
    EventUpdated,
    check_overlap,
    create_event,
    event_day,
    get_free_slots,
    get_schedule,
    remove_event,
    update_event,
)

Executable = Executed | MissingField | NoAction


def execute(
    repos: Repositories,
    call: ToolCall,
    *,
    now: datetime,
    shown_habit: HabitProposal | None = None,
) -> Executable:
    """Executes `call`. It may raise on an unexpected error: the workflow step catches it.

    `shown_habit` is the habit proposal of the previous reply, the one a "no" rejects. A new
    proposal in the result is marked as shown: its counter starts again."""
    today = today_date_string(now)
    result = _dispatch(repos, call, today, now, shown_habit)
    if isinstance(result, Executed):
        _log_write(repos, result.outcome, now)
        if result.habit is not None:
            mark_proposed(repos, result.habit)
    return result


def _log_write(repos: Repositories, outcome: object, now: datetime) -> None:
    """Records the last real write on the schedule, for undo.

    Only the three actions that write to `schedule`: rules and usual times stay out of undo
    for now (v1, kept simple on purpose). One log entry per write, so undo always acts on the
    last one, never on a downstream effect (accepted proposals/cascades are writes of their
    own, with their own row).
    """
    match outcome:
        case EventCreated(event_id=event_id, title=title, start=start, end=end):
            repos.action_log.record(
                kind="add_event",
                summary=f"added '{title}'",
                undo_data={"event_id": event_id, "title": title, "start": start, "end": end},
                now=now,
            )
        case EventUpdated(
            event_id=event_id,
            original_title=original_title,
            original_start=original_start,
            original_end=original_end,
        ):
            repos.action_log.record(
                kind="update_event",
                summary=f"changed '{original_title}'",
                undo_data={
                    "event_id": event_id,
                    "title": original_title,
                    "start": original_start,
                    "end": original_end,
                },
                now=now,
            )
        case EventRemoved(event_id=event_id, removed_title=title, start=start, end=end):
            repos.action_log.record(
                kind="remove_event",
                summary=f"deleted '{title}'",
                undo_data={"event_id": event_id, "title": title, "start": start, "end": end},
                now=now,
            )


def _dispatch(
    repos: Repositories,
    call: ToolCall,
    today: str,
    now: datetime,
    shown_habit: HabitProposal | None,
) -> Executable:
    match call:
        case AddEventCall():
            return _add_event(repos, call, today, now)
        case AddEventRelativeCall():
            return _add_event_relative(repos, call, today, now)
        case UpdateEventCall():
            return _update_event(repos, call, today, now)
        case RemoveEventCall():
            return _remove_event(repos, call, today)
        case GetScheduleCall():
            day = call.date or today
            entries = get_schedule(repos, default_date=day, time_band=call.time_band, now=now)
            following = None
            if entries and call.time_band is None:  # a whole day: how the next one starts
                next_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
                following = next(iter(get_schedule(repos, default_date=next_day, now=now)), None)
            return Executed(call, ScheduleView(day, call.time_band, tuple(entries), following))
        case GetFreeSlotsCall():
            day = call.date or today
            slots = get_free_slots(
                repos,
                default_date=day,
                min_duration_minutes=call.min_duration_minutes,
                time_band=call.time_band,
                now=now,
            )
            return Executed(
                call, FreeSlotsView(day, call.time_band, call.min_duration_minutes, tuple(slots))
            )
        case CheckOverlapCall():
            if call.time is None or call.end_time is None:
                return MissingField(call, "time" if call.time is None else "end_time")
            day = call.date or today
            start = combine_date_and_time(day, call.time)
            end = combine_date_and_time(day, call.end_time)
            return Executed(call, check_overlap(repos, start=start, end=end))
        case AddEventRuleCall():
            return _add_event_rule(repos, call)
        case RemoveEventRuleCall():
            # A single title is the event the rule is about, whichever field it is in.
            titles = [t for t in (call.trigger_title, call.related_title) if t is not None]
            if not titles:
                return MissingField(call, "trigger_title")
            outcome = remove_event_rule(
                repos,
                trigger_title=titles[0],
                related_title=titles[1] if len(titles) > 1 else None,
                field=call.field,
            )
            return Executed(call, outcome)
        case DismissHabitCall():
            dismissed = dismiss_habit(repos, shown_habit)
            return Executed(call, HabitDismissed(dismissed), habit=next_habit_proposal(repos))
        case NoActionCall():
            return NoAction()
    raise AssertionError(f"Unhandled tool call: {call!r}")


def _add_event(repos: Repositories, call: AddEventCall, today: str, now: datetime) -> Executable:
    if call.title is None:
        return MissingField(call, "title")
    # `date` and `time` are two independent readings of the model: here they are combined into
    # the single ISO datetime the services expect, never the other way round.
    day = call.date or today
    start = combine_date_and_time(day, call.time) if call.time else None
    end = combine_date_and_time(day, call.end_time) if call.end_time else None
    outcome = create_event(
        repos,
        title=call.title,
        date=day,
        start=start,
        end=end,
        duration_minutes=call.duration_minutes,
        with_gap=call.with_gap,
        anchor_time=call.anchor_time,
        now=now,
    )
    return Executed(call, outcome, habit=_habit_after_creation(repos, outcome))


def _add_event_relative(
    repos: Repositories, call: AddEventRelativeCall, today: str, now: datetime
) -> Executable:
    if call.title is None:
        return MissingField(call, "title")
    if call.anchor_description is None and call.position is None and call.offset_minutes is None:
        # Nothing relative was said ("add the shower"): it is a plain addition, placed with
        # the rules like any event without a time.
        plain = AddEventCall(
            tool="add_event",
            title=call.title,
            date=call.date,
            duration_minutes=call.duration_minutes,
        )
        return _add_event(repos, plain, today, now)
    # Each missing value has its own question: the service treats them all as "more data needed".
    for field in ("anchor_description", "position"):
        if getattr(call, field) is None:
            return MissingField(call, field, subject=call.title)
    outcome = create_event_relative(
        repos,
        title=call.title,
        date=call.date or today,
        anchor_description=call.anchor_description,
        anchor_time=call.anchor_time,
        position=call.position,
        offset_minutes=call.offset_minutes,
        duration_minutes=call.duration_minutes,
        now=now,
    )
    if isinstance(outcome, RelativeNeedsInfo):  # only the duration is left: no usual one either
        return MissingField(call, "duration_minutes", subject=call.title)
    return Executed(call, outcome, habit=_habit_after_creation(repos, outcome))


def _update_event(
    repos: Repositories, call: UpdateEventCall, today: str, now: datetime
) -> Executable:
    if call.event_description is None and call.event_time is None:
        return MissingField(call, "event_description")
    # `new_date` alone (without `new_time`) is not enough to build a new start, so it does not
    # count as a change: the same behaviour as before, when it could not be expressed.
    changes = (
        call.new_time,
        call.new_end_time,
        call.new_duration_minutes,
        call.shift_minutes,
        call.extend_minutes,
        call.new_title,
    )
    if all(change is None for change in changes):
        return MissingField(call, "change", subject=call.event_description)
    event_date = call.event_date or event_day(
        repos, description=call.event_description or "", at_time=call.event_time, today=today
    )
    # The day of the new time: the one said explicitly, otherwise the one the event is found
    # on, never composed by the model into a single field.
    day = call.new_date or event_date
    new_start = combine_date_and_time(day, call.new_time) if call.new_time else None
    new_end = combine_date_and_time(day, call.new_end_time) if call.new_end_time else None
    outcome = update_event(
        repos,
        event_description=call.event_description or "",
        default_date=today,
        event_date=event_date,
        event_time=call.event_time,
        new_start=new_start,
        new_end=new_end,
        new_duration_minutes=call.new_duration_minutes,
        shift_minutes=call.shift_minutes,
        extend_minutes=call.extend_minutes,
        new_title=call.new_title,
    )
    return Executed(call, outcome)


def _remove_event(repos: Repositories, call: RemoveEventCall, today: str) -> Executable:
    if call.event_description is None and call.event_time is None:
        return MissingField(call, "event_description")
    event_date = call.event_date or event_day(
        repos, description=call.event_description or "", at_time=call.event_time, today=today
    )
    outcome = remove_event(
        repos,
        event_description=call.event_description or "",
        default_date=today,
        event_date=event_date,
        event_time=call.event_time,
    )
    return Executed(call, outcome)


def _add_event_rule(repos: Repositories, call: AddEventRuleCall) -> Executable:
    trigger_title = call.trigger_title
    if trigger_title is None and call.gap_minutes and call.related_title:
        trigger_title = GAP_TITLE  # free time around an event needs no name
    if trigger_title is None:
        return MissingField(call, "trigger_title")
    outcome = add_event_rule(
        repos,
        trigger_title=trigger_title,
        related_title=call.related_title,
        position=call.position,
        offset_minutes=call.offset_minutes,
        gap_minutes=call.gap_minutes,
        typical_start=call.typical_start,
        typical_duration_minutes=call.typical_duration_minutes,
    )
    habit = next_habit_proposal(repos) if isinstance(outcome, AddEventRuleSuccess) else None
    return Executed(call, outcome, habit=habit)


def _habit_after_creation(repos: Repositories, outcome: object) -> HabitProposal | None:
    """A habit proposal is appended only after a creation with no other questions.

    Downstream proposals and conflicts take precedence: one question per reply.
    """
    if isinstance(outcome, EventCreated) and not outcome.rule_proposals:
        return next_habit_proposal(repos)
    return None
