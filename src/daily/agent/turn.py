"""The turn: a linear three-step Agno workflow (parse → execute → respond).

Step 1 asks the `Parser` for a tool call, step 2 executes it on the services, step 3 tells
about it with a template. **No exception crosses the steps**: every step catches everything
and returns a structured outcome, and `max_retries=0` keeps Agno from repeating a write on
the schedule. Even if Agno itself fails, `TurnService.handle` answers with a fixed message.

The history is the Agno workflow's own, in memory: (message, reply) pairs. The parser always
receives the last `_MAX_HISTORY_RUNS` exchanges, not only after a pending question.

Some buttons skip the model (`choose`): their choice is executed as it is and told with the
same templates. It goes in the turn log, not in the history.

Database access is single-threaded: the steps are `async` and use the services
synchronously, and a lock serializes the turns (one user, one model).
"""

import asyncio
import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime

from agno.db.in_memory import InMemoryDb
from agno.run import RunContext
from agno.workflow import Step, StepInput, StepOutput, Workflow

from daily.agent.execute import execute
from daily.agent.parser import Exchange, ParseFailure, Parser, ParseRequest
from daily.agent.results import (
    Choice,
    Executed,
    Execution,
    ExecutionError,
    MissingField,
    NoAction,
    ParseFailed,
    kind_of,
)
from daily.agent.schemas import NoActionCall, RemoveEventRuleCall, ToolCall
from daily.agent.templates import GENERIC, Reply, render
from daily.domain.models import EventRule
from daily.services.context import Repositories
from daily.services.habits import HabitProposal
from daily.services.rules import UsualTimes, remove_rule_target

logger = logging.getLogger("daily.agent")

SESSION_ID = "main"
LAST_REPLY = "last_reply"
_TIME_FORMAT = "%Y-%m-%dT%H:%M"
# Two uses: how many runs stay in the process memory (periodic pruning) AND how many
# exchanges at most the model receives on each turn. The same number on purpose: the process
# memory makes no sense shorter than the window the model can see.
_MAX_HISTORY_RUNS = 3


@dataclass(frozen=True, slots=True)
class TurnResult:
    """The outcome of a turn: what is shown and what happened."""

    reply: Reply
    execution: Execution | None = None
    turn_id: str | None = None


class TurnService:
    """One conversation turn at a time. Keeps the workflow and the history in memory."""

    def __init__(self, repos: Repositories, parser: Parser, *, session_id: str = SESSION_ID):
        self._repos = repos
        self._parser = parser
        self._session_id = session_id
        self._lock = asyncio.Lock()
        # The habit proposal of the last reply: the one a "no" rejects.
        self._shown_habit: HabitProposal | None = None
        # The whole reply of the running turn, with its styles and choices: the workflow
        # keeps only its text.
        self._reply: Reply | None = None
        self._workflow = Workflow(
            name="turn",
            db=InMemoryDb(),
            telemetry=False,
            steps=[
                Step(name="parse", executor=self._parse, max_retries=0),
                Step(name="execute", executor=self._execute, max_retries=0),
                Step(name="respond", executor=self._respond, max_retries=0),
            ],
        )

    async def handle(
        self, message: str, *, now: datetime, timezone: str | None = None
    ) -> TurnResult:
        """Processes a message. It never raises: in the worst case it answers with a fixed
        message."""
        started = time.monotonic()
        async with self._lock:
            try:
                run = await self._workflow.arun(
                    input=message,
                    session_id=self._session_id,
                    additional_data={"now": now.strftime(_TIME_FORMAT), "timezone": timezone},
                )
                result = TurnResult(self._reply_of(run), self._execution_of(run))
            except Exception:
                logger.exception("The turn workflow raised")
                result = TurnResult(render(ExecutionError(), now))
            duration_ms = round((time.monotonic() - started) * 1000)
            turn_id = self._log_turn(message, result, now=now, duration_ms=duration_ms)
            self._prune_history()
            return replace(result, turn_id=turn_id)

    async def choose(self, choice: Choice, label: str, *, now: datetime) -> TurnResult:
        """Runs a button's choice without the model and tells about it like a turn. The turn
        log keeps it, with the button's label as the message. It never raises."""
        started = time.monotonic()
        async with self._lock:
            try:
                execution = self._run_choice(choice, now)
                result = TurnResult(render(execution, now), execution)
            except Exception:
                logger.exception("The choice raised")
                result = TurnResult(render(ExecutionError(), now))
            duration_ms = round((time.monotonic() - started) * 1000)
            turn_id = self._log_turn(label, result, now=now, duration_ms=duration_ms)
            return replace(result, turn_id=turn_id)

    async def forget(self) -> None:
        """Starts the conversation over: the history and the pending question are dropped.
        The schedule, the rules and the turn log stay."""
        async with self._lock:
            self._workflow.delete_session(self._session_id)
            self._shown_habit = None

    def _run_choice(self, choice: Choice, now: datetime) -> Execution:
        if isinstance(choice, EventRule | UsualTimes):
            outcome = remove_rule_target(self._repos, choice)
            return Executed(_removal_call(choice), outcome)
        return self._executed(choice, now)

    def _executed(self, call: ToolCall, now: datetime) -> Execution:
        """Executes a tool call and remembers the habit proposal it brings, if any."""
        execution = execute(self._repos, call, now=now, shown_habit=self._shown_habit)
        self._shown_habit = execution.habit if isinstance(execution, Executed) else None
        return execution

    # --- turn log and history cap -------------------------------------------------

    def _log_turn(
        self, message: str, result: TurnResult, *, now: datetime, duration_ms: int
    ) -> str | None:
        """Records the whole turn, whatever the outcome: it must never make the turn fail.

        `duration_ms` includes the wait for the lock: it is the time the caller really
        perceives, not only the model's work."""
        try:
            call = _call_of(result.execution)
            return self._repos.turn_log.record(
                message=message,
                tool=call.tool if call is not None else None,
                call_json=call.model_dump(exclude_none=True) if call is not None else None,
                kind=kind_of(result.execution),
                reply_text=result.reply.text,
                duration_ms=duration_ms,
                now=now,
            )
        except Exception:
            logger.exception("The turn log raised")
            return None

    def _prune_history(self) -> None:
        """Keeps only the last `_MAX_HISTORY_RUNS` runs in memory: never makes the turn fail."""
        try:
            runs = self._workflow.db.get_runs(
                session_id=self._session_id, sort_by="created_at", sort_order="asc"
            )
            if len(runs) > _MAX_HISTORY_RUNS:
                stale = runs[: len(runs) - _MAX_HISTORY_RUNS]
                self._workflow.db.delete_runs([run.run_id for run in stale])
        except Exception:
            logger.exception("The history pruning raised")

    # --- reading the outcome -------------------------------------------------------

    def _reply_of(self, run) -> Reply:
        reply, self._reply = self._reply, None
        if reply is not None and reply.text == run.content:
            return reply
        stored = self._session_state().get(LAST_REPLY) or {}
        text = run.content if isinstance(run.content, str) else GENERIC
        return Reply(
            text=text,
            quick_replies=tuple(stored.get("quick_replies", ())),
            expects_reply=bool(stored.get("expects_reply", False)),
        )

    @staticmethod
    def _execution_of(run) -> Execution | None:
        for result in run.step_results or []:
            if result.step_name == "execute" and result.content is not None:
                return result.content
        return None

    def _session_state(self) -> dict:
        session = self._workflow.get_session(session_id=self._session_id)
        data = getattr(session, "session_data", None) or {}
        return data.get("session_state") or {}

    # --- the three steps -----------------------------------------------------------

    async def _parse(self, step_input: StepInput, run_context: RunContext) -> StepOutput:
        try:
            extra = step_input.additional_data or {}
            request = ParseRequest(
                message=str(step_input.input),
                now=datetime.strptime(extra["now"], _TIME_FORMAT),
                timezone=extra.get("timezone"),
                history=_recent_exchange(step_input),
            )
            result = await self._parser.parse(request)
        except Exception as error:  # the parser must not stop the turn
            logger.exception("The parse step raised")
            result = ParseFailure("error", type(error).__name__)
        return StepOutput(content=result)

    async def _execute(self, step_input: StepInput, run_context: RunContext) -> StepOutput:
        parsed = step_input.previous_step_content
        try:
            now = datetime.strptime((step_input.additional_data or {})["now"], _TIME_FORMAT)
            if isinstance(parsed, ParseFailure):
                execution: Execution = ParseFailed(parsed)
            elif isinstance(parsed, NoActionCall):
                last = (run_context.session_state or {}).get(LAST_REPLY) or {}
                execution = NoAction(after_question=bool(last.get("expects_reply")))
            elif parsed is None or isinstance(parsed, str | dict):
                execution = ExecutionError()  # the previous step produced nothing
            else:
                execution = self._executed(_as_call(parsed), now)
        except Exception:
            logger.exception("The execute step raised")
            execution = ExecutionError()
        if not isinstance(execution, Executed):
            self._shown_habit = None  # a proposal the user let pass
        return StepOutput(content=execution)

    async def _respond(self, step_input: StepInput, run_context: RunContext) -> StepOutput:
        try:
            now = datetime.strptime((step_input.additional_data or {})["now"], _TIME_FORMAT)
            reply = render(step_input.previous_step_content, now)
        except Exception:
            logger.exception("The respond step raised")
            reply = Reply(GENERIC)
        self._reply = reply
        if run_context.session_state is not None:
            run_context.session_state[LAST_REPLY] = {
                "expects_reply": reply.expects_reply,
                "quick_replies": list(reply.quick_replies),
            }
        return StepOutput(content=reply.text)


def _as_call(parsed: object) -> ToolCall:
    return parsed  # type: ignore[return-value]


def _call_of(execution: Execution | None) -> ToolCall | None:
    """The tool call behind the outcome, if there was a valid one (for the turn log)."""
    match execution:
        case Executed(call=call) | MissingField(call=call):
            return call
        case _:
            return None


def _recent_exchange(step_input: StepInput) -> tuple[Exchange, ...]:
    """The last `_MAX_HISTORY_RUNS` exchanges, always, not only after a pending question.
    `get_workflow_history` already returns them in chronological order (oldest first), the
    same order in which they are written in the prompt."""
    pairs = step_input.get_workflow_history(num_runs=_MAX_HISTORY_RUNS)
    return tuple(Exchange(user=str(user), reply=str(reply)) for user, reply in pairs)


def _removal_call(target: EventRule | UsualTimes) -> RemoveEventRuleCall:
    """The tool call a removal picked by button stands for: for the turn log."""
    if isinstance(target, UsualTimes):
        return RemoveEventRuleCall(
            tool="remove_event_rule",
            trigger_title=target.title,
            field=target.field,  # type: ignore[arg-type]
        )
    return RemoveEventRuleCall(
        tool="remove_event_rule",
        trigger_title=target.trigger_title,
        related_title=target.related_title,
    )
