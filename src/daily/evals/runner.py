"""Runs the cases with a parser and compares the final state of the schedule.

Every case has its own in-memory database and a real turn (workflow, adapters, templates,
history): the parser is the only thing that changes from one run to the next. A case is right
if the final schedule, rules and usual times are the expected ones **and** the given reply
kinds and phrases match. The produced tool call is compared with the expected one only as a
diagnostic: two different but equivalent tool calls give the same state and are both right.
"""

import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Any

from daily.agent.parser import ParseFailure, Parser, ParseRequest
from daily.agent.results import kind_of
from daily.agent.schemas import ToolCall, parse_tool_call
from daily.agent.turn import TurnService
from daily.evals.cases import Case, CaseError, Seed, Triple
from daily.evals.parsers import ParserFactory
from daily.services.context import Repositories
from daily.storage.db import open_database

logger = logging.getLogger("daily.evals")


@dataclass(frozen=True, slots=True)
class TurnRecord:
    message: str
    kind: str
    reply: str
    quick_replies: tuple[str, ...]
    latency_ms: float
    expected_call: dict[str, Any] | None
    actual_call: dict[str, Any] | None
    parse_detail: str = ""  # if the parser failed: what it produced or the error

    @property
    def tool_match(self) -> bool | None:
        if self.expected_call is None:
            return None
        return (self.actual_call or {}).get("tool") == self.expected_call["tool"]

    @property
    def exact_match(self) -> bool | None:
        if self.expected_call is None:
            return None
        return self.actual_call == self.expected_call


@dataclass(frozen=True, slots=True)
class CaseResult:
    case: Case
    failures: tuple[str, ...]
    turns: tuple[TurnRecord, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def wrote_unexpected(self) -> bool:
        """Left something extra in the schedule or among the rules (a wrong move included):
        it is the costly error, because the user did not ask for it."""
        return any(f.startswith(("schedule: extra", "rules: extra")) for f in self.failures)


class _RecordingParser:
    """Wraps a parser and remembers what it produced on every turn."""

    def __init__(self, inner: Parser) -> None:
        self._inner = inner
        self.results: list[ToolCall | ParseFailure] = []

    async def parse(self, request: ParseRequest) -> ToolCall | ParseFailure:
        result = await self._inner.parse(request)
        self.results.append(result)
        return result


async def run_case(case: Case, factory: ParserFactory) -> CaseResult:
    """Runs a case. It does not raise: an error becomes a failed case with its reason."""
    conn = open_database()
    try:
        repos = Repositories.from_connection(conn)
        _seed(repos, case.seed)
        parser = _RecordingParser(factory(case))
        service = TurnService(repos, parser)
        records: list[TurnRecord] = []
        failures: list[str] = []
        for index, turn in enumerate(case.turns, start=1):
            started = time.perf_counter()
            result = await service.handle(turn.say, now=case.now)
            elapsed = (time.perf_counter() - started) * 1000
            kind = kind_of(result.execution)
            produced = parser.results[index - 1] if len(parser.results) >= index else None
            records.append(
                TurnRecord(
                    message=turn.say,
                    kind=kind,
                    reply=result.reply.text,
                    quick_replies=result.reply.quick_replies,
                    latency_ms=elapsed,
                    expected_call=_dump(parse_tool_call(turn.call)) if turn.call else None,
                    actual_call=_dump(produced) if produced and not _failed(produced) else None,
                    parse_detail=produced.detail if isinstance(produced, ParseFailure) else "",
                )
            )
            failures += _check_turn(index, turn, kind, result)
        failures += _check_state(case, conn)
        return CaseResult(case, tuple(failures), tuple(records))
    except CaseError:
        raise
    except Exception as error:
        logger.exception("Case %s failed with an exception", case.id)
        return CaseResult(case, (f"exception: {type(error).__name__}: {error}",), ())
    finally:
        conn.close()


async def run_cases(
    cases: list[Case],
    factory: ParserFactory,
    progress: Callable[[int, int, CaseResult], None] | None = None,
) -> list[CaseResult]:
    """Runs the cases in order. `progress(index, total, result)` is called after each one."""
    results = []
    for index, case in enumerate(cases, start=1):
        results.append(await run_case(case, factory))
        if progress:
            progress(index, len(cases), results[-1])
    return results


def _failed(produced: object) -> bool:
    return isinstance(produced, ParseFailure)


def _dump(call: ToolCall) -> dict[str, Any]:
    return call.model_dump(exclude_none=True, mode="json")


# --- starting state --------------------------------------------------------------------


def _seed(repos: Repositories, seed: Seed) -> None:
    ids: dict[str, str] = {}
    for event in seed.events:
        parent_id = ids.get(event.parent) if event.parent else None
        if event.parent and parent_id is None:
            raise CaseError(f"the parent '{event.parent}' must come earlier in the seed")
        ids[event.title] = repos.schedule.insert_event(
            title=event.title,
            start=event.start,
            end=event.end,
            derived_from_event_id=parent_id,
        )
    for rule in seed.rules:
        repos.event_rules.insert_rule(
            trigger_title=rule.trigger,
            related_title=rule.related,
            position=rule.position,
            offset_minutes=rule.offset,
            gap_minutes=rule.gap,
        )
    for default in seed.defaults:
        repos.event_defaults.upsert(
            default.title, typical_start=default.start, typical_duration_minutes=default.duration
        )


# --- checks ----------------------------------------------------------------------------


def _check_turn(index: int, turn, kind: str, result) -> list[str]:
    problems = []
    if turn.kind is not None and turn.kind != kind:
        problems.append(f"turn {index}: reply kind {kind!r} instead of {turn.kind!r}")
    for phrase in turn.reply_has:
        if phrase not in result.reply.text:
            problems.append(
                f"turn {index}: the reply does not contain {phrase!r}: {result.reply.text!r}"
            )
    if turn.quick_replies is not None and turn.quick_replies != result.reply.quick_replies:
        problems.append(
            f"turn {index}: quick replies {result.reply.quick_replies} instead of "
            f"{turn.quick_replies}"
        )
    return problems


def _title_matches(expected: str, actual: str) -> bool:
    options = {alternative.strip().casefold() for alternative in expected.split("|")}
    return actual.strip().casefold() in options


def _diff_events(expected: list[Triple], actual: list[Triple]) -> tuple[list[Triple], list[Triple]]:
    """(missing, extra): comparison between sets, ignoring order and title variants."""
    remaining = list(actual)
    missing = []
    for title, start, end in expected:
        found = next(
            (
                i
                for i, (a_title, a_start, a_end) in enumerate(remaining)
                if (a_start, a_end) == (start, end) and _title_matches(title, a_title)
            ),
            None,
        )
        if found is None:
            missing.append((title, start, end))
        else:
            remaining.pop(found)
    return missing, remaining


def _same_rule(expected: tuple[str, str], actual: tuple[str, str]) -> bool:
    return _title_matches(expected[0], actual[0]) and _title_matches(expected[1], actual[1])


def _diff_rules(
    expected: list[tuple[str, str]], actual: list[tuple[str, str]]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(missing, extra) between rules, with the title variants of the expected ones."""
    remaining = list(actual)
    missing = []
    for rule in expected:
        found = next((i for i, act in enumerate(remaining) if _same_rule(rule, act)), None)
        if found is None:
            missing.append(rule)
        else:
            remaining.pop(found)
    return missing, remaining


def _expected_events(case: Case) -> list[Triple]:
    expect = case.expect
    if expect.events is not None:
        return list(expect.events)
    current = [(e.title, e.start, e.end) for e in case.seed.events]
    try:
        for old, new in expect.moved:
            current.remove(old)
            current.append(new)
        for gone in expect.removed:
            current.remove(gone)
    except ValueError as error:
        raise CaseError(f"{case.id}: an event to remove or move is not in the seed") from error
    return current + list(expect.added)


def _check_state(case: Case, conn: Connection) -> list[str]:
    problems = []
    actual = [(r["title"], r["start"], r["end"]) for r in conn.execute("SELECT * FROM schedule")]
    missing, extra = _diff_events(_expected_events(case), actual)
    if missing:
        problems.append(f"schedule: missing {missing}")
    if extra:
        problems.append(f"schedule: extra {extra}")

    actual_rules = [
        (r["trigger_title"], r["related_title"])
        for r in conn.execute("SELECT trigger_title, related_title FROM event_rules")
    ]
    expected_rules = [(r.trigger, r.related) for r in case.seed.rules]
    for gone in case.expect.rules_removed:
        expected_rules = [r for r in expected_rules if not _same_rule(gone, r)]
    expected_rules += list(case.expect.rules_added)
    missing_rules, extra_rules = _diff_rules(expected_rules, actual_rules)
    if missing_rules:
        problems.append(f"rules: missing {missing_rules}")
    if extra_rules:
        problems.append(f"rules: extra {extra_rules}")

    defaults = {
        r["title"].casefold(): (r["typical_start"], r["typical_duration_minutes"])
        for r in conn.execute("SELECT * FROM event_defaults")
    }
    expected_defaults = {d.title.casefold(): (d.start, d.duration) for d in case.seed.defaults}
    for title, start, duration in case.expect.defaults:
        expected_defaults[title.casefold()] = (start, duration)
    for title in case.expect.defaults_removed:
        expected_defaults.pop(title.casefold(), None)
    if defaults != expected_defaults:
        problems.append(f"usual times: {defaults} instead of {expected_defaults}")

    statuses = _pattern_statuses(conn)
    for name, wanted in (
        ("confirmed", case.expect.confirmed_patterns),
        ("rejected", case.expect.rejected_patterns),
    ):
        if wanted is not None and statuses[name] != wanted:
            problems.append(f"habits {name}: {statuses[name]} instead of {wanted}")
    return problems


def _pattern_statuses(conn: Connection) -> Counter[str]:
    counts: Counter[str] = Counter()
    for table in (
        "event_patterns",
        "sequence_patterns",
        "gap_patterns",
        "trigger_duration_patterns",
    ):
        for row in conn.execute(f"SELECT status FROM {table}"):
            counts[row["status"]] += 1
    return counts
