"""The eval cases, read from YAML files.

Compact form for a single-message case:

    - id: creation-01
      say: "add gym tomorrow at 6 pm for an hour"
      call: {tool: add_event, title: gym, date: "{tomorrow}", time: "18:00", duration_minutes: 60}
      added: [[gym, "{tomorrow}T18:00", "{tomorrow}T19:00"]]

With several messages, `turns:` is a list of `{say, call, kind, reply_has, quick_replies}`.
`call` is the expected tool call: the "oracle" parser uses it (validating the runner) and it
serves as a diagnostic for the other parsers; it does **not** decide whether a case is right.
The final state decides: `added`/`removed`/`moved` relative to the starting database (or
`events`, the complete set). With no expectations the schedule must stay as it was: one
creation too many is an error.

In the structured fields (never in `say`) `{today}`, `{tomorrow}`, `{day_after_tomorrow}`,
`{yesterday}`, `{mon}`…`{sun}` (the first Monday…Sunday after today) and `{+N}`/`{-N}` are
dates relative to `now`, so they are not computed by hand. An expected title may have
variants: `accountant|the accountant`.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from daily.agent.schemas import parse_tool_call

DEFAULT_NOW = "2026-09-21T15:30"  # a Monday
SPLITS = ("dev", "holdout")

Triple = tuple[str, str, str]

_ISO_MINUTE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
_MACRO = re.compile(r"\{([^{}]+)\}")
_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
_KEYS = {
    "id", "section", "split", "now", "seed", "say", "call", "kind", "reply_has", "quick_replies",
    "turns", "added", "removed", "moved", "events", "rules_added", "rules_removed", "defaults",
    "defaults_removed", "confirmed_patterns", "rejected_patterns", "temporal",
}  # fmt: skip
_TURN_KEYS = {"say", "call", "kind", "reply_has", "quick_replies"}


class CaseError(ValueError):
    """A badly written case: it stops right away, with the case id."""


@dataclass(frozen=True, slots=True)
class SeedEvent:
    title: str
    start: str
    end: str
    parent: str | None = None  # title of another starting event: this one derives from it


@dataclass(frozen=True, slots=True)
class SeedRule:
    trigger: str
    related: str
    position: str
    offset: int | None = None
    gap: int | None = None


@dataclass(frozen=True, slots=True)
class SeedDefault:
    title: str
    start: str | None = None
    duration: int | None = None


@dataclass(frozen=True, slots=True)
class Seed:
    events: tuple[SeedEvent, ...] = ()
    rules: tuple[SeedRule, ...] = ()
    defaults: tuple[SeedDefault, ...] = ()


@dataclass(frozen=True, slots=True)
class Turn:
    say: str
    call: dict[str, Any] | None = None
    kind: str | None = None
    reply_has: tuple[str, ...] = ()
    quick_replies: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class Expect:
    added: tuple[Triple, ...] = ()
    removed: tuple[Triple, ...] = ()
    moved: tuple[tuple[Triple, Triple], ...] = ()
    events: tuple[Triple, ...] | None = None
    rules_added: tuple[tuple[str, str], ...] = ()
    rules_removed: tuple[tuple[str, str], ...] = ()
    defaults: tuple[tuple[str, str | None, int | None], ...] = ()
    defaults_removed: tuple[str, ...] = ()
    confirmed_patterns: int | None = None
    rejected_patterns: int | None = None


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    section: str
    split: str
    now: datetime
    seed: Seed
    turns: tuple[Turn, ...]
    expect: Expect
    temporal: str | None = None


# --- symbolic dates --------------------------------------------------------------------


def _day(name: str, today: date) -> date:
    fixed = {"today": 0, "tomorrow": 1, "day_after_tomorrow": 2, "yesterday": -1}
    if name in fixed:
        return today + timedelta(days=fixed[name])
    if re.fullmatch(r"[+-]\d+", name):
        return today + timedelta(days=int(name))
    if name in _WEEKDAYS:
        return today + timedelta(days=(_WEEKDAYS[name] - today.weekday()) % 7 or 7)
    raise CaseError(f"unknown symbolic date: {{{name}}}")


def _expand(value: Any, today: date) -> Any:
    if isinstance(value, str):
        return _MACRO.sub(lambda m: _day(m.group(1), today).isoformat(), value)
    if isinstance(value, list):
        return [_expand(item, today) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, today) for key, item in value.items()}
    return value


# --- loading ---------------------------------------------------------------------------


def load_cases(path: Path) -> list[Case]:
    """All the cases of a file or of a folder (one `.yaml` file per section)."""
    files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    cases: list[Case] = []
    seen: set[str] = set()
    for file in files:
        raw = yaml.safe_load(file.read_text(encoding="utf-8")) or []
        if not isinstance(raw, list):
            raise CaseError(f"{file.name}: a list of cases was expected")
        for item in raw:
            case = _case(item, section=file.stem)
            if case.id in seen:
                raise CaseError(f"duplicate id: {case.id}")
            seen.add(case.id)
            cases.append(case)
    return cases


def _case(item: dict[str, Any], *, section: str) -> Case:
    case_id = str(item.get("id", "?"))
    try:
        return _build(item, section)
    except CaseError as error:
        raise CaseError(f"{case_id}: {error}") from None
    except (ValidationError, ValueError, TypeError, KeyError, IndexError) as error:
        raise CaseError(f"{case_id}: {error}") from error


def _build(item: dict[str, Any], section: str) -> Case:
    unknown = set(item) - _KEYS
    if unknown:
        raise CaseError(f"unknown keys: {sorted(unknown)}")
    if "id" not in item:
        raise CaseError("missing id")
    now = datetime.strptime(str(item.get("now", DEFAULT_NOW)), "%Y-%m-%dT%H:%M")
    today = now.date()
    split = item.get("split", "dev")
    if split not in SPLITS:
        raise CaseError(f"invalid split: {split!r}")

    if ("say" in item) == ("turns" in item):
        raise CaseError("either `say` (one message) or `turns` is needed, not both nor neither")
    turn_items = (
        item["turns"] if "turns" in item else [{k: item[k] for k in _TURN_KEYS & set(item)}]
    )
    turns = tuple(_turn(t) for t in turn_items)

    expanded = _expand({k: v for k, v in item.items() if k in _STRUCTURED}, today)
    return Case(
        id=str(item["id"]),
        section=str(item.get("section", section)),
        split=split,
        now=now,
        seed=_seed(expanded.get("seed") or {}),
        turns=tuple(_expand_turn(t, today) for t in turns),
        expect=_expect(expanded),
        temporal=item.get("temporal"),
    )


_STRUCTURED = {
    "seed", "added", "removed", "moved", "events", "rules_added", "rules_removed", "defaults",
    "defaults_removed", "confirmed_patterns", "rejected_patterns",
}  # fmt: skip


def _turn(item: dict[str, Any]) -> Turn:
    unknown = set(item) - _TURN_KEYS
    if unknown or "say" not in item:
        raise CaseError(f"invalid turn (keys: {sorted(item)})")
    call = item.get("call")
    quick = item.get("quick_replies")
    return Turn(
        say=str(item["say"]),
        call=call,
        kind=item.get("kind"),
        reply_has=tuple(item.get("reply_has", ())),
        quick_replies=tuple(quick) if quick is not None else None,
    )


def _expand_turn(turn: Turn, today: date) -> Turn:
    if turn.call is None:
        return turn
    call = _expand(turn.call, today)
    parse_tool_call(call)  # the expected tool call must be valid
    return Turn(turn.say, call, turn.kind, turn.reply_has, turn.quick_replies)


def _triple(value: Any) -> Triple:
    title, start, end = value
    for moment in (start, end):
        if not _ISO_MINUTE.match(str(moment)):
            raise CaseError(f"invalid time: {moment!r}")
    return (str(title), str(start), str(end))


def _seed(raw: dict[str, Any]) -> Seed:
    events = []
    for entry in raw.get("events", ()):
        if isinstance(entry, dict):
            title, start, end = _triple([entry["title"], entry["start"], entry["end"]])
            events.append(SeedEvent(title, start, end, entry.get("parent")))
        else:
            events.append(SeedEvent(*_triple(entry)))
    rules = tuple(SeedRule(*(list(r) + [None] * (5 - len(r)))) for r in raw.get("rules", ()))
    defaults = tuple(
        SeedDefault(*(list(d) + [None] * (3 - len(d)))) for d in raw.get("defaults", ())
    )
    return Seed(tuple(events), rules, defaults)


def _expect(raw: dict[str, Any]) -> Expect:
    events = raw.get("events")
    return Expect(
        added=tuple(_triple(t) for t in raw.get("added", ())),
        removed=tuple(_triple(t) for t in raw.get("removed", ())),
        moved=tuple((_triple(old), _triple(new)) for old, new in raw.get("moved", ())),
        events=tuple(_triple(t) for t in events) if events is not None else None,
        rules_added=tuple((r[0], r[1]) for r in raw.get("rules_added", ())),
        rules_removed=tuple((r[0], r[1]) for r in raw.get("rules_removed", ())),
        defaults=tuple(
            (title, *(list(values) + [None] * (2 - len(values))))
            for title, values in (raw.get("defaults") or {}).items()
        ),
        defaults_removed=tuple(raw.get("defaults_removed", ())),
        confirmed_patterns=raw.get("confirmed_patterns"),
        rejected_patterns=raw.get("rejected_patterns"),
    )
