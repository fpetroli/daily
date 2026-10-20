"""The tool calls the model can produce: the schema the server imposes on the output.

The model does **not** write text for the user and sees no internal identifiers. Every field
is optional and must be **omitted when unknown**: a value is never invented and `null` is
never written (fewer tokens, less latency). The model normalizes dates, times, durations and
shifts; here only the shape is checked.

The field names are the ones tested in the early experiments: fields that find an event use
`event_*`, new values use `new_*`, so the two meanings never get mixed up. Tool calls are
stateless: everything needed to execute them is in the call, and the values the server does
not remember (time of an anchor not in the schedule, duration, "with or without a break",
time to choose among events with the same name) have a field of their own.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from daily.domain.timeband import TimeBand
from daily.valuetypes import (
    DurationMinutes,
    IsoDate,
    ShiftMinutes,
    TimeOfDay,
    Title,
)

Position = Literal["before", "after"]
RuleField = Literal["typical_start", "typical_duration_minutes"]


class _Call(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AddEventCall(_Call):
    """Adds an event: at a precise time, or from rules and usual times.

    `date` and `time` are two independent readings (a day from the calendar, a time from the
    message): the model no longer has to combine them into a single ISO value, where it used
    to lose the time by writing only `date`. An omitted `date` is today; an omitted `time`
    is "no precise time", which places the event using the rules."""

    tool: Literal["add_event"]
    title: Title | None = None
    date: IsoDate | None = None
    time: TimeOfDay | None = None
    end_time: TimeOfDay | None = None
    duration_minutes: DurationMinutes | None = None
    anchor_time: TimeOfDay | None = None
    with_gap: bool | None = None


class AddEventRelativeCall(_Call):
    """Adds an event right before or after another one already in the schedule."""

    tool: Literal["add_event_relative"]
    title: Title | None = None
    anchor_description: str | None = None
    anchor_time: TimeOfDay | None = None
    date: IsoDate | None = None
    position: Position | None = None
    offset_minutes: Annotated[int, Field(ge=0, le=24 * 60)] | None = None
    duration_minutes: DurationMinutes | None = None


class UpdateEventCall(_Call):
    """Moves or changes an event already in the schedule.

    `new_date` and `new_time`/`new_end_time` are the same split as in `AddEventCall`: a day
    from the calendar and a time from the message, never a single ISO value composed by the
    model. An omitted `new_date` is the day the event is searched on (`event_date`, or
    today)."""

    tool: Literal["update_event"]
    event_description: str | None = None
    event_date: IsoDate | None = None
    event_time: TimeOfDay | None = None
    new_date: IsoDate | None = None
    new_time: TimeOfDay | None = None
    new_end_time: TimeOfDay | None = None
    new_duration_minutes: DurationMinutes | None = None
    shift_minutes: ShiftMinutes | None = None
    extend_minutes: ShiftMinutes | None = None
    new_title: Title | None = None


class RemoveEventCall(_Call):
    """Deletes an event already in the schedule."""

    tool: Literal["remove_event"]
    event_description: str | None = None
    event_date: IsoDate | None = None
    event_time: TimeOfDay | None = None


class GetScheduleCall(_Call):
    """Shows the schedule of one day."""

    tool: Literal["get_schedule"]
    date: IsoDate | None = None
    time_band: TimeBand | None = None


class GetFreeSlotsCall(_Call):
    """Looks for the free slots of one day."""

    tool: Literal["get_free_slots"]
    date: IsoDate | None = None
    min_duration_minutes: DurationMinutes | None = None
    time_band: TimeBand | None = None


class CheckOverlapCall(_Call):
    """Checks whether a time range is already taken.

    Same date/time split as the other tool calls: an omitted `date` is today."""

    tool: Literal["check_overlap"]
    date: IsoDate | None = None
    time: TimeOfDay | None = None
    end_time: TimeOfDay | None = None


class AddEventRuleCall(_Call):
    """Stores a rule or a usual time."""

    tool: Literal["add_event_rule"]
    trigger_title: Title | None = None
    related_title: Title | None = None
    position: Position | None = None
    offset_minutes: Annotated[int, Field(ge=0, le=24 * 60)] | None = None
    gap_minutes: DurationMinutes | None = None
    typical_start: TimeOfDay | None = None
    typical_duration_minutes: DurationMinutes | None = None


class RemoveEventRuleCall(_Call):
    """Removes a rule or a usual time."""

    tool: Literal["remove_event_rule"]
    trigger_title: Title | None = None
    related_title: Title | None = None
    field: RuleField | None = None


class DismissHabitCall(_Call):
    """The user rejects the habit suggestion just proposed."""

    tool: Literal["dismiss_habit"]


class NoActionCall(_Call):
    """No action on the schedule: the message is not about it or cannot be understood."""

    tool: Literal["none"]


def tool_of(model: type[_Call]) -> str:
    """The tool name of a tool call model (the value of its `tool` field)."""
    return model.model_fields["tool"].annotation.__args__[0]


TOOL_CALL_MODELS: tuple[type[_Call], ...] = (
    AddEventCall,
    AddEventRelativeCall,
    UpdateEventCall,
    RemoveEventCall,
    GetScheduleCall,
    GetFreeSlotsCall,
    CheckOverlapCall,
    AddEventRuleCall,
    RemoveEventRuleCall,
    DismissHabitCall,
    NoActionCall,
)

ToolCall = Annotated[
    AddEventCall
    | AddEventRelativeCall
    | UpdateEventCall
    | RemoveEventCall
    | GetScheduleCall
    | GetFreeSlotsCall
    | CheckOverlapCall
    | AddEventRuleCall
    | RemoveEventRuleCall
    | DismissHabitCall
    | NoActionCall,
    Field(discriminator="tool"),
]

TOOL_NAMES: tuple[str, ...] = tuple(tool_of(model) for model in TOOL_CALL_MODELS)

_TOOL_CALL = TypeAdapter(ToolCall)


def parse_tool_call(payload: str | bytes | dict[str, Any]) -> ToolCall:
    """Validates the model's output (JSON or already decoded). Raises `ValidationError`."""
    if isinstance(payload, str | bytes):
        return _TOOL_CALL.validate_json(payload)
    return _TOOL_CALL.validate_python(payload)


# --- JSON schema imposed on llama-server ----------------------------------------------

# Notes and length limits do not enter the grammar; the shape of the values does: `pattern`,
# `enum` and integer bounds (verified on llama-server: it applies `exclusiveMinimum` too).
# So the model cannot write a negative duration or an hour 25.
_DROPPED = {"title", "default", "examples", "description", "minLength", "maxLength"}


def _plain(node: Any, defs: dict[str, Any]) -> Any:
    """Plain schema for a grammar: references resolved, `null` removed, no notes."""
    if isinstance(node, list):
        return [_plain(item, defs) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        return _plain(defs[node["$ref"].rsplit("/", 1)[-1]], defs)
    if "anyOf" in node:
        options = [o for o in node["anyOf"] if o.get("type") != "null"]
        if len(options) == 1:
            return _plain(options[0], defs)
        return {"anyOf": [_plain(o, defs) for o in options]}
    return {key: _plain(value, defs) for key, value in node.items() if key not in _DROPPED}


def _variant_schema(model: type[_Call]) -> dict[str, Any]:
    raw = model.model_json_schema()
    defs = raw.get("$defs", {})
    properties = {
        name: _plain(spec, defs) for name, spec in raw["properties"].items() if name != "tool"
    }
    properties = {"tool": {"const": tool_of(model)}, **properties}
    return {
        "type": "object",
        "properties": properties,
        "required": ["tool"],
        "additionalProperties": False,
    }


# Fields that exist only to answer a question of the assistant: without a pending question
# they have nothing to say, and with the full schema the model used them anyway, mistaking the
# time in "Thursday at 7 pm" for `anchor_time`.
ANSWER_ONLY_FIELDS: dict[str, frozenset[str]] = {
    "add_event": frozenset({"anchor_time", "with_gap"}),
}


def tool_call_json_schema(*, follow_up: bool = True) -> dict[str, Any]:
    """A union of tool calls, each with only its allowed fields and no `null`.

    The grammar that constrains generation is derived from this schema: the model cannot
    invent a field or a tool. `follow_up` is true when a question or a proposal is pending:
    only then are the fields needed to answer allowed (`ANSWER_ONLY_FIELDS`).
    """
    variants = []
    for model in TOOL_CALL_MODELS:
        variant = _variant_schema(model)
        if not follow_up:
            for name in ANSWER_ONLY_FIELDS.get(tool_of(model), ()):
                del variant["properties"][name]
        variants.append(variant)
    return {"oneOf": variants}
