import json

import pytest
from pydantic import ValidationError

from daily.agent.schemas import (
    TOOL_CALL_MODELS,
    TOOL_NAMES,
    AddEventCall,
    GetScheduleCall,
    NoActionCall,
    parse_tool_call,
    tool_call_json_schema,
    tool_of,
)

# The sentences of the early experiment and the tool calls the model produced for them.
SPIKE_OUTPUTS = [
    '{"tool":"add_event","title":"gym","date":"2026-09-22","time":"18:00","end_time":"19:00"}',
    '{"tool":"update_event","new_date":"2026-09-23","new_time":"09:00"}',
    '{"tool":"get_schedule","date":"2026-09-24","time_band":"afternoon"}',
    '{"tool":"get_free_slots","date":"2026-09-22","time_band":"evening","min_duration_minutes":60}',
    '{"tool":"remove_event","event_date":"2026-09-21"}',
    '{"tool":"update_event","shift_minutes":-30}',
    '{"tool":"add_event_relative","position":"after"}',
    '{"tool":"none"}',
    '{"tool":"add_event_rule","trigger_title":"shower","related_title":"gym","position":"after"}',
]


def test_eleven_tools_with_unique_names():
    assert len(TOOL_CALL_MODELS) == 11
    assert len(set(TOOL_NAMES)) == 11
    assert {"add_event", "dismiss_habit", "none", "get_free_slots"} <= set(TOOL_NAMES)


@pytest.mark.parametrize("payload", SPIKE_OUTPUTS)
def test_outputs_from_the_early_experiment_are_valid(payload):
    assert parse_tool_call(payload).tool in TOOL_NAMES


def test_accepts_json_bytes_and_dict():
    payload = {"tool": "add_event", "title": "yoga", "date": "2026-09-22", "time": "19:00"}

    assert parse_tool_call(json.dumps(payload)) == parse_tool_call(json.dumps(payload).encode())
    assert parse_tool_call(payload) == AddEventCall(**payload)


def test_omitted_fields_are_none():
    call = parse_tool_call('{"tool":"add_event","title":"yoga"}')

    fields = (call.date, call.time, call.end_time, call.duration_minutes, call.with_gap)
    assert fields == (None, None, None, None, None)


def test_an_explicit_null_counts_as_omitted():
    call = parse_tool_call('{"tool":"add_event","title":"yoga","date":null,"time":null}')

    assert call.date is None and call.time is None


def test_no_action_and_dismiss_have_no_fields():
    assert isinstance(parse_tool_call('{"tool":"none"}'), NoActionCall)
    assert parse_tool_call({"tool": "dismiss_habit"}).tool == "dismiss_habit"
    with pytest.raises(ValidationError):
        parse_tool_call({"tool": "dismiss_habit", "title": "gym"})  # no fields


def test_time_band_is_an_enum():
    assert parse_tool_call({"tool": "get_schedule", "time_band": "evening"}) == GetScheduleCall(
        tool="get_schedule", time_band="evening"
    )
    with pytest.raises(ValidationError):
        parse_tool_call({"tool": "get_schedule", "time_band": "dawn"})


@pytest.mark.parametrize(
    "payload",
    [
        {"tool": "invented"},
        {"title": "no tool"},
        {"tool": "add_event", "name": "unknown field"},
        {"tool": "add_event", "time": "tomorrow at 6 pm"},
        {"tool": "add_event", "time": "25:00"},
        {"tool": "add_event", "date": "2026-02-30"},
        {"tool": "add_event", "date": "22/09/2026"},
        {"tool": "add_event", "anchor_time": "18"},
        {"tool": "add_event", "title": ""},
        {"tool": "add_event", "duration_minutes": 0},
        {"tool": "add_event", "duration_minutes": "an hour"},
        {"tool": "add_event", "with_gap": "maybe"},
        {"tool": "update_event", "shift_minutes": 99999},
        {"tool": "add_event_relative", "position": "beside"},
        {"tool": "add_event_relative", "offset_minutes": -5},
        {"tool": "remove_event_rule", "field": "start"},
    ],
)
def test_invalid_values_and_fields_are_rejected(payload):
    with pytest.raises(ValidationError):
        parse_tool_call(payload)


def test_invalid_json_is_rejected():
    with pytest.raises(ValidationError):
        parse_tool_call("{non json")


# --- the schema imposed on the server --------------------------------------------------


def test_schema_is_a_union_with_one_variant_per_tool():
    schema = tool_call_json_schema()

    assert len(schema["oneOf"]) == len(TOOL_CALL_MODELS)
    assert [v["properties"]["tool"]["const"] for v in schema["oneOf"]] == list(TOOL_NAMES)


def test_each_variant_allows_only_its_model_fields():
    for model, variant in zip(TOOL_CALL_MODELS, tool_call_json_schema()["oneOf"], strict=True):
        assert set(variant["properties"]) == set(model.model_fields), tool_of(model)
        assert variant["required"] == ["tool"]
        assert variant["additionalProperties"] is False


def test_schema_is_flat_and_without_null():
    text = json.dumps(tool_call_json_schema())

    # `"title": "` would be the JSON Schema annotation; the event's `title` field is another
    # matter and has an object as its value.
    for forbidden in ('"$ref"', '"$defs"', '"anyOf"', '"null"', '"title": "', '"discriminator"'):
        assert forbidden not in text, forbidden


def test_schema_constrains_the_shape_of_values():
    variants = {v["properties"]["tool"]["const"]: v["properties"] for v in _variants()}

    assert variants["get_schedule"]["time_band"]["enum"] == [
        "morning",
        "afternoon",
        "evening",
        "night",
    ]
    assert variants["add_event_relative"]["position"]["enum"] == ["before", "after"]
    date_pattern = variants["add_event"]["date"]["pattern"]
    assert date_pattern.startswith("^20[0-9]{2}-(0[1-9]|1[0-2])-") and date_pattern.endswith("$")
    assert variants["add_event"]["time"]["pattern"] == r"^([01][0-9]|2[0-3]):[0-5][0-9]$"
    assert variants["add_event"]["anchor_time"]["type"] == "string"
    assert variants["add_event"]["with_gap"]["type"] == "boolean"
    assert variants["update_event"]["shift_minutes"] == {
        "type": "integer",
        "minimum": -10080,
        "maximum": 10080,
    }
    assert variants["add_event"]["duration_minutes"] == {
        "type": "integer",
        "exclusiveMinimum": 0,
        "maximum": 1440,
    }
    assert variants["add_event_relative"]["offset_minutes"]["minimum"] == 0


def _variants():
    return tool_call_json_schema()["oneOf"]


def test_patterns_do_not_use_d_which_the_llama_cpp_converter_ignores():
    """With `\\d` the server constrains nothing and the model writes "13:30" instead of an ISO
    date (measured in the eval): the patterns use `[0-9]`."""
    text = json.dumps(tool_call_json_schema())

    assert "\\\\d" not in text
    assert "[0-9]" in text


@pytest.mark.parametrize("value", ["2026-13-01", "2026-09-32", "2026-9-22", "22/09/2026"])
def test_patterns_reject_impossible_months_and_days(value):
    with pytest.raises(ValidationError):
        parse_tool_call({"tool": "add_event", "date": value})


@pytest.mark.parametrize("value", ["25:00", "10:60", "9:00", "2026-09-22T10:00"])
def test_patterns_reject_impossible_hours_and_minutes(value):
    with pytest.raises(ValidationError):
        parse_tool_call({"tool": "add_event", "time": value})
