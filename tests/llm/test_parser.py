import json
from datetime import datetime

import httpx
import pytest

from daily.agent.parser import Exchange, ParseFailure, ParseRequest
from daily.agent.prompts import build_messages
from daily.agent.schemas import AddEventCall, NoActionCall, tool_call_json_schema
from daily.llm.grammar import compact_grammar
from daily.llm.parser import LlamaServerParser, LlmCall, describe

NOW = datetime(2026, 9, 21, 15, 30)
REQUEST = ParseRequest("add yoga tomorrow at 7 pm for 45 minutes", NOW)


def completion(content, *, finish="stop", **extra):
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": finish}
        ],
        "usage": {"prompt_tokens": 1500, "completion_tokens": 40},
        "timings": {"cache_n": 1400, "prompt_ms": 3300.0, "predicted_ms": 4400.0},
        **extra,
    }


def parser_with(handler, **kwargs):
    return LlamaServerParser("http://llm.test/v1", transport=httpx.MockTransport(handler), **kwargs)


def answering(body, status=200):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return (
            httpx.Response(status, json=body)
            if not isinstance(body, str)
            else httpx.Response(status, text=body)
        )

    return handler, seen


async def test_a_valid_tool_call_and_the_request_is_the_expected_one():
    handler, seen = answering(
        completion(
            '{"tool":"add_event","title":"yoga","date":"2026-09-22","time":"19:00",'
            '"duration_minutes":45}'
        )
    )
    parser = parser_with(handler)

    result = await parser.parse(REQUEST)

    assert result == AddEventCall(
        tool="add_event", title="yoga", date="2026-09-22", time="19:00", duration_minutes=45
    )
    [request] = seen
    assert request.url == "http://llm.test/v1/chat/completions"
    body = json.loads(request.content)
    assert body["messages"] == build_messages(REQUEST)
    assert body["temperature"] == 0 and body["stream"] is False and body["cache_prompt"] is True
    assert body["max_tokens"] == 160
    assert "response_format" not in body
    assert body["grammar"] == compact_grammar(tool_call_json_schema(follow_up=False))


async def test_history_goes_in_the_turn_not_in_the_prefix():
    handler, seen = answering(completion('{"tool":"none"}'))
    request = ParseRequest(
        "yes", NOW, history=(Exchange("add yoga", 'What time do you want "yoga"?'),)
    )

    await parser_with(handler).parse(request)

    *prefix, user = json.loads(seen[0].content)["messages"]
    assert "yoga" not in json.dumps(prefix)  # the history does not enter the cached prefix
    assert "User: add yoga" in user["content"]


async def test_with_a_pending_question_the_grammar_allows_answer_fields():
    handler, seen = answering(completion('{"tool":"none"}'))
    parser = parser_with(handler)
    question = ParseRequest("no", NOW, history=(Exchange("add dinner", "Shall I keep the gap?"),))

    await parser.parse(question)
    await parser.parse(REQUEST)

    def add_event_rule(grammar: str) -> str:
        return next(x for x in grammar.splitlines() if x.startswith("call-add-event ::="))

    with_history, without = (add_event_rule(json.loads(r.content)["grammar"]) for r in seen)
    assert "with_gap" in with_history and "anchor_time" in with_history
    assert "with_gap" not in without and "anchor_time" not in without


async def test_no_action():
    handler, _ = answering(completion('{"tool":"none"}'))

    assert await parser_with(handler).parse(REQUEST) == NoActionCall(tool="none")


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "",
        '{"tool":"invented"}',
        '{"tool":"add_event","time":"tomorrow at 7 pm"}',
        '{"tool":"add_event","time":"19:00"',  # truncated
    ],
)
async def test_invalid_output_is_invalid_output_with_the_model_text(content):
    handler, _ = answering(completion(content))

    result = await parser_with(handler).parse(REQUEST)

    assert isinstance(result, ParseFailure) and result.kind == "invalid_output"
    assert result.detail == content[:300]


async def test_a_response_without_choices_is_invalid_output():
    handler, _ = answering({"choices": []})

    result = await parser_with(handler).parse(REQUEST)

    assert result.kind == "invalid_output" and "unexpected response" in result.detail


async def test_a_non_json_body_is_invalid_output():
    handler, _ = answering("<html>proxy</html>")

    assert (await parser_with(handler).parse(REQUEST)).kind == "invalid_output"


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.ConnectTimeout("slow"), "timeout"),
        (httpx.ConnectError("refused"), "unavailable"),
        (httpx.RemoteProtocolError("closed"), "unavailable"),
    ],
)
async def test_network_errors_are_timeout_or_unavailable(error, kind):
    def handler(request):
        raise error

    result = await parser_with(handler).parse(REQUEST)

    assert isinstance(result, ParseFailure) and result.kind == kind


@pytest.mark.parametrize(
    ("status", "kind"), [(500, "unavailable"), (503, "unavailable"), (400, "error")]
)
async def test_http_errors(status, kind):
    handler, _ = answering("Loading model", status)

    result = await parser_with(handler).parse(REQUEST)

    assert result.kind == kind and f"HTTP {status}" in result.detail


async def test_call_statistics_are_recorded():
    handler, _ = answering(completion('{"tool":"none"}'))
    parser = parser_with(handler)

    await parser.parse(REQUEST)

    assert parser.calls == [LlmCall(1500, 1400, 40, 3300.0, 4400.0, "stop")]


async def test_a_response_truncated_by_the_token_limit_is_reported():
    handler, _ = answering(completion('{"tool":"add_event"', finish="length"))
    parser = parser_with(handler, max_tokens=5)

    result = await parser.parse(REQUEST)

    assert result.kind == "invalid_output"
    assert parser.calls[0].finish_reason == "length"
    assert "1 answers truncated" in " ".join(describe(parser.calls))


def test_summary_shows_tokens_and_times():
    calls = [
        LlmCall(1500, 1400, 40, 3300.0, 4400.0, "stop"),
        LlmCall(1600, 1500, 60, 100.0, 6000.0, "stop"),
    ]

    lines = describe(calls)

    assert lines[0].startswith("Model: 2 calls · prompt tokens (median) 1600")
    assert "generated tokens (median) 60" in lines[0]
    assert describe([]) == []


async def test_the_client_closes():
    parser = parser_with(lambda request: httpx.Response(200, json=completion('{"tool":"none"}')))

    await parser.aclose()

    with pytest.raises(RuntimeError):
        await parser.parse(REQUEST)


async def test_warm_up_sends_the_day_prompt_with_a_single_generated_token():
    handler, seen = answering(completion("{"))
    parser = parser_with(handler)

    assert await parser.warm_up(NOW) is True

    body = json.loads(seen[0].content)
    assert body["max_tokens"] == 1
    assert body["messages"][0]["content"] == build_messages(REQUEST)[0]["content"]
    assert parser.calls == []  # not a parse call


async def test_warm_up_does_not_raise_if_the_server_does_not_answer():
    def handler(request):
        raise httpx.ConnectError("refused")

    assert await parser_with(handler).warm_up(NOW) is False
