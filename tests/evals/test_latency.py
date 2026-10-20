import json
import re
from datetime import datetime

import httpx

from daily.evals import latency
from daily.evals.__main__ import main
from daily.llm.parser import LlamaServerParser

START = datetime(2026, 9, 21, 15, 30)


def parser_answering(seen: list[str], *, prompt_ms=900.0, predicted_ms=3600.0):
    def handler(request: httpx.Request) -> httpx.Response:
        user = json.loads(request.content)["messages"][-1]["content"]
        seen.append(user)
        body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": '{"tool":"none"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3900, "completion_tokens": 30},
            "timings": {"cache_n": 3870, "prompt_ms": prompt_ms, "predicted_ms": predicted_ms},
        }
        return httpx.Response(200, json=body)

    return LlamaServerParser("http://llm.test/v1", transport=httpx.MockTransport(handler))


async def test_every_request_has_a_different_time():
    """It is the point of the measurement: with a fixed time the prompt tail would stay cached."""
    seen: list[str] = []
    parser = parser_answering(seen)

    samples, failed = await latency.measure(parser, START, repeat=2)

    assert failed == 0 and len(samples) == 2 * len(latency.MESSAGES)
    clocks = [re.search(r"Current time: (\d\d:\d\d)", user).group(1) for user in seen]
    assert len(set(clocks)) == len(clocks)


async def test_report_shows_turn_tail_and_generation():
    parser = parser_answering([], prompt_ms=1200.0, predicted_ms=3000.0)

    samples, failed = await latency.measure(parser, START)
    text = latency.render(samples, failed)

    assert "18 requests" in text
    assert "Turn: p50" in text and "p95" in text
    assert "median 30 tokens, 3.00 s (10.0 tokens per second)" in text
    assert "Prompt tail (to process again): median 30 tokens, 1.20 s" in text
    assert "failed" not in text


async def test_a_failed_request_does_not_skew_the_measurements():
    def handler(request):
        raise httpx.ConnectError("refused")

    parser = LlamaServerParser("http://llm.test/v1", transport=httpx.MockTransport(handler))

    samples, failed = await latency.measure(parser, START)

    assert samples == [] and failed == len(latency.MESSAGES)
    assert latency.render(samples, failed) == "No successful request.\n"


def test_command_exists_and_stops_if_the_model_does_not_answer(capsys):
    assert main(["latency", "--llm-url", "http://127.0.0.1:9/v1", "--llm-timeout", "1"]) == 2
    assert "does not answer" in capsys.readouterr().err
