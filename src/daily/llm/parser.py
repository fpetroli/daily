"""The parser with the real model: a direct call to `llama-server`.

It does not go through Agno's `Agent` or `output_schema`, which would make every field
required: we impose the schema, with optional fields and no `null`, on the server ourselves,
as a grammar. The model therefore cannot invent a tool or a field; `parse_tool_call` validates
the values after generation.

The request is deterministic (`temperature` 0) and asks the server to reuse the prompt cache:
the prefix (instructions and tools, then the calendar) is identical from one turn to the next
and only the current turn is processed again. Reasoning is switched off on the server
(`--reasoning off`).

Every error becomes a `ParseFailure` with its kind: the turn ends with a fixed message.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from pydantic import ValidationError

from daily.agent.parser import ParseFailure, ParseRequest
from daily.agent.prompts import DEFAULT_STYLE, PromptStyle, build_messages
from daily.agent.schemas import ToolCall, parse_tool_call, tool_call_json_schema
from daily.llm.grammar import compact_grammar


@dataclass(frozen=True, slots=True)
class LlmCall:
    """What a call cost: it shows where the latency comes from."""

    prompt_tokens: int
    cached_tokens: int  # of the prompt tokens, how many the server already had cached
    generated_tokens: int
    prompt_ms: float  # processing of the new prompt tokens only
    generation_ms: float
    finish_reason: str | None


class LlamaServerParser:
    """A `Parser` that asks `llama-server` (OpenAI-compatible API) for the tool call."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 300.0,
        max_tokens: int = 160,
        model: str = "gemma",
        prompt_style: PromptStyle = DEFAULT_STYLE,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """`timeout` is the read timeout: it must cover cold processing of the prompt (tens
        of seconds with the prefix not cached); the connection has a short cap.
        `prompt_style` is the prompt shape: 'classic' for E4B, 'turns' for E2B."""
        self._max_tokens = max_tokens
        self._model = model
        self._style = prompt_style
        # Two grammars: with a pending question the answer fields are allowed.
        self._grammars = {
            follow_up: compact_grammar(tool_call_json_schema(follow_up=follow_up))
            for follow_up in (False, True)
        }
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=3.0),
            transport=transport,
        )
        self.calls: list[LlmCall] = []

    async def aclose(self) -> None:
        await self._client.aclose()

    async def warm_up(self, now: datetime) -> bool:
        """Makes the server process the prompt prefix (instructions, calendar, examples).

        Cold, it takes tens of seconds (~30 tokens per second): better to pay them before the
        first message, at startup and on every change of day (the calendar is in the prefix).
        A dummy request with a single generated token; the wait is generous because this is
        precisely the slow case. It does not raise: `False` if the server did not answer.
        """
        body = self.request_body(ParseRequest("hi", now))
        body["max_tokens"] = 1
        try:
            response = await self._client.post("/chat/completions", json=body, timeout=900.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def request_body(self, request: ParseRequest) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": build_messages(request, self._style),
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "stream": False,
            "cache_prompt": True,
            "grammar": self._grammars[bool(request.history)],
        }

    async def parse(self, request: ParseRequest) -> ToolCall | ParseFailure:
        try:
            response = await self._client.post("/chat/completions", json=self.request_body(request))
        except httpx.TimeoutException as error:
            return ParseFailure("timeout", type(error).__name__)
        except httpx.HTTPError as error:
            return ParseFailure("unavailable", f"{type(error).__name__}: {error}")

        if response.status_code >= 500:
            return ParseFailure(
                "unavailable", f"HTTP {response.status_code}: {response.text[:200]}"
            )
        if response.status_code >= 400:
            return ParseFailure("error", f"HTTP {response.status_code}: {response.text[:200]}")

        try:
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
            return ParseFailure("invalid_output", f"unexpected response: {response.text[:200]}")

        self.calls.append(_stats(data, choice))
        try:
            return parse_tool_call(content)
        except ValidationError:
            return ParseFailure("invalid_output", content[:300])


def _stats(data: dict[str, Any], choice: dict[str, Any]) -> LlmCall:
    usage = data.get("usage") or {}
    timings = data.get("timings") or {}
    return LlmCall(
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        cached_tokens=int(timings.get("cache_n", 0)),
        generated_tokens=int(usage.get("completion_tokens", 0)),
        prompt_ms=float(timings.get("prompt_ms", 0.0)),
        generation_ms=float(timings.get("predicted_ms", 0.0)),
        finish_reason=choice.get("finish_reason"),
    )


def describe(calls: list[LlmCall]) -> list[str]:
    """Summary lines for the eval report: where the time goes."""
    if not calls:
        return []

    def median(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    truncated = sum(1 for c in calls if c.finish_reason == "length")
    lines = [
        f"Model: {len(calls)} calls · prompt tokens (median) "
        f"{median([c.prompt_tokens for c in calls]):.0f}, of which cached "
        f"{median([c.cached_tokens for c in calls]):.0f} · generated tokens (median) "
        f"{median([c.generated_tokens for c in calls]):.0f}",
        f"Time (median): prompt {median([c.prompt_ms for c in calls]):.0f} ms, generation "
        f"{median([c.generation_ms for c in calls]):.0f} ms",
    ]
    if truncated:
        lines.append(f"Warning: {truncated} answers truncated by the token limit.")
    return lines
