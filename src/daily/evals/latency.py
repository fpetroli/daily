"""Latency of a turn as it would be in production.

In the eval cases the time is always the same (15:30): the turn header stays cached and the
latency comes out optimistic. In production the time changes every minute, and from there on
the prompt is processed again. Here every request has a different time, so the prompt tail
costs what it really will. The sentences are standalone messages of every kind; they are not
for judging correctness, only for measuring time.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from daily.agent.parser import ParseFailure, ParseRequest
from daily.evals.report import percentile
from daily.llm.parser import LlamaServerParser, LlmCall

MESSAGES = (
    "add gym tomorrow at 6 pm for an hour",
    "what am I doing on saturday morning",
    "move the dentist by an hour",
    "delete tomorrow's meeting",
    "do I have a free hour on tuesday?",
    "yoga on thursday at 7 pm for 45 minutes",
    "put the shower after the gym",
    "remember that mass is at 10",
    "lengthen tennis by 20 minutes",
    "what time is my lunch tomorrow?",
    "thanks a lot",
    "add dinner with Paolo on friday at 9 pm for two hours",
    "remove the 9 pm event",
    "can I put a checkup today from 12 to 1 pm?",
    "in an hour I have a half-hour call",
    "add reading",
    "move the lesson to thursday at 5 pm",
    "on October 12 at 9 I have the doctor for 30 minutes",
)
CLOCK_STEP = timedelta(minutes=7)


@dataclass(frozen=True, slots=True)
class Sample:
    wall_ms: float
    call: LlmCall


async def measure(
    parser: LlamaServerParser, start: datetime, *, repeat: int = 1
) -> tuple[list[Sample], int]:
    """(samples, failed requests). Every request has a different time from the previous one."""
    samples: list[Sample] = []
    failed = 0
    step = 0
    for _ in range(repeat):
        for message in MESSAGES:
            before = len(parser.calls)
            started = time.perf_counter()
            result = await parser.parse(ParseRequest(message, start + CLOCK_STEP * step))
            wall_ms = (time.perf_counter() - started) * 1000
            step += 1
            if isinstance(result, ParseFailure) and len(parser.calls) == before:
                failed += 1
                continue
            samples.append(Sample(wall_ms, parser.calls[-1]))
    return samples, failed


def render(samples: list[Sample], failed: int = 0) -> str:
    if not samples:
        return "No successful request.\n"
    turns = [s.wall_ms for s in samples]
    tails = [s.call.prompt_tokens - s.call.cached_tokens for s in samples]
    tail_ms = [s.call.prompt_ms for s in samples]
    generated = [s.call.generated_tokens for s in samples]
    generation_ms = [s.call.generation_ms for s in samples]
    speed = 1000 * sum(generated) / max(sum(generation_ms), 1)
    lines = [
        f"# Latency per turn (different time on every request), {len(samples)} requests",
        "",
        f"Turn: p50 {percentile(turns, 0.5) / 1000:.1f} s · "
        f"p95 {percentile(turns, 0.95) / 1000:.1f} s · max {max(turns) / 1000:.1f} s",
        f"Prompt tail (to process again): median {percentile(tails, 0.5):.0f} tokens, "
        f"{percentile(tail_ms, 0.5) / 1000:.2f} s",
        f"Generation: median {percentile(generated, 0.5):.0f} tokens, "
        f"{percentile(generation_ms, 0.5) / 1000:.2f} s ({speed:.1f} tokens per second)",
    ]
    if failed:
        lines.append(f"Warning: {failed} failed requests, excluded from the measurements.")
    return "\n".join(lines) + "\n"
