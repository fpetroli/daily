"""`python -m daily.datagen --count 500 --dialogues 250 --out data/train.jsonl`.

Writes one example per line: the chat in the 'tuned' prompt shape and the tool call as the
assistant's answer, compact as the model must write it. `--count` single messages, then
`--dialogues` conversations run through the agent, one example per turn. The same `--seed`
gives the same file.
"""

import argparse
import asyncio
import json
import logging
import random
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from daily.agent.parser import ParseRequest
from daily.agent.prompts import build_messages
from daily.agent.schemas import parse_tool_call
from daily.datagen import dialogues
from daily.datagen.single import Context, generate
from daily.datagen.values import titles
from daily.evals.cases import Case, load_cases

_TITLE_FIELDS = (
    "title", "event_description", "anchor_description", "trigger_title", "related_title",
    "new_title",
)  # fmt: skip
_FIRST_DAY = datetime(2026, 1, 1)
_DAYS = 2 * 365


def normalize(message: str) -> str:
    """Lower case, no punctuation, single spaces: "Delete the gym!" equals "delete the gym"."""
    return " ".join(re.sub(r"[^\w\s']", " ", message.lower()).split())


def eval_titles(cases: list[Case]) -> set[str]:
    """Every title the eval cases use: in the expected calls and in the starting database."""
    found: set[str] = set()
    for case in cases:
        for turn in case.turns:
            for field in _TITLE_FIELDS:
                if turn.call and isinstance(turn.call.get(field), str):
                    found.update(turn.call[field].split("|"))
        found.update(event.title for event in case.seed.events)
        found.update(rule.trigger for rule in case.seed.rules)
        found.update(rule.related for rule in case.seed.rules)
        found.update(default.title for default in case.seed.defaults)
    return found


def random_now(rng: random.Random) -> datetime:
    """A moment between 7:00 and 22:55, on a five-minute grid, in 2026-2027."""
    day = _FIRST_DAY + timedelta(days=rng.randrange(_DAYS))
    return day.replace(hour=rng.randint(7, 22), minute=rng.randrange(0, 60, 5))


def example(request: ParseRequest, call: dict[str, Any], source: str) -> dict[str, Any]:
    tool_call = parse_tool_call(call)  # raises if a generator wrote an invalid call
    answer = json.dumps(
        tool_call.model_dump(exclude_none=True), ensure_ascii=False, separators=(",", ":")
    )
    messages = build_messages(request, "tuned")
    return {
        "messages": [*messages, {"role": "assistant", "content": answer}],
        "tool": call["tool"],
        "source": source,
    }


def _prelude(
    rng: random.Random, now: datetime, pool: list[str], held_out: set[str]
) -> list[tuple[str, Any]]:
    """Half of the conversations start after one or two unrelated requests."""
    if rng.random() < 0.5:
        return []
    turns = [generate(Context(rng, now, pool)) for _ in range(rng.choice([1, 1, 2]))]
    return [(message, call) for message, call in turns if normalize(message) not in held_out]


async def _run_dialogues(
    count: int, rng: random.Random, pool: list[str], held_out: set[str], dropped: Counter[str]
) -> list[tuple[ParseRequest, dict[str, Any], str]]:
    found: list[tuple[ParseRequest, dict[str, Any], str]] = []
    done = 0
    while done < count:
        now = random_now(rng)
        name = dialogues.pick(rng)
        dialogue = dialogues.DIALOGUES[name][1](Context(rng, now, pool))
        if normalize(dialogue.turns[0].message) in held_out:
            dropped["equal to an eval case"] += 1
            continue
        try:
            turns = await dialogues.run(dialogue, now, _prelude(rng, now, pool, held_out), rng)
        except dialogues.UnexpectedReply:
            try:  # the prelude may have asked something, or met the dialogue's events
                turns = await dialogues.run(dialogue, now, [], rng)
            except dialogues.UnexpectedReply:
                dropped["unexpected reply"] += 1
                continue
        found += [(request, call, name) for request, call in turns]
        done += 1
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m daily.datagen", description=__doc__)
    parser.add_argument("--count", type=int, default=500, help="single-message examples")
    parser.add_argument("--dialogues", type=int, default=250, help="conversations to run")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True, help="JSONL file to write")
    parser.add_argument("--cases", type=Path, default=Path("evals/cases"), help="eval cases")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    held_out = {normalize(turn.say) for case in cases for turn in case.turns}
    pool = titles(eval_titles(cases))
    rng = random.Random(args.seed)

    seen: set[str] = set()
    tools: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        while sum(tools.values()) < args.count:
            now = random_now(rng)
            message, call = generate(Context(rng, now, pool))
            key = normalize(message)
            if key in held_out:
                dropped["equal to an eval case"] += 1
                continue
            if key in seen and call["tool"] != "none":
                dropped["duplicate"] += 1
                continue
            seen.add(key)
            row = example(ParseRequest(message, now), call, "single")
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            tools[call["tool"]] += 1
        singles = sum(tools.values())

        logging.disable(logging.WARNING)  # the agent logs every turn
        try:
            turns = asyncio.run(_run_dialogues(args.dialogues, rng, pool, held_out, dropped))
        finally:
            logging.disable(logging.NOTSET)
        sources: Counter[str] = Counter()
        for request, call, name in turns:
            out.write(json.dumps(example(request, call, name), ensure_ascii=False) + "\n")
            tools[call["tool"]] += 1
            sources[name] += 1

    print(
        f"{sum(tools.values())} examples in {args.out} ({len(pool)} titles): {singles} single, "
        f"{len(turns)} from {args.dialogues} conversations",
        file=sys.stderr,
    )
    for tool, count in tools.most_common():
        print(f"  {tool}: {count}", file=sys.stderr)
    for name, count in sources.most_common():
        print(f"  conversation {name}: {count}", file=sys.stderr)
    for reason, count in dropped.items():
        print(f"  dropped, {reason}: {count}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
