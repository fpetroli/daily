"""Terminal command: `python -m daily.cli chat`.

It tries the whole turn (parse → execute → respond) without Telegram. Without a model
(`--llm-url`) the "model" is you: a line starting with `{` is a tool call in JSON, as the
model would write it, and goes through the same adapters, the same templates and the same
history as the real turn. Other lines get a warning.

By default the database is in memory (lost at the end of the session); `--db` uses a file.
It never touches the installed service's data, unless told to.
"""

import argparse
import asyncio
import sys
from collections.abc import Iterable
from datetime import datetime
from typing import TextIO

from daily.agent.parser import ParseFailure, Parser, ParseRequest
from daily.agent.schemas import ToolCall, parse_tool_call
from daily.agent.turn import TurnService
from daily.config import Settings
from daily.llm.parser import LlamaServerParser
from daily.services.context import Repositories
from daily.storage.db import MEMORY, open_database

NO_MODEL = 'No model connected: write the tool call in JSON, for example {"tool": "get_schedule"}.'


class JsonLineParser:
    """The terminal parser: the line is the tool call in JSON."""

    async def parse(self, request: ParseRequest) -> ToolCall | ParseFailure:
        text = request.message.strip()
        if not text.startswith("{"):
            return ParseFailure("unavailable", NO_MODEL)
        try:
            return parse_tool_call(text)
        except ValueError as error:
            return ParseFailure("invalid_output", str(error))


async def run_chat(
    lines: Iterable[str],
    out: TextIO,
    *,
    repos: Repositories,
    now: datetime,
    timezone: str | None = None,
    parser: Parser | None = None,
) -> None:
    """One turn per line. Prints the reply, the quick replies and whether it awaits an answer.

    Without `parser` the "model" is the terminal itself (`JsonLineParser`): a line that does
    not start with `{` gets a warning and runs no turn. With a real parser (`--llm-url`) every
    non-empty line is a natural-language message, as the user would write it.
    """
    json_only = parser is None
    service = TurnService(repos, parser or JsonLineParser())
    for line in lines:
        message = line.strip()
        if not message:
            continue
        if json_only and not message.startswith("{"):
            print(f"! {NO_MODEL}", file=out)
            continue
        result = await service.handle(message, now=now, timezone=timezone)
        first, *rest = result.reply.text.split("\n")  # a multi-line reply stays one block
        print("\n".join([f"> {first}", *(f"  {line}" if line else "" for line in rest)]), file=out)
        if result.reply.quick_replies:
            print(f"  [{' | '.join(result.reply.quick_replies)}]", file=out)


async def _run(args: argparse.Namespace, repos: Repositories, now: datetime) -> None:
    llama: LlamaServerParser | None = None
    model_parser: Parser | None = None
    if args.llm_url:
        defaults = Settings(_env_file=None)
        llama = LlamaServerParser(
            args.llm_url,
            timeout=defaults.llm_timeout_seconds,
            max_tokens=defaults.llm_max_tokens,
            prompt_style=args.prompt_style or defaults.llm_prompt_style,
        )
        model_parser = llama
    try:
        await run_chat(
            _stdin_lines(),
            sys.stdout,
            repos=repos,
            now=now,
            timezone=args.timezone,
            parser=model_parser,
        )
    finally:
        if llama is not None:
            await llama.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m daily.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="try the turn from the terminal")
    chat.add_argument("--db", default=MEMORY, help="SQLite file (default: in memory)")
    chat.add_argument(
        "--now",
        help="simulated local time YYYY-MM-DDTHH:MM (default: the machine's time)",
    )
    chat.add_argument("--timezone", help="IANA time zone, for example Europe/Rome")
    chat.add_argument(
        "--llm-url",
        help="llama-server URL (e.g. http://127.0.0.1:8082/v1): uses the real model instead "
        "of a hand-written JSON tool call.",
    )
    chat.add_argument(
        "--prompt-style",
        choices=["classic", "turns"],
        default=None,
        help="prompt shape: 'classic' for E4B, 'turns' for E2B (default: the one in "
        "Settings/DAILY_LLM_PROMPT_STYLE). Requires --llm-url.",
    )
    args = parser.parse_args(argv)

    now = datetime.strptime(args.now, "%Y-%m-%dT%H:%M") if args.now else datetime.now()
    repos = Repositories.from_connection(open_database(args.db))
    print(f"Simulated time: {now:%Y-%m-%d %H:%M}. Empty line or Ctrl-D to quit.", file=sys.stderr)
    asyncio.run(_run(args, repos, now))
    return 0


def _stdin_lines() -> Iterable[str]:
    for line in sys.stdin:
        if not line.strip():
            return
        yield line


if __name__ == "__main__":
    raise SystemExit(main())
