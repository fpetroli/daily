"""Terminal commands: `python -m daily.cli chat`, `reports`, `assign` and `handle`.

It tries the whole turn (parse → execute → respond) without Telegram. Without a model
(`--llm-url`) the "model" is you: a line starting with `{` is a tool call in JSON, as the
model would write it, and goes through the same adapters, the same templates and the same
history as the real turn. Other lines get a warning.

By default the database is in memory (lost at the end of the session); `--db` uses a file.
It never touches the installed service's data, unless told to.

`reports` lists the reports (`/report`) still open, `assign` says where the fix of an open one
will go (it stays open), `handle` closes one once it is fixed, saying where. A report has no
expiry: it is open until it is fixed. They need `--db`.
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
from daily.storage.turn_log_repo import OUTCOMES, AmbiguousTurnId, Report

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


def print_reports(reports: list[Report], out: TextIO) -> None:
    """One block per report: short id, time, outcome kind, the exchange and the note."""
    if not reports:
        print("No open reports.", file=out)
        return
    for report in reports:
        if report.handled_at:
            state = f"  [fixed {report.handled_at} · {report.outcome}]"
        else:
            state = f"  [to fix → {report.outcome}]" if report.outcome else ""
        print(f"{report.turn_id[:8]}  {report.created_at}  {report.kind}{state}", file=out)
        print(f"  You:  {report.message}", file=out)
        print(f"  Bot:  {report.reply_text}", file=out)
        print(f"  Note: {report.note or '(none)'}", file=out)
        print("", file=out)


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
    reports = commands.add_parser("reports", help="list the reports still to handle")
    reports.add_argument("--db", required=True, help="SQLite file")
    reports.add_argument("--all", action="store_true", help="also the fixed ones")
    reports.add_argument("--outcome", choices=OUTCOMES, help="only those going there")
    for name, help_ in (
        ("assign", "say where the fix of an open report will go (it stays open)"),
        ("handle", "close a report once it is fixed, saying where"),
    ):
        command = commands.add_parser(name, help=help_)
        command.add_argument("--db", required=True, help="SQLite file")
        command.add_argument("turn_id", help="the id of the reported turn, or its first characters")
        command.add_argument("outcome", choices=OUTCOMES, help="where the fix goes")
    args = parser.parse_args(argv)

    if args.command == "reports":
        repos = Repositories.from_connection(open_database(args.db))
        listed = repos.turn_log.reports(include_handled=args.all, outcome=args.outcome)
        print_reports(listed, sys.stdout)
        return 0
    if args.command in ("assign", "handle"):
        repos = Repositories.from_connection(open_database(args.db))
        change = repos.turn_log.handle if args.command == "handle" else repos.turn_log.assign
        try:
            changed = change(args.turn_id, args.outcome)
        except AmbiguousTurnId:
            print(f"More than one report starts with {args.turn_id}: write more of it.")
            return 1
        if changed is None:
            print(f"No report starts with {args.turn_id}.")
            return 1
        state = "fixed" if args.command == "handle" else "to fix"
        print(f"{changed}: {state} → {args.outcome}.")
        return 0

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
