"""`python -m daily.evals run --parser oracle --split dev`.

The `holdout` set is held out: it is for the final measurement, **never** for tuning the
prompt or the schema (they would adapt to those cases and the measurement would lose its
value). That is why the default is `dev` and `all` warns.
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from daily.agent.prompts import PROMPT_STYLES
from daily.config import Settings
from daily.evals import latency
from daily.evals.cases import Case, CaseError, load_cases
from daily.evals.parsers import PARSERS
from daily.evals.report import Summary, render, summarize, to_json
from daily.evals.runner import CaseResult, run_cases
from daily.llm.parser import LlamaServerParser, describe

_DEFAULTS = Settings(_env_file=None)


def _progress(done: int, total: int, result: CaseResult) -> None:
    mark = "ok" if result.passed else "WRONG"
    ms = sum(t.latency_ms for t in result.turns)
    print(f"[{done}/{total}] {result.case.id} {mark} ({ms:.0f} ms)", file=sys.stderr, flush=True)


async def _run_llama(
    cases: list[Case], args: argparse.Namespace
) -> tuple[list[CaseResult], tuple[str, ...]]:
    parser = LlamaServerParser(
        args.llm_url,
        timeout=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        prompt_style=args.prompt_style,
    )
    try:
        started = time.perf_counter()
        warmed = await parser.warm_up(cases[0].now)
        print(
            f"Prompt warm-up: {'ok' if warmed else 'FAILED'} "
            f"({time.perf_counter() - started:.0f} s), outside the measurements.",
            file=sys.stderr,
            flush=True,
        )
        parser.calls.clear()
        results = await run_cases(cases, lambda case: parser, _progress)
    finally:
        await parser.aclose()
    return results, tuple(describe(parser.calls))


async def _run_latency(args: argparse.Namespace) -> int:
    parser = LlamaServerParser(
        args.llm_url,
        timeout=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        prompt_style=args.prompt_style,
    )
    start = datetime(2026, 9, 21, 15, 30)
    try:
        print("Prompt warm-up…", file=sys.stderr, flush=True)
        if not await parser.warm_up(start):
            print("The model does not answer.", file=sys.stderr)
            return 2
        parser.calls.clear()
        samples, failed = await latency.measure(parser, start, repeat=args.repeat)
    finally:
        await parser.aclose()
    print(latency.render(samples, failed))
    return 0 if samples else 1


def _add_llm_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--llm-url", default=_DEFAULTS.llm_base_url, help="llama-server, with /v1")
    command.add_argument("--llm-timeout", type=float, default=_DEFAULTS.llm_timeout_seconds)
    command.add_argument("--llm-max-tokens", type=int, default=_DEFAULTS.llm_max_tokens)
    command.add_argument(
        "--prompt-style",
        choices=PROMPT_STYLES,
        default=_DEFAULTS.llm_prompt_style,
        help="prompt shape: classic (E4B), turns (E2B) or tuned (fine-tuned model)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m daily.evals")
    commands = parser.add_subparsers(dest="command", required=True)
    speed = commands.add_parser("latency", help="measure the latency of a turn with the model")
    _add_llm_options(speed)
    speed.add_argument("--repeat", type=int, default=1, help="times to repeat the sentences")
    run = commands.add_parser("run", help="run the cases with a parser")
    run.add_argument("--parser", choices=[*sorted(PARSERS), "llama"], default="oracle")
    _add_llm_options(run)
    run.add_argument("--case", action="append", help="only this case (repeatable)")
    run.add_argument("--cases", type=Path, default=Path("evals/cases"), help="cases folder")
    run.add_argument("--section", action="append", help="only this section (repeatable)")
    run.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    run.add_argument("--json", type=Path, help="write the details to a JSON file")
    run.add_argument("--fail-under", type=float, help="exit with 1 below this percentage")
    args = parser.parse_args(argv)
    if args.command == "latency":
        return asyncio.run(_run_latency(args))

    try:
        cases = load_cases(args.cases)
    except (CaseError, FileNotFoundError) as error:
        print(f"Invalid cases: {error}", file=sys.stderr)
        return 2
    if args.case:
        cases = [c for c in cases if c.id in args.case]
    if args.section:
        cases = [c for c in cases if c.section in args.section]
    if args.split != "all":
        cases = [c for c in cases if c.split == args.split]
    else:
        print("Warning: `all` includes the held-out set (holdout).", file=sys.stderr)
    if not cases:
        print("No case selected.", file=sys.stderr)
        return 2

    notes: tuple[str, ...] = ()
    if args.parser == "llama":
        results, notes = asyncio.run(_run_llama(cases, args))
    else:
        results = asyncio.run(run_cases(cases, PARSERS[args.parser]))
    summary: Summary = summarize(results, args.parser)
    print(render(summary, notes=notes))
    if args.json:
        args.json.write_text(json.dumps(to_json(summary, results), ensure_ascii=False, indent=1))
    if args.fail_under is not None and summary.overall.percent < args.fail_under:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
