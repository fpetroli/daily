"""The Telegram bot: a message becomes a turn, a button becomes the user's answer, and the
commands read or change the schedule without the model.

The bot asks Telegram for new messages (long polling): the service opens no port. Only the
user ids in the allowlist are served, and only in private chats: the schedule is personal.
The time of a turn is the sending time of the message in the configured time zone.

Updates are handled one at a time, in order: one user, one model, and a turn already
serializes on its own lock. Only the buttons of the last reply act: the model reads the
answer against the recent history, so an answer to an older question would be misread.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import httpx

from daily.agent.execute import execute
from daily.agent.results import Executed, Execution
from daily.agent.schemas import parse_tool_call
from daily.agent.templates import Reply, Styled, render, rules_text, undone_text
from daily.agent.turn import TurnService
from daily.services.context import Repositories
from daily.services.schedule import EventCreated, EventRemoved, EventUpdated
from daily.services.undo import undo_last_action
from daily.storage.db import transaction
from daily.telegram.api import Button, TelegramError

logger = logging.getLogger("daily.telegram")

COMMANDS = [
    ("today", "Today's schedule"),
    ("tomorrow", "Tomorrow's schedule"),
    ("rules", "Rules and usual times"),
    ("undo", "Undo the last change"),
    ("report", "Flag the last reply: /report what went wrong"),
    ("settings", "Habit detection settings"),
    ("help", "What I can do"),
]
HELP = (
    'Write to me in plain words, for example "gym tomorrow at 18:00 for an hour", "move the '
    'dentist to Friday at 10" or "what do I have on Monday?".\n\nCommands:\n'
    + "\n".join(f"/{name} - {text}" for name, text in COMMANDS)
)
NOT_ALLOWED = (
    "This bot is private. Your Telegram user id is {user_id}: add it to "
    "DAILY_TELEGRAM_ALLOWED_USERS to use it."
)
ONLY_TEXT = "I can only read text messages."
UNKNOWN_COMMAND = "I don't know that command.\n\n" + HELP
EXPIRED = "This button has expired."
NOTHING_TO_REPORT = "There is no reply to report yet."
REPORTED = "Thanks: I flagged the last reply."
UNDO = "Undo"

# key: (minimum, maximum, unit, meaning)
SETTINGS = {
    "repetition_threshold": (1, 10, "", "repetitions before a habit is proposed"),
    "time_tolerance": (0, 60, " minutes", "how far two times can be and still match"),
    "duration_tolerance": (0, 60, " minutes", "how far two durations can be and still match"),
}

_TYPING_SECONDS = 4.0  # Telegram shows "typing" for about 5 s
_RETRY_SECONDS = (1.0, 60.0)  # first and longest wait after a failed `getUpdates`
_SHORT_LABEL = 12  # quick replies up to this length share a row


class Api(Protocol):
    async def get_me(self) -> dict[str, Any]: ...
    async def get_updates(self, offset: int | None) -> list[dict[str, Any]]: ...
    async def send_message(
        self,
        chat_id: int,
        text: str,
        buttons: list[list[Button]] | None = None,
        styles: Sequence[Styled] = (),
    ) -> dict[str, Any]: ...
    async def edit_message_text(self, chat_id: int, message_id: int, text: str) -> None: ...
    async def remove_buttons(self, chat_id: int, message_id: int) -> None: ...
    async def answer_callback_query(
        self, callback_query_id: str, text: str | None = None
    ) -> None: ...
    async def send_typing(self, chat_id: int) -> None: ...
    async def set_commands(self, commands: list[tuple[str, str]]) -> None: ...


@dataclass(frozen=True, slots=True)
class _Prompt:
    """The last reply that has buttons: the only one whose buttons act."""

    seq: int
    chat_id: int
    message_id: int
    text: str
    quick_replies: tuple[str, ...]


class Bot:
    def __init__(
        self,
        api: Api,
        turn: TurnService,
        repos: Repositories,
        *,
        allowed_users: frozenset[int],
        timezone: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._api = api
        self._turn = turn
        self._repos = repos
        self._allowed = allowed_users
        self._timezone = timezone
        self._zone = ZoneInfo(timezone) if timezone else None
        self._clock = clock
        self._seq = 0
        self._prompt: _Prompt | None = None

    async def run(self) -> None:
        """Polls Telegram forever. Raises only when the token is refused."""
        me = await self._api.get_me()
        logger.info("Connected to Telegram as @%s: waiting for messages.", me.get("username"))
        try:
            await self._api.set_commands(COMMANDS)
        except (TelegramError, httpx.HTTPError):
            logger.warning("Could not register the command menu")
        offset: int | None = None
        wait = _RETRY_SECONDS[0]
        while True:
            try:
                updates = await self._api.get_updates(offset)
            except TelegramError as error:
                if error.code in (401, 404):
                    raise  # a wrong token: retrying does not help
                logger.warning("Telegram refused the polling (%s), retrying in %.0f s", error, wait)
            except httpx.HTTPError as error:
                logger.warning(
                    "Telegram unreachable (%s), retrying in %.0f s", type(error).__name__, wait
                )
            else:
                wait = _RETRY_SECONDS[0]
                for update in updates:
                    offset = update["update_id"] + 1
                    await self.handle_update(update)
                continue
            await asyncio.sleep(wait)
            wait = min(wait * 2, _RETRY_SECONDS[1])

    async def handle_update(self, update: dict[str, Any]) -> None:
        """Never raises: a failed update is logged and the next one is handled."""
        try:
            if (message := update.get("message")) is not None:
                await self._on_message(message)
            elif (query := update.get("callback_query")) is not None:
                await self._on_button(query)
        except Exception:
            logger.exception("Update %s failed", update.get("update_id"))

    # --- incoming messages ---------------------------------------------------------

    async def _on_message(self, message: dict[str, Any]) -> None:
        chat = message.get("chat") or {}
        if chat.get("type") != "private":
            return
        chat_id = chat["id"]
        user_id = (message.get("from") or {}).get("id")
        if user_id not in self._allowed:
            await self._api.send_message(chat_id, NOT_ALLOWED.format(user_id=user_id))
            return
        text = (message.get("text") or "").strip()
        if not text:
            await self._api.send_message(chat_id, ONLY_TEXT)
            return
        now = self._now(message.get("date"))
        if text.startswith("/"):
            await self._command(chat_id, text, now)
        else:
            await self._run_turn(chat_id, text, now)

    async def _run_turn(self, chat_id: int, text: str, now: datetime) -> None:
        typing = asyncio.ensure_future(self._keep_typing(chat_id))
        try:
            result = await self._turn.handle(text, now=now, timezone=self._timezone)
        finally:
            typing.cancel()
            with suppress(asyncio.CancelledError):
                await typing
        await self._close_prompt()
        self._seq += 1
        reply = result.reply
        buttons = _buttons(self._seq, reply.quick_replies, _wrote_schedule(result.execution))
        sent = await self._api.send_message(chat_id, reply.text, buttons, reply.styles)
        if buttons:
            self._prompt = _Prompt(
                self._seq, chat_id, sent["message_id"], reply.text, reply.quick_replies
            )

    async def _keep_typing(self, chat_id: int) -> None:
        while True:
            with suppress(TelegramError, httpx.HTTPError):
                await self._api.send_typing(chat_id)
            await asyncio.sleep(_TYPING_SECONDS)

    async def _close_prompt(self) -> None:
        """Takes the buttons off the last reply: a newer message makes them stale."""
        prompt, self._prompt = self._prompt, None
        if prompt is not None:
            with suppress(TelegramError, httpx.HTTPError):
                await self._api.remove_buttons(prompt.chat_id, prompt.message_id)

    # --- buttons -------------------------------------------------------------------

    async def _on_button(self, query: dict[str, Any]) -> None:
        if (query.get("from") or {}).get("id") not in self._allowed:
            await self._api.answer_callback_query(query["id"])
            return
        kind, seq, index = _parse_button(query.get("data") or "")
        prompt = self._prompt
        if (
            prompt is None
            or seq != prompt.seq
            or kind not in ("q", "u")
            or (kind == "q" and index >= len(prompt.quick_replies))
        ):
            await self._api.answer_callback_query(query["id"], EXPIRED)
            message = query.get("message") or {}
            if "chat" in message:
                with suppress(TelegramError, httpx.HTTPError):
                    await self._api.remove_buttons(message["chat"]["id"], message["message_id"])
            return
        await self._api.answer_callback_query(query["id"])
        self._prompt = None
        now = self._now(None)
        if kind == "u":
            with suppress(TelegramError, httpx.HTTPError):
                await self._api.remove_buttons(prompt.chat_id, prompt.message_id)
            await self._api.send_message(prompt.chat_id, self._undo(now))
            return
        answer = prompt.quick_replies[index]
        # The chosen answer stays visible under the question, in place of the buttons.
        with suppress(TelegramError, httpx.HTTPError):
            await self._api.edit_message_text(
                prompt.chat_id, prompt.message_id, f"{prompt.text}\n\n» {answer}"
            )
        await self._run_turn(prompt.chat_id, answer, now)

    # --- commands ------------------------------------------------------------------

    async def _command(self, chat_id: int, text: str, now: datetime) -> None:
        name, _, args = text[1:].partition(" ")
        name = name.split("@", 1)[0].lower()  # "/today@my_bot" in the command menu
        args = args.strip()
        match name:
            case "start" | "help":
                reply = HELP
            case "today" | "tomorrow":
                days = 1 if name == "tomorrow" else 0
                schedule = self._schedule(now.date() + timedelta(days=days), now)
                await self._api.send_message(chat_id, schedule.text, styles=schedule.styles)
                return
            case "rules":
                reply = rules_text(
                    self._repos.event_rules.get_all(), self._repos.event_defaults.list_all()
                )
            case "undo":
                await self._close_prompt()
                reply = self._undo(now)
            case "report":
                reply = self._report(args)
            case "settings":
                reply = self._settings(args)
            case _:
                reply = UNKNOWN_COMMAND
        await self._api.send_message(chat_id, reply)

    def _schedule(self, day: date, now: datetime) -> Reply:
        """The same read as "what do I have today?", without the model."""
        call = parse_tool_call({"tool": "get_schedule", "date": day.isoformat()})
        return render(execute(self._repos, call, now=now), now)

    def _undo(self, now: datetime) -> str:
        with transaction(self._repos.conn):
            undone = undo_last_action(self._repos)
        return undone_text(undone, now.date())

    def _report(self, note: str) -> str:
        turn_id = self._repos.turn_log.latest_turn_id()
        if turn_id is None:
            return NOTHING_TO_REPORT
        self._repos.turn_log.flag(turn_id, note)
        return REPORTED

    def _settings(self, args: str) -> str:
        if not args:
            return self._settings_text()
        key, _, value = args.partition(" ")
        if key not in SETTINGS:
            return f"There is no setting called {key!r}.\n\n" + self._settings_text()
        low, high, unit, _ = SETTINGS[key]
        try:
            number = int(value.strip())
        except ValueError:
            number = None
        if number is None or not low <= number <= high:
            return f"{key} must be a whole number from {low} to {high}."
        self._repos.app_settings.set_int(key, number)
        return f"Done: {key} is now {number}{unit}."

    def _settings_text(self) -> str:
        settings = self._repos.app_settings
        lines = [
            f"- {key}: {settings.get_int(key)}{unit} ({meaning})"
            for key, (_, _, unit, meaning) in SETTINGS.items()
        ]
        return (
            "Settings:\n" + "\n".join(lines) + "\n\nChange one with /settings <name> <value>, "
            "for example /settings repetition_threshold 4."
        )

    def _now(self, timestamp: int | None) -> datetime:
        """The user's local time to the minute, with no time zone, as the turn expects."""
        moment = datetime.fromtimestamp(
            timestamp if timestamp is not None else self._clock(), self._zone
        )
        return moment.replace(tzinfo=None, second=0, microsecond=0)


def _wrote_schedule(execution: Execution | None) -> bool:
    """Whether the turn changed the schedule, so that it can be undone."""
    return isinstance(execution, Executed) and isinstance(
        execution.outcome, EventCreated | EventUpdated | EventRemoved
    )


def _buttons(seq: int, quick_replies: tuple[str, ...], undo: bool) -> list[list[Button]]:
    rows: list[list[Button]] = []
    choices = [(label, f"q:{seq}:{index}") for index, label in enumerate(quick_replies)]
    if all(len(label) <= _SHORT_LABEL for label in quick_replies):
        rows.append(choices)
    else:
        rows.extend([choice] for choice in choices)
    if undo:
        rows.append([(UNDO, f"u:{seq}")])
    return [row for row in rows if row]


def _parse_button(data: str) -> tuple[str, int, int]:
    """`q:<seq>:<index>` (a quick reply) or `u:<seq>` (undo); anything else never matches."""
    kind, _, rest = data.partition(":")
    seq, _, index = rest.partition(":")
    try:
        return kind, int(seq), int(index or 0)
    except ValueError:
        return "", -1, 0
