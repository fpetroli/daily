"""The Telegram bot: a message becomes a turn, a button becomes the user's answer, and the
commands read or change the schedule without the model. A button whose reply says exactly
what it does (Yes and No of a habit proposal, the rule to remove) skips the model too.

The bot asks Telegram for new messages (long polling): the service opens no port. Only the
user ids in the allowlist are served, and only in private chats: the schedule is personal.
The time of a turn is the sending time of the message in the configured time zone.

The bot keeps the ids of the messages of the chat, the user's and its own, so that `/clear`
can delete those Telegram still allows (the last 48 hours) and start the conversation over.

The bot writes first only around an event: `reminder_minutes` before it starts it sends the
event alone, with a notification (0 turns this off); when it starts it clears the chat as
`/clear` does, without the confirmation, and sends the day's schedule as `/today`, where the
events already over are struck through, silently. The check runs at the start of every
minute against the schedule, so a moved or deleted event leaves no stale message; a moment
that passed while the bot was off, or before the event was written, gives no message.

Updates are handled one at a time, in order, and a reminder waits for the update in progress:
one user, one model, and a turn already serializes on its own lock. Only the buttons of the
last reply act: the model reads the answer against the recent history, so an answer to an
older question would be misread.
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
from daily.agent.results import Choice, Executed, Execution
from daily.agent.schemas import parse_tool_call
from daily.agent.templates import (
    Reply,
    Styled,
    reminder_reply,
    render,
    rules_reply,
    undone_text,
)
from daily.agent.turn import TurnResult, TurnService
from daily.domain.models import ScheduleEvent
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
    ("clear", "Clear the chat and start over"),
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
CLEARED = "Chat cleared. Your schedule and rules are untouched."
OLDER_KEPT = (
    "Telegram lets me delete only the last 48 hours: clear older messages from the chat menu."
)
NOT_DELETED = "I could not delete some messages: clear them from the chat menu."

# key: (minimum, maximum, unit, meaning)
SETTINGS = {
    "repetition_threshold": (1, 10, "", "repetitions before a habit is proposed"),
    "time_tolerance": (0, 60, " minutes", "how far two times can be and still match"),
    "duration_tolerance": (0, 60, " minutes", "how far two durations can be and still match"),
    "reminder_minutes": (
        0,
        120,
        " minutes",
        "how long before an event I tell you it is starting; 0 is off",
    ),
}

_TYPING_SECONDS = 4.0  # Telegram shows "typing" for about 5 s
_RETRY_SECONDS = (1.0, 60.0)  # first and longest wait after a failed `getUpdates`
_REMIND_DELAY = 1.0  # seconds into each minute when the reminders are checked
_SHORT_LABEL = 12  # quick replies up to this length share a row
# Telegram deletes only messages younger than 48 hours: one minute of margin.
_DELETABLE_SECONDS = 48 * 3600 - 60


class Api(Protocol):
    async def get_me(self) -> dict[str, Any]: ...
    async def get_updates(self, offset: int | None) -> list[dict[str, Any]]: ...
    async def send_message(
        self,
        chat_id: int,
        text: str,
        buttons: list[list[Button]] | None = None,
        styles: Sequence[Styled] = (),
        silent: bool = False,
    ) -> dict[str, Any]: ...
    async def edit_message_text(self, chat_id: int, message_id: int, text: str) -> None: ...
    async def remove_buttons(self, chat_id: int, message_id: int) -> None: ...
    async def delete_messages(self, chat_id: int, message_ids: Sequence[int]) -> None: ...
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
    choices: tuple[Choice | None, ...] = ()


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
        self._lock = asyncio.Lock()  # an update and a reminder never interleave
        self._reminded_until: datetime | None = None

    async def run(self) -> None:
        """Polls Telegram forever. Raises only when the token is refused."""
        me = await self._api.get_me()
        logger.info("Connected to Telegram as @%s: waiting for messages.", me.get("username"))
        try:
            await self._api.set_commands(COMMANDS)
        except (TelegramError, httpx.HTTPError):
            logger.warning("Could not register the command menu")
        reminders = asyncio.ensure_future(self._remind_forever())
        try:
            await self._poll()
        finally:
            reminders.cancel()
            with suppress(asyncio.CancelledError):
                await reminders

    async def _poll(self) -> None:
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
            async with self._lock:
                if (message := update.get("message")) is not None:
                    await self._on_message(message)
                elif (query := update.get("callback_query")) is not None:
                    await self._on_button(query)
        except Exception:
            logger.exception("Update %s failed", update.get("update_id"))

    # --- reminders -----------------------------------------------------------------

    async def _remind_forever(self) -> None:
        while True:
            await self.remind()
            await asyncio.sleep(60 - self._clock() % 60 + _REMIND_DELAY)

    async def remind(self) -> None:
        """One check of what fell since the previous one: the events that started clear each
        chat and get the day, silently; then the events starting in `reminder_minutes` are
        announced, so that the clearing does not take them away. The first check only starts
        counting. Never raises."""
        now = self._now(None)
        since, self._reminded_until = self._reminded_until, now
        if since is None or now <= since:
            return
        try:
            async with self._lock:
                started = _starting(self._repos, since, now)
                for day in sorted({date.fromisoformat(event.start[:10]) for event in started}):
                    for chat_id in sorted(self._allowed):  # private chat id = user id
                        await self._start_day(chat_id, day, now)
                minutes = self._repos.app_settings.reminder_minutes()
                ahead = timedelta(minutes=minutes)
                # a late check never announces an event that already started
                soon = _starting(self._repos, max(since + ahead, now), now + ahead)
                if minutes and soon:
                    left = datetime.fromisoformat(soon[0].start) - now  # N, unless late
                    reply = reminder_reply(soon, int(left.total_seconds()) // 60)
                    for chat_id in sorted(self._allowed):
                        await self._remind_chat(chat_id, reply)
        except Exception:
            logger.exception("The reminder check failed")

    async def _start_day(self, chat_id: int, day: date, now: datetime) -> None:
        try:
            await self._wipe(chat_id)
            reply = self._schedule(day, now)
            await self._send(chat_id, reply.text, styles=reply.styles, silent=True)
        except (TelegramError, httpx.HTTPError):
            logger.warning("Could not send the day to chat %s", chat_id)

    async def _remind_chat(self, chat_id: int, reply: Reply) -> None:
        try:
            await self._send(chat_id, reply.text, styles=reply.styles)
        except (TelegramError, httpx.HTTPError):
            logger.warning("Could not send the reminder to chat %s", chat_id)

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
        self._repos.chat_messages.record(
            chat_id, message["message_id"], message.get("date") or int(self._clock())
        )
        text = (message.get("text") or "").strip()
        if not text:
            await self._send(chat_id, ONLY_TEXT)
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
        await self._reply(chat_id, result)

    async def _reply(self, chat_id: int, result: TurnResult) -> None:
        """Sends the reply of a turn with its buttons: they are the only ones that act."""
        await self._close_prompt()
        self._seq += 1
        reply = result.reply
        buttons = _buttons(self._seq, reply.quick_replies, _wrote_schedule(result.execution))
        sent = await self._send(chat_id, reply.text, buttons, reply.styles)
        if buttons:
            self._prompt = _Prompt(
                self._seq,
                chat_id,
                sent["message_id"],
                reply.text,
                reply.quick_replies,
                reply.choices,
            )

    async def _send(
        self,
        chat_id: int,
        text: str,
        buttons: list[list[Button]] | None = None,
        styles: Sequence[Styled] = (),
        silent: bool = False,
    ) -> dict[str, Any]:
        """Sends a message and keeps its id, so that `/clear` can delete it."""
        sent = await self._api.send_message(chat_id, text, buttons, styles, silent=silent)
        self._repos.chat_messages.record(chat_id, sent["message_id"], int(self._clock()))
        return sent

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
            await self._send(prompt.chat_id, self._undo(now))
            return
        answer = prompt.quick_replies[index]
        # The chosen answer stays visible under the question, in place of the buttons.
        with suppress(TelegramError, httpx.HTTPError):
            await self._api.edit_message_text(
                prompt.chat_id, prompt.message_id, f"{prompt.text}\n\n» {answer}"
            )
        choice = prompt.choices[index] if index < len(prompt.choices) else None
        if choice is None:
            await self._run_turn(prompt.chat_id, answer, now)
        else:  # the button says exactly what to do: no model in between
            await self._reply(prompt.chat_id, await self._turn.choose(choice, answer, now=now))

    # --- commands ------------------------------------------------------------------

    async def _command(self, chat_id: int, text: str, now: datetime) -> None:
        name, _, args = text[1:].partition(" ")
        name = name.split("@", 1)[0].lower()  # "/today@my_bot" in the command menu
        args = args.strip()
        reply: str | Reply
        match name:
            case "start" | "help":
                reply = HELP
            case "today" | "tomorrow":
                days = 1 if name == "tomorrow" else 0
                reply = self._schedule(now.date() + timedelta(days=days), now)
            case "rules":
                reply = rules_reply(
                    self._repos.event_rules.get_all(), self._repos.event_defaults.list_all()
                )
            case "undo":
                await self._close_prompt()
                reply = self._undo(now)
            case "report":
                reply = self._report(args)
            case "settings":
                reply = self._settings(args)
            case "clear":
                reply = await self._clear(chat_id)
            case _:
                reply = UNKNOWN_COMMAND
        if isinstance(reply, str):
            reply = Reply(reply)
        await self._send(chat_id, reply.text, styles=reply.styles)

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

    async def _clear(self, chat_id: int) -> str:
        note = await self._wipe(chat_id)
        return f"{CLEARED}\n\n{note}" if note else CLEARED

    async def _wipe(self, chat_id: int) -> str:
        """Deletes the messages Telegram still lets the bot delete and starts the conversation
        over: the model no longer reads replies the user cannot see. Returns what could not be
        deleted, if anything."""
        self._prompt = None  # its message goes with the others
        await self._turn.forget()
        oldest = int(self._clock()) - _DELETABLE_SECONDS
        messages = self._repos.chat_messages.list_for(chat_id)
        recent = [message_id for message_id, sent_at in messages if sent_at > oldest]
        note = OLDER_KEPT if len(recent) < len(messages) else ""
        try:
            if recent:
                await self._api.delete_messages(chat_id, recent)
        except (TelegramError, httpx.HTTPError):
            logger.warning("Could not delete the chat messages")
            note = NOT_DELETED
        self._repos.chat_messages.forget(chat_id)
        return note

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


def _starting(repos: Repositories, after: datetime, by: datetime) -> list[ScheduleEvent]:
    """The events that start after `after` and by `by`, in order."""
    first, last = after.isoformat(timespec="minutes"), by.isoformat(timespec="minutes")
    days = sorted({after.date(), by.date()})
    return [
        event
        for day in days
        for event in repos.schedule.events_for_date(day.isoformat())
        if first < event.start <= last
    ]


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
