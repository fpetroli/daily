"""The bot against a fake Telegram: the turn, the model and the database are the real ones,
with the scripted parser in place of the model."""

from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from daily.agent.turn import TurnService
from daily.services.context import Repositories
from daily.telegram import bot as bot_module
from daily.telegram.api import TelegramError
from daily.telegram.bot import (
    CLEARED,
    EXPIRED,
    NOT_ALLOWED,
    NOT_DELETED,
    OLDER_KEPT,
    ONLY_TEXT,
    Bot,
)
from scripted import ScriptedParser
from turn_log import read_turn

USER = 42
ZONE = "Europe/Rome"
# Monday, September 21, 2026, 15:30 in Rome.
NOW = datetime(2026, 9, 21, 15, 30)
TODAY = "2026-09-21"
TOMORROW = "2026-09-22"
SENT_AT = int(NOW.replace(tzinfo=ZoneInfo(ZONE)).timestamp())


class FakeApi:
    """Records every call; `send_message` answers with growing message ids."""

    def __init__(self) -> None:
        self.sent: list[tuple[int, str, list | None]] = []
        self.styles: list[tuple] = []
        self.edited: list[tuple[int, int, str]] = []
        self.unbuttoned: list[tuple[int, int]] = []
        self.deleted: list[tuple[int, list[int]]] = []
        self.delete_error: Exception | None = None
        self.answered: list[tuple[str, str | None]] = []
        self.commands: list[tuple[str, str]] = []
        self.updates: list = []
        self.offsets: list[int | None] = []

    async def get_me(self):
        return {"username": "daily_test_bot"}

    async def get_updates(self, offset):
        self.offsets.append(offset)
        item = self.updates.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def send_message(self, chat_id, text, buttons=None, styles=()):
        self.sent.append((chat_id, text, buttons))
        self.styles.append(tuple(styles))
        return {"message_id": 100 + len(self.sent)}

    async def edit_message_text(self, chat_id, message_id, text):
        self.edited.append((chat_id, message_id, text))

    async def remove_buttons(self, chat_id, message_id):
        self.unbuttoned.append((chat_id, message_id))

    async def delete_messages(self, chat_id, message_ids):
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append((chat_id, list(message_ids)))

    async def answer_callback_query(self, callback_query_id, text=None):
        self.answered.append((callback_query_id, text))

    async def send_typing(self, chat_id):
        pass

    async def set_commands(self, commands):
        self.commands = commands

    @property
    def texts(self) -> list[str]:
        return [text for _, text, _ in self.sent]


@pytest.fixture
def repos(conn) -> Repositories:
    return Repositories.from_connection(conn)


@pytest.fixture
def parser() -> ScriptedParser:
    return ScriptedParser()


@pytest.fixture
def api() -> FakeApi:
    return FakeApi()


@pytest.fixture
def bot(api, repos, parser) -> Bot:
    return Bot(
        api,
        TurnService(repos, parser),
        repos,
        allowed_users=frozenset({USER}),
        timezone=ZONE,
        clock=lambda: SENT_AT,
    )


def message(text=None, *, user=USER, chat_type="private", **extra):
    body = {
        "message_id": 1,
        "date": SENT_AT,
        "chat": {"id": user, "type": chat_type},
        "from": {"id": user},
        **extra,
    }
    if text is not None:
        body["text"] = text
    return {"update_id": 1, "message": body}


def press(data, *, user=USER, message_id=101):
    return {
        "update_id": 2,
        "callback_query": {
            "id": "cb",
            "from": {"id": user},
            "data": data,
            "message": {"message_id": message_id, "chat": {"id": user}},
        },
    }


def gym_tomorrow():
    return {
        "tool": "add_event",
        "title": "gym",
        "date": TOMORROW,
        "time": "18:00",
        "duration_minutes": 60,
    }


# --- who is served ---------------------------------------------------------------------


async def test_an_unknown_user_only_gets_their_id(bot, api, parser):
    await bot.handle_update(message("add gym", user=7))

    assert api.texts == [NOT_ALLOWED.format(user_id=7)]
    assert parser.requests == []


async def test_group_chats_are_ignored(bot, api, parser):
    await bot.handle_update(message("add gym", chat_type="group"))

    assert api.sent == []
    assert parser.requests == []


async def test_a_message_without_text_gets_a_hint(bot, api):
    await bot.handle_update(message(sticker={"file_id": "x"}))

    assert api.texts == [ONLY_TEXT]


# --- turns -----------------------------------------------------------------------------


async def test_a_message_is_a_turn_at_its_sending_time_in_the_configured_zone(bot, api, parser):
    parser.then({"tool": "none"})

    await bot.handle_update(message("hi"))

    [request] = parser.requests
    assert (request.message, request.now, request.timezone) == ("hi", NOW, ZONE)
    assert len(api.sent) == 1


async def test_a_write_gets_an_undo_button_that_undoes_it(bot, api, parser, repos):
    parser.then(gym_tomorrow())

    await bot.handle_update(message("gym tomorrow at 6 pm for an hour"))

    [(_, text, buttons)] = api.sent
    assert text == "Done: gym tomorrow from 18:00 to 19:00."
    assert buttons == [[("Undo", "u:1")]]

    await bot.handle_update(press("u:1"))

    assert repos.schedule.events_for_date(TOMORROW) == []
    assert api.unbuttoned == [(USER, 101)]
    assert api.texts[-1] == 'Undone: I deleted "gym" tomorrow from 18:00 to 19:00.'
    assert parser.requests[1:] == []  # the undo never goes through the model


async def test_a_read_has_no_buttons(bot, api, parser):
    parser.then({"tool": "get_schedule"})

    await bot.handle_update(message("what do I have today?"))

    assert not api.sent[0][2]


async def test_a_quick_reply_is_sent_as_the_users_answer(bot, api, parser, repos):
    parser.then(
        {"tool": "add_event_rule", "trigger_title": "shower", "related_title": "gym"},
        {
            "tool": "add_event_rule",
            "trigger_title": "shower",
            "related_title": "gym",
            "position": "after",
        },
    )

    await bot.handle_update(message("the shower goes with the gym"))
    question, buttons = api.texts[0], api.sent[0][2]
    assert buttons == [[("Before", "q:1:0"), ("After", "q:1:1")]]

    await bot.handle_update(press("q:1:1"))

    assert api.edited == [(USER, 101, f"{question}\n\n» After")]
    assert parser.requests[1].message == "After"
    assert parser.requests[1].history[-1].reply == question
    assert len(repos.event_rules.get_all()) == 1


async def test_long_quick_replies_go_one_per_row(bot, api, parser):
    parser.then({"tool": "add_event_rule", "trigger_title": "stretching", "related_title": "gym"})
    await bot.handle_update(message("stretching with the gym"))
    assert all(len(row) == 2 for row in api.sent[0][2])  # "Before"/"After" are short

    buttons = bot_module._buttons(3, ("the meeting at 10:00", "the meeting at 15:00"), False)

    assert buttons == [[("the meeting at 10:00", "q:3:0")], [("the meeting at 15:00", "q:3:1")]]


async def test_a_new_message_takes_the_buttons_off_the_previous_reply(bot, api, parser):
    parser.then(gym_tomorrow(), {"tool": "none"})

    await bot.handle_update(message("gym tomorrow at 6 pm for an hour"))
    await bot.handle_update(message("thanks"))

    assert api.unbuttoned == [(USER, 101)]


async def test_an_old_button_has_expired(bot, api, parser, repos):
    parser.then(gym_tomorrow(), {"tool": "none"})
    await bot.handle_update(message("gym tomorrow at 6 pm for an hour"))
    await bot.handle_update(message("thanks"))

    await bot.handle_update(press("u:1"))

    assert api.answered == [("cb", EXPIRED)]
    assert len(repos.schedule.events_for_date(TOMORROW)) == 1
    assert len(parser.requests) == 2


async def test_a_button_of_somebody_else_does_nothing(bot, api, parser, repos):
    parser.then(gym_tomorrow())
    await bot.handle_update(message("gym tomorrow at 6 pm for an hour"))

    await bot.handle_update(press("u:1", user=7))

    assert len(repos.schedule.events_for_date(TOMORROW)) == 1


async def test_a_failing_update_does_not_stop_the_bot(bot, api, parser, caplog):
    async def broken(*args, **kwargs):
        raise httpx.ConnectError("down")

    api.send_message = broken
    parser.then({"tool": "none"})

    await bot.handle_update(message("hi"))

    assert "Update 1 failed" in caplog.text


# --- commands --------------------------------------------------------------------------


async def test_today_and_tomorrow_read_without_the_model(bot, api, parser, repos):
    repos.schedule.insert_event(title="gym", start=f"{TOMORROW}T18:00", end=f"{TOMORROW}T19:00")

    await bot.handle_update(message("/today"))
    await bot.handle_update(message("/tomorrow@daily_bot"))

    assert api.texts == [
        "Today · Monday, September 21\n\nNothing planned.",
        "Tomorrow · Tuesday, September 22\n\ngym\n18:00 → 19:00 · 1 h",
    ]
    assert [[s.style for s in styles] for styles in api.styles] == [[], ["bold"]]
    repos.schedule.insert_event(title="dinner", start=f"{TODAY}T20:00", end=f"{TODAY}T21:00")
    await bot.handle_update(message("/today"))
    assert api.texts[-1].endswith(
        "──── 21 h ────\n\nTomorrow · Tuesday, September 22\n\ngym\n18:00 → 19:00 · 1 h"
    )
    assert parser.requests == []


async def test_rules_lists_what_was_taught(bot, api, repos):
    repos.event_rules.insert_rule(
        trigger_title="shower", related_title="gym", position="after", offset_minutes=0
    )

    await bot.handle_update(message("/rules"))

    assert api.texts == ["Rules and usual times\n\ngym\nshower right after"]
    assert [s.style for s in api.styles[0]] == ["bold"]


async def test_undo_command(bot, api, parser, repos):
    await bot.handle_update(message("/undo"))
    parser.then(gym_tomorrow())
    await bot.handle_update(message("gym tomorrow at 6 pm for an hour"))
    await bot.handle_update(message("/undo"))

    assert api.texts[0] == "There is nothing to undo."
    assert api.texts[-1].startswith("Undone:")
    assert repos.schedule.events_for_date(TOMORROW) == []
    assert api.unbuttoned == [(USER, 102)]  # the Undo button of the write is gone


async def test_report_flags_the_last_turn(bot, api, parser, repos):
    await bot.handle_update(message("/report"))
    parser.then({"tool": "none"})
    await bot.handle_update(message("hi"))

    await bot.handle_update(message("/report it should have greeted me"))

    assert api.texts[0] == "There is no reply to report yet."
    assert api.texts[-1] == "Thanks: I flagged the last reply."
    turn_id = repos.turn_log.latest_turn_id()
    assert read_turn(repos.conn, turn_id).flag_note == "it should have greeted me"


async def test_settings_are_shown_and_changed(bot, api, repos):
    await bot.handle_update(message("/settings"))
    await bot.handle_update(message("/settings repetition_threshold 4"))
    await bot.handle_update(message("/settings time_tolerance 90"))
    await bot.handle_update(message("/settings colour blue"))

    shown, changed, out_of_range, unknown = api.texts
    assert "- repetition_threshold: 3 (repetitions before a habit is proposed)" in shown
    assert changed == "Done: repetition_threshold is now 4."
    assert out_of_range == "time_tolerance must be a whole number from 0 to 60."
    assert unknown.startswith("There is no setting called 'colour'.")
    assert repos.app_settings.repetition_threshold() == 4
    assert repos.app_settings.time_tolerance() == 15


async def test_start_help_and_unknown_commands(bot, api):
    await bot.handle_update(message("/start"))
    await bot.handle_update(message("/frobnicate"))

    assert "/today - Today's schedule" in api.texts[0]
    assert api.texts[1].startswith("I don't know that command.")


async def test_clear_deletes_the_chat_and_starts_the_conversation_over(bot, api, parser, repos):
    parser.then(gym_tomorrow(), {"tool": "none"})
    await bot.handle_update(message("gym tomorrow at 6 pm for an hour", message_id=1))
    await bot.handle_update(message("/today", message_id=3))

    await bot.handle_update(message("/clear", message_id=5))

    assert api.deleted == [(USER, [1, 3, 5, 101, 102])]
    assert api.texts[-1] == CLEARED
    assert repos.chat_messages.list_for(USER) == [(103, SENT_AT)]  # only the confirmation
    assert len(repos.schedule.events_for_date(TOMORROW)) == 1  # the schedule stays

    await bot.handle_update(press("u:1"))  # the Undo button went with its message
    assert api.answered == [("cb", EXPIRED)]

    await bot.handle_update(message("thanks", message_id=7))
    assert parser.requests[-1].history == ()  # the model no longer reads the old replies


async def test_clear_leaves_what_telegram_no_longer_deletes_and_says_so(bot, api, repos):
    repos.chat_messages.record(USER, 1, SENT_AT - 48 * 3600)

    await bot.handle_update(message("/clear", message_id=2))

    assert api.deleted == [(USER, [2])]
    assert api.texts[-1] == f"{CLEARED}\n\n{OLDER_KEPT}"
    assert repos.chat_messages.list_for(USER) == [(101, SENT_AT)]


async def test_clear_that_telegram_refuses_says_so(bot, api, repos):
    api.delete_error = TelegramError("deleteMessages", 400, "Bad Request")

    await bot.handle_update(message("/clear", message_id=2))

    assert api.texts[-1] == f"{CLEARED}\n\n{NOT_DELETED}"


# --- polling ---------------------------------------------------------------------------


async def test_polling_handles_updates_in_order_and_confirms_them(bot, api, parser, monkeypatch):
    monkeypatch.setattr(bot_module.asyncio, "sleep", _no_wait)
    parser.then({"tool": "none"})
    first = message("hi") | {"update_id": 7}
    api.updates = [
        [first],
        httpx.ReadTimeout("slow"),
        TelegramError("getUpdates", 409, "Conflict"),
        [],
        TelegramError("getUpdates", 401, "Unauthorized"),
    ]

    with pytest.raises(TelegramError):
        await bot.run()

    assert api.offsets == [None, 8, 8, 8, 8]
    assert len(api.sent) == 1
    assert [name for name, _ in api.commands][:2] == ["today", "tomorrow"]


_real_sleep = bot_module.asyncio.sleep


async def _no_wait(seconds):
    await _real_sleep(0)  # still yields, as the typing loop needs
