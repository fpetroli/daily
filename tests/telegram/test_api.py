import json

import httpx
import pytest

from daily.agent.templates import Styled
from daily.telegram.api import MAX_DELETE, MAX_TEXT, TelegramApi, TelegramError

TOKEN = "123456:ABC-test-token-value"


def recording_api(answer):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=answer)

    return TelegramApi(TOKEN, transport=httpx.MockTransport(handler)), calls


async def test_send_message_with_buttons():
    api, calls = recording_api({"ok": True, "result": {"message_id": 5}})

    sent = await api.send_message(42, "Shall I?", [[("Yes", "q:1:0"), ("No", "q:1:1")]])

    assert sent == {"message_id": 5}
    [(path, body)] = calls
    assert path == f"/bot{TOKEN}/sendMessage"
    assert body == {
        "chat_id": 42,
        "text": "Shall I?",
        "reply_markup": {
            "inline_keyboard": [
                [
                    {"text": "Yes", "callback_data": "q:1:0"},
                    {"text": "No", "callback_data": "q:1:1"},
                ]
            ]
        },
    }


async def test_a_silent_message_has_no_notification():
    api, calls = recording_api({"ok": True, "result": {"message_id": 5}})

    await api.send_message(42, "Today", silent=True)
    await api.send_message(42, "In 5 minutes")

    [(_, silent), (_, loud)] = calls
    assert silent["disable_notification"] is True
    assert "disable_notification" not in loud


async def test_an_edit_keeps_bold_and_struck_through():
    api, calls = recording_api({"ok": True, "result": True})

    await api.edit_message_text(42, 7, "gym · done", [Styled("bold", 0, 3)])

    [(path, body)] = calls
    assert path == f"/bot{TOKEN}/editMessageText"
    assert body == {
        "chat_id": 42,
        "message_id": 7,
        "text": "gym · done",
        "entities": [{"type": "bold", "offset": 0, "length": 3}],
    }


async def test_without_buttons_no_markup_and_long_texts_are_cut():
    api, calls = recording_api({"ok": True, "result": {"message_id": 5}})

    await api.send_message(42, "x" * (MAX_TEXT + 10))

    [(_, body)] = calls
    assert "reply_markup" not in body
    assert "entities" not in body
    assert len(body["text"]) == MAX_TEXT


async def test_styles_become_entities_counted_in_utf16_units():
    api, calls = recording_api({"ok": True, "result": {"message_id": 5}})
    text = "Today 🏃 · run\n07:00"

    await api.send_message(
        42, text, styles=[Styled("bold", 10, 13), Styled("strikethrough", 14, 19)]
    )

    [(_, body)] = calls
    assert body["text"] == text
    # The emoji takes two UTF-16 units: everything after it moves by one.
    assert body["entities"] == [
        {"type": "bold", "offset": 11, "length": 3},
        {"type": "strikethrough", "offset": 15, "length": 5},
    ]


async def test_styles_past_the_length_limit_are_cut_or_dropped():
    api, calls = recording_api({"ok": True, "result": {"message_id": 5}})

    await api.send_message(
        42,
        "x" * (MAX_TEXT + 10),
        styles=[
            Styled("bold", MAX_TEXT - 2, MAX_TEXT + 5),
            Styled("bold", MAX_TEXT + 1, MAX_TEXT + 3),
        ],
    )

    [(_, body)] = calls
    assert body["entities"] == [{"type": "bold", "offset": MAX_TEXT - 2, "length": 2}]


async def test_delete_messages_goes_in_batches():
    api, calls = recording_api({"ok": True, "result": True})

    await api.delete_messages(42, list(range(1, MAX_DELETE + 3)))

    assert [path.rsplit("/", 1)[1] for path, _ in calls] == ["deleteMessages"] * 2
    assert calls[0][1] == {"chat_id": 42, "message_ids": list(range(1, MAX_DELETE + 1))}
    assert calls[1][1]["message_ids"] == [MAX_DELETE + 1, MAX_DELETE + 2]


async def test_polling_asks_only_for_messages_and_buttons():
    api, calls = recording_api({"ok": True, "result": []})

    assert await api.get_updates(8) == []

    [(path, body)] = calls
    assert path.endswith("/getUpdates")
    assert body["offset"] == 8
    assert body["allowed_updates"] == ["message", "callback_query"]


async def test_a_refusal_raises_with_its_code():
    api, _ = recording_api({"ok": False, "error_code": 401, "description": "Unauthorized"})

    with pytest.raises(TelegramError) as raised:
        await api.send_typing(42)

    assert raised.value.code == 401
    assert TOKEN not in str(raised.value)
