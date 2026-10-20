import json

import httpx
import pytest

from daily.telegram.api import MAX_TEXT, TelegramApi, TelegramError

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


async def test_without_buttons_no_markup_and_long_texts_are_cut():
    api, calls = recording_api({"ok": True, "result": {"message_id": 5}})

    await api.send_message(42, "x" * (MAX_TEXT + 10))

    [(_, body)] = calls
    assert "reply_markup" not in body
    assert len(body["text"]) == MAX_TEXT


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
