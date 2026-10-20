"""The few methods of the Telegram Bot API the bot uses, over httpx.

The token is part of every URL: nothing here logs a URL or an httpx exception text, and the
service lowers the `httpx` logger to warnings for the same reason.
"""

from collections.abc import Sequence
from typing import Any

import httpx

from daily.agent.templates import Styled

TELEGRAM_URL = "https://api.telegram.org"
# Long polling: Telegram holds `getUpdates` open up to this many seconds when there is
# nothing new; the HTTP timeout must be longer.
POLL_SECONDS = 50
_HTTP_TIMEOUT = POLL_SECONDS + 15
MAX_TEXT = 4096  # longest message Telegram accepts
MAX_DELETE = 100  # most messages `deleteMessages` takes at once

# One inline button: (label, callback data). Telegram allows at most 64 bytes of data.
Button = tuple[str, str]


class TelegramError(Exception):
    """Telegram answered `ok: false`. `code` is the HTTP-like error code (401, 409, ...)."""

    def __init__(self, method: str, code: int | None, description: str) -> None:
        super().__init__(f"{method}: {code} {description}")
        self.method = method
        self.code = code
        self.description = description


class TelegramApi:
    def __init__(
        self,
        token: str,
        *,
        base_url: str = TELEGRAM_URL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=f"{base_url}/bot{token}/", timeout=_HTTP_TIMEOUT, transport=transport
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _call(self, method: str, **params: Any) -> Any:
        body = {key: value for key, value in params.items() if value is not None}
        response = await self._client.post(method, json=body)
        try:
            data = response.json()
        except ValueError:
            raise TelegramError(method, response.status_code, "not a JSON answer") from None
        if not data.get("ok"):
            raise TelegramError(method, data.get("error_code"), data.get("description", ""))
        return data["result"]

    async def get_me(self) -> dict[str, Any]:
        return await self._call("getMe")

    async def get_updates(self, offset: int | None) -> list[dict[str, Any]]:
        return await self._call(
            "getUpdates",
            offset=offset,
            timeout=POLL_SECONDS,
            allowed_updates=["message", "callback_query"],
        )

    async def send_message(
        self,
        chat_id: int,
        text: str,
        buttons: list[list[Button]] | None = None,
        styles: Sequence[Styled] = (),
    ) -> dict[str, Any]:
        text = text[:MAX_TEXT]
        return await self._call(
            "sendMessage",
            chat_id=chat_id,
            text=text,
            entities=_entities(text, styles),
            reply_markup=_keyboard(buttons),
        )

    async def edit_message_text(self, chat_id: int, message_id: int, text: str) -> None:
        """Replaces the text and, with it, removes the buttons."""
        await self._call(
            "editMessageText", chat_id=chat_id, message_id=message_id, text=text[:MAX_TEXT]
        )

    async def remove_buttons(self, chat_id: int, message_id: int) -> None:
        await self._call(
            "editMessageReplyMarkup",
            chat_id=chat_id,
            message_id=message_id,
            reply_markup={"inline_keyboard": []},
        )

    async def delete_messages(self, chat_id: int, message_ids: Sequence[int]) -> None:
        """Deletes the messages, `MAX_DELETE` per call; Telegram skips those it cannot find."""
        for start in range(0, len(message_ids), MAX_DELETE):
            await self._call(
                "deleteMessages",
                chat_id=chat_id,
                message_ids=list(message_ids[start : start + MAX_DELETE]),
            )

    async def answer_callback_query(self, callback_query_id: str, text: str | None = None) -> None:
        await self._call("answerCallbackQuery", callback_query_id=callback_query_id, text=text)

    async def send_typing(self, chat_id: int) -> None:
        await self._call("sendChatAction", chat_id=chat_id, action="typing")

    async def set_commands(self, commands: list[tuple[str, str]]) -> None:
        await self._call(
            "setMyCommands",
            commands=[{"command": name, "description": text} for name, text in commands],
        )


def _keyboard(buttons: list[list[Button]] | None) -> dict[str, Any] | None:
    if not buttons:
        return None
    return {
        "inline_keyboard": [
            [{"text": label, "callback_data": data} for label, data in row] for row in buttons
        ]
    }


def _entities(text: str, styles: Sequence[Styled]) -> list[dict[str, Any]] | None:
    """The styles as message entities, which Telegram measures in UTF-16 code units: no
    markup in the text, so nothing in a title can break the message. A style cut off by the
    length limit is shortened or dropped."""
    entities = []
    for styled in styles:
        end = min(styled.end, len(text))
        if styled.start < end:
            entities.append(
                {
                    "type": styled.style,
                    "offset": _utf16(text[: styled.start]),
                    "length": _utf16(text[styled.start : end]),
                }
            )
    return entities or None


def _utf16(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2
