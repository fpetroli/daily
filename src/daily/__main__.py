"""The service: `python -m daily` runs the Telegram bot.

It opens the database, connects to the model server and polls Telegram until it is stopped.
The configuration comes from the `DAILY_` environment variables (`config.py`).
"""

import asyncio
import contextlib
import logging
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

from daily.agent.turn import TurnService
from daily.config import DB_FILENAME, Settings
from daily.llm.parser import LlamaServerParser
from daily.services.context import Repositories
from daily.storage.db import open_database
from daily.telegram.api import TelegramApi, TelegramError
from daily.telegram.bot import Bot

_REWARM_CHECK_SECONDS = 3600.0
logger = logging.getLogger("daily")


async def _keep_prompt_warm(parser: LlamaServerParser, zone: ZoneInfo | None) -> None:
    """Warms up the prompt prefix again when the user's day changes.

    The calendar is part of the cached prefix, so the first message of a new day would pay
    the cold prompt (about 150 s) without this. It is only a cache optimization: every real
    turn passes the message's own time to the parser anyway.
    """
    warmed_for: date | None = None
    logger.info("Preparing the model's prompt: the first time it can take a few minutes.")
    while True:
        now = datetime.now(zone).replace(tzinfo=None)
        if now.date() != warmed_for:
            if await parser.warm_up(now):
                warmed_for = now.date()
                logger.info("The model is ready.")
            else:
                logger.warning("Prompt warm-up failed: the model does not answer.")
        await asyncio.sleep(_REWARM_CHECK_SECONDS)


async def serve(settings: Settings) -> None:
    if not settings.telegram_bot_token:
        raise SystemExit(
            "DAILY_TELEGRAM_BOT_TOKEN is not set: create a bot with @BotFather and put its "
            "token in the configuration."
        )
    if not settings.telegram_allowed_users:
        logger.warning(
            "DAILY_TELEGRAM_ALLOWED_USERS is empty: the bot only tells each user their id."
        )
    conn = open_database(settings.data_dir / DB_FILENAME)
    repos = Repositories.from_connection(conn)
    parser = LlamaServerParser(
        settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
        max_tokens=settings.llm_max_tokens,
    )
    api = TelegramApi(settings.telegram_bot_token)
    bot = Bot(
        api,
        TurnService(repos, parser),
        repos,
        allowed_users=settings.telegram_allowed_users,
        timezone=settings.timezone,
    )
    zone = ZoneInfo(settings.timezone) if settings.timezone else None
    rewarm = (
        asyncio.ensure_future(_keep_prompt_warm(parser, zone))
        if settings.llm_warmup_enabled
        else None
    )
    try:
        await bot.run()
    finally:
        if rewarm is not None:
            rewarm.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await rewarm
        await api.aclose()
        await parser.aclose()
        conn.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # httpx logs every request URL at INFO, and the bot token is part of Telegram's URLs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(serve(Settings()))
    except TelegramError as error:
        logger.error(
            "Telegram refused the bot token (%s): check DAILY_TELEGRAM_BOT_TOKEN.", error.code
        )
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
