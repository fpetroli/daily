"""Backend configuration, read from environment variables prefixed with `DAILY_`."""

from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

DB_FILENAME = "daily.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DAILY_", env_file=".env", extra="ignore")

    data_dir: Path = Field(
        default=Path("data"),
        description="Folder of the domain SQLite file (daily.db). "
        "Chat sessions live in memory, not on disk.",
    )
    telegram_bot_token: str | None = Field(
        default=None,
        min_length=20,
        description="Token of the Telegram bot (from @BotFather). The service does not start "
        "without it.",
    )
    telegram_allowed_users: Annotated[frozenset[int], NoDecode] = Field(
        default=frozenset(),
        description="Telegram user ids allowed to use the bot, comma-separated. Anybody else "
        "only gets their own user id back, to add here.",
    )
    timezone: str | None = Field(
        default=None,
        description="IANA time zone of the user (for example Europe/Rome): the time of a "
        "message is its sending time in this zone. Default: the server's.",
    )
    llm_base_url: str = Field(
        default="http://127.0.0.1:8081/v1",
        description="llama-server (OpenAI-compatible API) of the model that picks the tool.",
    )
    llm_timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Maximum wait for the model's answer: it must cover the cold prompt "
        "(about 150 s with the prefix not cached).",
    )
    llm_prompt_style: Literal["classic", "turns"] = Field(
        default="classic",
        description="Prompt shape: 'classic' for Gemma 4 E4B, 'turns' for E2B.",
    )
    llm_max_tokens: int = Field(
        default=160, gt=0, description="Cap on the tokens generated for one tool call."
    )
    llm_warmup_enabled: bool = Field(
        default=True,
        description="Warms up the prompt prefix at startup and when the day changes. "
        "Off in tests, so that no real `llama-server` is called.",
    )

    @field_validator("telegram_allowed_users", mode="before")
    @classmethod
    def _comma_separated(cls, value: object) -> object:
        if isinstance(value, str):
            return frozenset(int(part) for part in value.replace(",", " ").split())
        return value

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                ZoneInfo(value)
            except (ZoneInfoNotFoundError, ValueError):
                raise ValueError(
                    f"unknown time zone {value!r}, expected e.g. Europe/Rome"
                ) from None
        return value
