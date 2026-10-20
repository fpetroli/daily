"""The interpretation step: from a message to a tool call.

A `Parser` is all the rest of the agent knows about the model. In tests it is a script,
in production it is the `llama-server` client: the workflow, the adapters and
the templates do not change.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from daily.agent.schemas import ToolCall


@dataclass(frozen=True, slots=True)
class Exchange:
    """A user message and the reply it received: what the user saw."""

    user: str
    reply: str


@dataclass(frozen=True, slots=True)
class ParseRequest:
    """What the parser receives.

    `history` always holds the latest available exchanges (up to 3), in chronological order,
    not only after a pending question. Every question is still worded so that the answer plus
    the last exchange are enough on their own, without relying on the longer history.
    """

    message: str
    now: datetime
    timezone: str | None = None
    history: tuple[Exchange, ...] = ()


@dataclass(frozen=True, slots=True)
class ParseFailure:
    """The model did not produce a usable tool call.

    `unavailable` and `timeout`: the model does not answer; `invalid_output`: it answered but
    the tool call is not valid; `error`: any other unexpected error.
    """

    kind: Literal["unavailable", "timeout", "invalid_output", "error"]
    detail: str = ""


class Parser(Protocol):
    async def parse(self, request: ParseRequest) -> ToolCall | ParseFailure:
        """Does not raise for a model that does not answer or answers badly: it returns
        `ParseFailure`. The workflow catches any unexpected exception anyway."""
        ...
