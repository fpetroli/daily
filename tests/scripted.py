"""A scripted parser: the "model" of the tests.

It returns the tool call it was told to return, as the model would, and remembers every
request it received: so tests can also check what the model would see (the history in
particular).
"""

from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import ValidationError

from daily.agent.parser import ParseFailure, ParseRequest
from daily.agent.schemas import ToolCall, parse_tool_call

Scripted = ToolCall | ParseFailure | dict[str, Any] | str


class ScriptedParser:
    """Answers in order with the given tool calls (dicts, JSON or already validated models).

    An item that is not a valid tool call becomes `ParseFailure("invalid_output")`, as would
    happen with a model that gets it wrong. With a function instead of the list it answers
    based on the request.
    """

    def __init__(
        self, script: Iterable[Scripted] | Callable[[ParseRequest], Scripted] = ()
    ) -> None:
        self._queue: deque[Scripted] = deque() if callable(script) else deque(script)
        self._function = script if callable(script) else None
        self.requests: list[ParseRequest] = []

    def then(self, *items: Scripted) -> "ScriptedParser":
        self._queue.extend(items)
        return self

    def answer_with(self, function: Callable[[ParseRequest], Scripted]) -> "ScriptedParser":
        """From now on it answers based on the request (and therefore the history), like a model."""
        self._function = function
        return self

    async def parse(self, request: ParseRequest) -> ToolCall | ParseFailure:
        self.requests.append(request)
        if self._function is not None:
            item = self._function(request)
        elif self._queue:
            item = self._queue.popleft()
        else:
            raise AssertionError(f"No scripted answer for: {request.message!r}")
        return _as_result(item)


def _as_result(item: Scripted) -> ToolCall | ParseFailure:
    if isinstance(item, ParseFailure):
        return item
    if isinstance(item, str | bytes | dict):
        try:
            return parse_tool_call(item)
        except ValidationError as error:
            return ParseFailure("invalid_output", str(error))
    return item
