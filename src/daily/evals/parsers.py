"""The parsers the eval can use. The runner knows nothing about the model: it needs a `Parser`."""

from collections.abc import Callable

from daily.agent.parser import ParseFailure, Parser, ParseRequest
from daily.agent.schemas import NoActionCall, ToolCall, parse_tool_call
from daily.evals.cases import Case


class OracleParser:
    """Returns the expected tool call for every turn: it validates the runner and the cases
    (100%).

    A turn without `call` gets "no action", as a model would for a message that is not about
    the schedule.
    """

    def __init__(self, case: Case) -> None:
        self._calls = [turn.call for turn in case.turns]
        self._next = 0

    async def parse(self, request: ParseRequest) -> ToolCall | ParseFailure:
        call = self._calls[self._next] if self._next < len(self._calls) else None
        self._next += 1
        return parse_tool_call(call) if call is not None else NoActionCall(tool="none")


ParserFactory = Callable[[Case], Parser]

PARSERS: dict[str, ParserFactory] = {"oracle": OracleParser}
