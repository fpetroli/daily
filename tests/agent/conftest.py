import sqlite3
from datetime import datetime

import pytest

from daily.agent.turn import TurnResult, TurnService
from daily.services.context import Repositories
from scripted import ScriptedParser

# Monday, September 21, 2026, half past noon: the tests' "today".
NOW = datetime(2026, 9, 21, 15, 30)
TODAY = "2026-09-21"
TOMORROW = "2026-09-22"


@pytest.fixture
def repos(conn: sqlite3.Connection) -> Repositories:
    return Repositories.from_connection(conn)


@pytest.fixture
def parser() -> ScriptedParser:
    return ScriptedParser()


@pytest.fixture
def turn(repos: Repositories, parser: ScriptedParser) -> TurnService:
    return TurnService(repos, parser)


async def say(turn: TurnService, message: str, *, now: datetime = NOW) -> TurnResult:
    return await turn.handle(message, now=now)


def add_event(repos: Repositories, title: str, start: str, end: str, **derived: str):
    event_id = repos.schedule.insert_event(title=title, start=start, end=end, **derived)
    return repos.schedule.get_by_id(event_id)
