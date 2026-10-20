import pytest

from daily.storage.chat_messages_repo import ChatMessagesRepository


@pytest.fixture
def repo(conn):
    return ChatMessagesRepository(conn)


def test_messages_are_kept_per_chat_oldest_first(repo):
    repo.record(42, 7, 1_000)
    repo.record(42, 3, 900)
    repo.record(9, 1, 800)
    repo.record(42, 7, 1_000)  # the same message twice is kept once

    assert repo.list_for(42) == [(3, 900), (7, 1_000)]


def test_forget_drops_only_that_chat(repo):
    repo.record(42, 1, 900)
    repo.record(9, 1, 800)

    repo.forget(42)

    assert repo.list_for(42) == []
    assert repo.list_for(9) == [(1, 800)]
