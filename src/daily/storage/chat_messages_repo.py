"""Access to `chat_messages`: the ids of the messages of a chat, to clear it."""

import sqlite3


class ChatMessagesRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def record(self, chat_id: int, message_id: int, sent_at: int) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO chat_messages (chat_id, message_id, sent_at) VALUES (?, ?, ?)",
            (chat_id, message_id, sent_at),
        )

    def list_for(self, chat_id: int) -> list[tuple[int, int]]:
        """(message_id, sent_at) pairs, oldest first."""
        rows = self._conn.execute(
            "SELECT message_id, sent_at FROM chat_messages WHERE chat_id = ? ORDER BY message_id",
            (chat_id,),
        ).fetchall()
        return [(row["message_id"], row["sent_at"]) for row in rows]

    def forget(self, chat_id: int) -> None:
        self._conn.execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
