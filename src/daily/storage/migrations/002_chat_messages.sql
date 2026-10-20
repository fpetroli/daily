-- chat_messages: the messages of a chat channel, both the user's and the bot's, so that the
-- chat can be cleared (`/clear`). Only the ids and the sending time (Unix seconds, as the
-- channel gives it): the text of a turn is in turn_log. The rows of a chat are deleted when
-- the chat is cleared.
CREATE TABLE chat_messages (
  chat_id INTEGER NOT NULL,
  message_id INTEGER NOT NULL,
  sent_at INTEGER NOT NULL,
  PRIMARY KEY (chat_id, message_id)
);
