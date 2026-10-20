"""Everything about language: interpreting a message, executing it and replying.

The model is a parser: it produces a tool call (`schemas`) from the message and the recent
conversation (`prompts`); the code executes it on the services (`execute`) and replies with
fixed messages (`templates`). The turn is a three-step workflow (`turn`).
"""
