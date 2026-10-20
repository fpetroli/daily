"""Evals on the final state of the schedule: they measure how well a parser understands
messages.

A case is a starting database, one or more messages and the state expected at the end of the
conversation (plus, optionally, the kind of reply of each turn). The runner executes the real
full turn (adapters, templates, workflow, history) with **any** parser and compares what is
left in the schedule: two different but equivalent tool calls (an explicit end or a duration)
give the same state and both count as correct.
"""
