"""Training examples labelled by construction, for fine-tuning a small parser model.

A generator picks the values first (a title, a day, a time, a duration), writes a message that
says them in one of many ways and builds the expected tool call from the same values: the
label cannot disagree with the sentence. Every example is a full chat in the 'tuned' prompt
shape (instructions, calendar, message) followed by the tool call the model must write.

The eval cases stay out: a message equal to one of them is dropped, and the titles come from
a pool that shares none with the cases, so the eval keeps measuring understanding, not recall.
"""
