-- turn_log: what happens to a report. A report has no expiry: it stays open until it is
-- fixed. `flag_outcome` is where the fix goes or went: 'code', 'prompt', 'model' (the model's
-- training) or 'no_change' (nothing to fix); it can be set while the report is still open.
-- `flag_handled_at` is when the report was fixed, and closes it. A new report on the same turn
-- opens it again.
ALTER TABLE turn_log ADD COLUMN flag_handled_at TEXT;
ALTER TABLE turn_log ADD COLUMN flag_outcome TEXT
  CHECK (flag_outcome IN ('code', 'prompt', 'model', 'no_change'));
