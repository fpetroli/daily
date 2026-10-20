-- The schema. Later changes go in new numbered files (002_..., 003_...), applied in order
-- on startup.
-- `preferences` and `notes` are logical domains, not nested tables: here all the physical
-- tables are at the same level.

-- schedule: events by date. The index on `start` also covers per-day queries (range scan
-- on a sortable ISO string), with no dedicated column.
CREATE TABLE schedule (
  event_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  start TEXT NOT NULL,
  end TEXT NOT NULL,
  derived_from_event_id TEXT,
  derived_from_rule_id TEXT
);
CREATE INDEX idx_schedule_start ON schedule(start);

-- preferences.event_rules: relations between two events.
CREATE TABLE event_rules (
  rule_id TEXT PRIMARY KEY,
  trigger_title TEXT NOT NULL COLLATE NOCASE,
  related_title TEXT NOT NULL COLLATE NOCASE,
  position TEXT NOT NULL,
  offset_minutes INTEGER,
  gap_minutes INTEGER
);
CREATE INDEX idx_event_rules_trigger ON event_rules(trigger_title);
CREATE INDEX idx_event_rules_related ON event_rules(related_title);

-- preferences.event_defaults: usual time and duration of a single event.
CREATE TABLE event_defaults (
  title TEXT PRIMARY KEY COLLATE NOCASE,
  typical_start TEXT,
  typical_duration_minutes INTEGER
);

-- preferences.app_settings: key-value, detection thresholds only.
CREATE TABLE app_settings (
  setting_key TEXT PRIMARY KEY,
  setting_value TEXT NOT NULL
);

-- notes.*: 4 tables with the same mechanism (counter + threshold + 2-week window +
-- state).
CREATE TABLE event_patterns (
  note_id TEXT PRIMARY KEY,
  title TEXT NOT NULL COLLATE NOCASE,
  typical_start TEXT,
  typical_duration_minutes INTEGER,
  occurrence_count INTEGER NOT NULL,
  first_occurrence_at TEXT NOT NULL,
  status TEXT NOT NULL
);
CREATE INDEX idx_event_patterns_title ON event_patterns(title);

CREATE TABLE trigger_duration_patterns (
  note_id TEXT PRIMARY KEY,
  title TEXT NOT NULL COLLATE NOCASE,
  typical_duration_minutes INTEGER,
  occurrence_count INTEGER NOT NULL,
  first_occurrence_at TEXT NOT NULL,
  status TEXT NOT NULL
);
CREATE INDEX idx_trigger_duration_patterns_title ON trigger_duration_patterns(title);

CREATE TABLE sequence_patterns (
  note_id TEXT PRIMARY KEY,
  trigger_title TEXT NOT NULL COLLATE NOCASE,
  related_title TEXT NOT NULL COLLATE NOCASE,
  position TEXT NOT NULL,
  offset_minutes INTEGER NOT NULL,
  occurrence_count INTEGER NOT NULL,
  first_occurrence_at TEXT NOT NULL,
  status TEXT NOT NULL
);
CREATE INDEX idx_sequence_patterns_pair ON sequence_patterns(trigger_title, related_title);

CREATE TABLE gap_patterns (
  note_id TEXT PRIMARY KEY,
  related_title TEXT NOT NULL COLLATE NOCASE,
  position TEXT NOT NULL,
  gap_minutes INTEGER NOT NULL,
  occurrence_count INTEGER NOT NULL,
  first_occurrence_at TEXT NOT NULL,
  status TEXT NOT NULL
);
CREATE INDEX idx_gap_patterns_anchor ON gap_patterns(related_title, position);

-- action_log: the last write performed by a turn, for undo.
-- Only real writes (create/move/delete an event) leave a row; reads and "no action" do not
-- touch it. Only one row at a time matters: undo always acts on the most recent one, then
-- the row is deleted (a single level, not a stack).
CREATE TABLE action_log (
  log_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  kind TEXT NOT NULL,
  summary TEXT NOT NULL,
  undo_data TEXT NOT NULL
);
CREATE INDEX idx_action_log_created_at ON action_log(created_at);

-- turn_log: every chat turn, whatever the outcome: raw material to build eval cases from
-- real use. `tool`/`call_json` are set only if the parser produced a valid tool call; `kind`
-- is the same taxonomy as `daily.agent.results.kind_of`. `duration_ms` is the time from the
-- start of processing to the ready reply. `flag_note` stays NULL until the user flags a
-- problem on that turn (`/report`); `flag_context_json` is set with it: the last 3 turns at
-- most (the window the model sees) up to the flagged one included.
CREATE TABLE turn_log (
  turn_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  message TEXT NOT NULL,
  tool TEXT,
  call_json TEXT,
  kind TEXT NOT NULL,
  reply_text TEXT NOT NULL,
  flag_note TEXT,
  duration_ms INTEGER,
  flag_context_json TEXT
);
CREATE INDEX idx_turn_log_created_at ON turn_log(created_at);

-- Habit detection thresholds at their default values.
INSERT INTO app_settings (setting_key, setting_value) VALUES
  ('repetition_threshold', '3'),
  ('time_tolerance', '15'),
  ('duration_tolerance', '10');
