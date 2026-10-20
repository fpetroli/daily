"""The repositories of one connection, grouped: this is what the services receive."""

import sqlite3
from dataclasses import dataclass
from typing import Self

from daily.services.resolver import EventDescriptionResolver
from daily.storage.action_log_repo import ActionLogRepository
from daily.storage.defaults_repo import EventDefaultsRepository
from daily.storage.notes_repos import (
    EventPatternsRepository,
    GapPatternsRepository,
    SequencePatternsRepository,
    TriggerDurationPatternsRepository,
)
from daily.storage.rules_repo import EventRulesRepository
from daily.storage.schedule_repo import ScheduleRepository
from daily.storage.settings_repo import AppSettingsRepository
from daily.storage.turn_log_repo import TurnLogRepository


@dataclass(frozen=True, slots=True)
class Repositories:
    conn: sqlite3.Connection
    schedule: ScheduleRepository
    event_rules: EventRulesRepository
    event_defaults: EventDefaultsRepository
    app_settings: AppSettingsRepository
    event_patterns: EventPatternsRepository
    sequence_patterns: SequencePatternsRepository
    gap_patterns: GapPatternsRepository
    trigger_duration_patterns: TriggerDurationPatternsRepository
    action_log: ActionLogRepository
    turn_log: TurnLogRepository
    resolver: EventDescriptionResolver

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection) -> Self:
        schedule = ScheduleRepository(conn)
        return cls(
            conn=conn,
            schedule=schedule,
            event_rules=EventRulesRepository(conn),
            event_defaults=EventDefaultsRepository(conn),
            app_settings=AppSettingsRepository(conn),
            event_patterns=EventPatternsRepository(conn),
            sequence_patterns=SequencePatternsRepository(conn),
            gap_patterns=GapPatternsRepository(conn),
            trigger_duration_patterns=TriggerDurationPatternsRepository(conn),
            action_log=ActionLogRepository(conn),
            turn_log=TurnLogRepository(conn),
            resolver=EventDescriptionResolver(schedule),
        )
