"""Access to `preferences.app_settings`: detection thresholds, key-value."""

import sqlite3


class AppSettingsRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_int(self, key: str) -> int:
        row = self._conn.execute(
            "SELECT setting_value FROM app_settings WHERE setting_key = ?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(f"app_settings: key {key!r} not found (missing seed?)")
        return int(row["setting_value"])

    def set_int(self, key: str, value: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO app_settings (setting_key, setting_value) VALUES (?, ?)",
            (key, str(value)),
        )

    def repetition_threshold(self) -> int:
        return self.get_int("repetition_threshold")

    def time_tolerance(self) -> int:
        return self.get_int("time_tolerance")

    def duration_tolerance(self) -> int:
        return self.get_int("duration_tolerance")
