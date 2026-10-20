"""daily: a scheduling assistant on Telegram, driven by a local language model."""

import os

# Agno telemetry is switched off before the framework is imported: the schedule is
# personal data, and no outgoing traffic leaves unless we chose it.
os.environ.setdefault("AGNO_TELEMETRY", "false")
