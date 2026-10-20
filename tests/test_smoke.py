import os


def test_agno_telemetry_off_by_default():
    import daily  # noqa: F401  (the import sets the variable)

    assert os.environ["AGNO_TELEMETRY"] == "false"
