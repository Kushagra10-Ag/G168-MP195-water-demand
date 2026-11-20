import pandas as pd

from mp195.data_checks_d import find_stuck_sensors


def test_stuck_sensor_dates():
    dates = pd.date_range(
        "2026-01-01",
        periods=30
    ).strftime("%Y-%m-%d")

    df = pd.DataFrame({
        "record_id": [f"R{i}" for i in range(30)],
        "date": dates,
        "temp": [20 + (i % 7) for i in range(30)],
    })

    df.loc[5:20, "temp"] = 33

    result = find_stuck_sensors(df, min_run=10)

    assert result[0]["start_date"] == "2026-01-06"
    assert result[0]["end_date"] == "2026-01-21"