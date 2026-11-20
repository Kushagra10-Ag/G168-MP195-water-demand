"""Data checks D: stuck sensor (owner: Tejaswita)."""

import pandas as pd

TRAIN_PATH = "data/raw/train.csv"
TARGET = "litres_per_day"
SKIP = (TARGET, "meter_litres_next_morning")


def longest_constant_run(series):
    """Return the longest consecutive run of equal values."""
    run_id = (series != series.shift()).cumsum()
    sizes = series.groupby(run_id).size()

    best = sizes.idxmax()
    positions = series.index[run_id == best]

    return (
        int(sizes.loc[best]),
        series.loc[positions[0]],
        positions[0],
        positions[-1],
    )


def find_stuck_sensors(df, min_run=10):
    """Find numeric columns repeating one value for a long run of rows."""

    df = df.sort_values(["date", "record_id"]).reset_index(drop=True)

    found = []

    for col in df.select_dtypes("number").columns:

        if col in SKIP:
            continue

        length, value, start, end = longest_constant_run(df[col])

        if length >= min_run:
            found.append({
                "column": col,
                "value": float(value),
                "rows": length,
                "start_date": str(df.loc[start, "date"]),
                "end_date": str(df.loc[end, "date"]),
            })

    return found


def report(df):
    return {"stuck_sensor": find_stuck_sensors(df)}


if __name__ == "__main__":
    data = pd.read_csv(TRAIN_PATH)
    print("Stuck sensors:", report(data)["stuck_sensor"])