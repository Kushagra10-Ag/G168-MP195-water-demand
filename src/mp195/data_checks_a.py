
"""Data checks A: duplicate rows and sentinel values (owner: Kushagra)."""

import pandas as pd

TRAIN_PATH = "data/raw/train.csv"

SENTINEL_CANDIDATES = [-1, -9, -99, -999, -9999, 99, 999, 9999, 99999]


def count_duplicate_rows(df):
    """Number of EXTRA copies, i.e. the rows you would delete."""
    return int(df.duplicated().sum())


def find_sentinels(df, min_count=3):
    """Look for an impossible number repeated many times in a numeric column."""
    found = []

    for col in df.select_dtypes("number").columns:
        for value in SENTINEL_CANDIDATES:
            count = int((df[col] == value).sum())

            if count < min_count:
                continue

            others = df.loc[df[col] != value, col]

            if len(others) == 0:
                continue

            if (
                (value > 0 and others.max() < value)
                or (value < 0 and others.min() > value)
            ):
                found.append({
                    "column": col,
                    "value": value,
                    "count": count
                })

    return found


def report(df):
    return {
        "duplicate_rows": {
            "count": count_duplicate_rows(df)
        },
        "sentinel_value": find_sentinels(df)
    }


if __name__ == "__main__":
    data = pd.read_csv(TRAIN_PATH)

    print("Rows in file:", len(data))

    result = report(data)

    print("Duplicate extra copies:", result["duplicate_rows"]["count"])
    print("Sentinel candidates found:", result["sentinel_value"])
