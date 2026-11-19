
import pandas as pd

from mp195.data_checks_a import count_duplicate_rows, find_sentinels


def test_duplicate_rows_counts_extra_copies():
    df = pd.DataFrame({
        "x": [1, 1, 1, 2],
        "y": ["a", "a", "a", "b"]
    })

    assert count_duplicate_rows(df) == 2


def test_sentinel_found_and_real_values_ignored():
    df = pd.DataFrame({
        "hours": [5.0, 6.0, 9999, 9999, 9999, 7.0],
        "flats": [99, 120, 300, 150, 99, 200]
    })

    found = find_sentinels(df)

    assert found == [
        {"column": "hours", "value": 9999, "count": 3}
    ]
