import pandas as pd

from mp195.data_checks_b import find_leaking_columns, find_target_typos


def test_leaking_column_is_found():
    train = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6],
        "meter": [10, 20, 31, 40, 52, 60],
        "litres_per_day": [10, 21, 30, 41, 50, 61],
    })
    holdout = pd.DataFrame({"a": [1, 2]})
    result = find_leaking_columns(train, holdout)
    assert result["leaks"][0]["column"] == "meter"


def test_target_typos_counted():
    df = pd.DataFrame({"litres_per_day": [100, 110, 120, 130, 140, 1500, 1600]})
    assert find_target_typos(df)["count"] == 2


def test_available_features_and_unrelated_missing_columns_are_not_leaks():
    train = pd.DataFrame({
        "available": [10, 20, 30, 40],
        "unrelated": [2, 4, 1, 3],
        "notes": ["a", "b", "c", "d"],
        "litres_per_day": [10, 20, 30, 40],
    })
    holdout = pd.DataFrame({"available": [50]})
    result = find_leaking_columns(train, holdout)
    assert result["missing_from_holdout"] == ["unrelated", "notes"]
    assert result["leaks"] == []


def test_regular_targets_are_not_flagged():
    df = pd.DataFrame({"litres_per_day": [100, 110, 120, 130, 140]})
    result = find_target_typos(df)
    assert result["count"] == 0
    assert result["threshold"] is None
