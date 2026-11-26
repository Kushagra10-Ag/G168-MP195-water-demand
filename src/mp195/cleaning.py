
"""Cleaning code for train.csv (owner: Kushagra)."""

import pandas as pd

from mp195.data_checks_a import find_sentinels
from mp195.data_checks_b import find_leaking_columns, find_target_typos
from mp195.data_checks_c import find_spelling_variants, find_unit_mixup
from mp195.data_checks_d import find_stuck_sensors

TARGET = "litres_per_day"


def drop_duplicate_rows(df):
    return df.drop_duplicates().reset_index(drop=True)


def normalise_spellings(df, spelling_issues):
    df = df.copy()

    for issue in spelling_issues:
        df[issue["column"]] = (
            df[issue["column"]]
            .map(issue["mapping"])
            .fillna(df[issue["column"]])
        )

    return df


def fix_target_typos(df, threshold, target=TARGET):
    df = df.copy()

    too_big = df[target] > threshold
    df.loc[too_big, target] = df.loc[too_big, target] / 10

    return df


def fix_units(df, unit_issues, group_col="tower"):
    df = df.copy()

    for issue in unit_issues:
        rows = df[group_col] == issue["group_value"]

        df.loc[rows, issue["column"]] = (
            df.loc[rows, issue["column"]] / issue["factor"]
        )

    return df


def blank_sentinels(df, sentinels):
    df = df.copy()

    for item in sentinels:
        df.loc[
            df[item["column"]] == item["value"],
            item["column"]
        ] = float("nan")

    return df


def blank_stuck_runs(df, stuck_sensors):
    df = df.copy()

    for item in stuck_sensors:
        in_run = (
            (df["date"] >= item["start_date"])
            & (df["date"] <= item["end_date"])
        )

        df.loc[in_run, item["column"]] = float("nan")

    return df


def clean_train(train, later, holdout):
    log = {"rows_in": len(train)}

    leaks = find_leaking_columns(train, holdout)["leaks"]
    leak_cols = [item["column"] for item in leaks]

    df = train.drop(columns=leak_cols)
    log["dropped_leak_columns"] = leak_cols

    before = len(df)
    df = drop_duplicate_rows(df)

    log["duplicates_removed"] = before - len(df)

    spelling = find_spelling_variants(df, later)
    df = normalise_spellings(df, spelling)

    log["spelling_columns_fixed"] = [
        item["column"] for item in spelling
    ]

    typos = find_target_typos(df)

    if typos["count"] > 0:
        df = fix_target_typos(df, typos["threshold"])

    log["target_typos_fixed"] = typos["count"]

    units = find_unit_mixup(df)
    df = fix_units(df, units)

    log["unit_fixes"] = units

    sentinels = find_sentinels(df)
    df = blank_sentinels(df, sentinels)

    log["sentinels_blanked"] = sentinels

    stuck = find_stuck_sensors(df)
    df = blank_stuck_runs(df, stuck)

    log["stuck_runs_blanked"] = stuck

    log["rows_out"] = len(df)

    return df, log


def clean_later(later, holdout, target=TARGET):
    keep = [
        c for c in later.columns
        if c in holdout.columns or c == target
    ]

    return later[keep].copy()
