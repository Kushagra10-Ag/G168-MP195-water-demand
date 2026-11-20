
"""Data checks C: unit mix-up and spelling variants (owner: Ayush)."""

import difflib
import re
import pandas as pd

TRAIN_PATH = "data/raw/train.csv"
LATER_PATH = "data/raw/later.csv"

GROUP_COL = "tower"
TARGET = "litres_per_day"

# Real unit conversions: wrong value = right value x factor
KNOWN_FACTORS = [10, 10.764, 12, 60, 100, 1000]


def pick_factor(bad_values, normal_values, ratio):
    """Choose the exact conversion factor.

    If the normal groups hold whole numbers, the right factor is the
    one that turns the bad group's values back into (almost) whole numbers.

    Otherwise, fall back to the known factor closest to the ratio.
    """

    big = ratio if ratio >= 1 else 1 / ratio

    if (normal_values % 1 == 0).all():

        def whole_error(f):
            x = bad_values / f if ratio >= 1 else bad_values * f
            return float((x - x.round()).abs().mean())

        return min(KNOWN_FACTORS, key=whole_error)

    return min(KNOWN_FACTORS, key=lambda f: abs(f - big))


def find_unit_mixup(
    df,
    group_col=GROUP_COL,
    skip=(TARGET,),
    min_ratio=3.0
):
    """Compare each numeric column's median per group value.

    Each group's median is compared with the median of all group medians.

    Median is used because sentinels and typos would distort the mean.
    If one group is far from the others, its unit may be different.
    """

    results = []

    for col in df.select_dtypes("number").columns:

        if col in skip:
            continue

        medians = df.groupby(group_col)[col].median()
        typical = medians.median()

        for group, med in medians.items():

            if med <= 0 or typical <= 0:
                continue

            ratio = med / typical

            if ratio >= min_ratio or ratio <= 1 / min_ratio:

                bad_values = df.loc[df[group_col] == group, col]
                normal_values = df.loc[df[group_col] != group, col]

                factor = pick_factor(
                    bad_values,
                    normal_values,
                    ratio
                )

                if ratio < 1:
                    factor = round(1 / factor, 6)

                results.append({
                    "column": col,
                    "group_value": group,
                    "ratio": round(float(ratio), 3),
                    "factor": factor
                })

    return results


def _clean(text):
    """Lower-case and remove spaces and punctuation."""
    return re.sub(r"[^a-z]", "", str(text).lower())


def find_spelling_variants(
    df,
    clean_df,
    skip=("record_id", "date")
):
    """Compare each text column with the clean spellings in later.csv."""

    results = []

    for col in df.select_dtypes(exclude="number").columns:

        if col in skip or col not in clean_df.columns:
            continue

        true_levels = sorted(clean_df[col].dropna().unique())

        cleaned_names = {
            _clean(v): v for v in true_levels
        }

        raw_levels = sorted(df[col].dropna().unique())

        bad = [
            v for v in raw_levels
            if v not in true_levels
        ]

        if bad:

            mapping = {}

            for v in raw_levels:

                match = difflib.get_close_matches(
                    _clean(v),
                    list(cleaned_names),
                    n=1,
                    cutoff=0.5
                )

                mapping[v] = (
                    cleaned_names[match[0]]
                    if match else None
                )

            results.append({
                "column": col,
                "bad_spellings": bad,
                "true_levels": len(true_levels),
                "mapping": mapping
            })

    return results


def report(train, later):
    """Generate the complete data quality report."""

    return {
        "unit_mixup": find_unit_mixup(train),
        "spelling_variants": find_spelling_variants(
            train,
            later
        )
    }


if __name__ == "__main__":

    train = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)

    result = report(train, later)

    print("Unit mix-up candidates:")
    print(result["unit_mixup"])

    print("\nSpelling variant columns:")
    print(result["spelling_variants"])

    for col in train.select_dtypes(exclude="number").columns:

        if col not in ("record_id", "date"):

            print(f"\nLevels in {col}:")
            print(train[col].value_counts())
