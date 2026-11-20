"""Data checks B: leaking column and target typos (owner: Vaibhav)."""

import pandas as pd

TRAIN_PATH = "data/raw/train.csv"
HOLDOUT_PATH = "data/raw/holdout_features.csv"
TARGET = "litres_per_day"


def find_leaking_columns(train, holdout, target=TARGET, min_spearman=0.95):
    """Find numeric columns absent at prediction time that track the target.

    Spearman rank correlation is used because a few typo targets would
    distort ordinary correlation.
    """
    missing = [c for c in train.columns if c not in holdout.columns and c != target]
    leaks = []
    for col in missing:
        if pd.api.types.is_numeric_dtype(train[col]):
            rho = float(train[col].corr(train[target], method="spearman"))
            if rho >= min_spearman:
                leaks.append({"column": col, "spearman_with_target": round(rho, 4)})
    return {"missing_from_holdout": missing, "leaks": leaks}


def find_target_typos(df, target=TARGET, min_gap_ratio=2.0):
    """Flag high targets above the largest multiplicative gap.

    Extra-zero typos form a small group far above ordinary targets.
    A gap is evidence of a suspected typo, not proof of its correct value.
    """
    values = df[target].sort_values().reset_index(drop=True)
    ratios = values.shift(-1) / values
    gap_at = int(ratios.iloc[:-1].idxmax())
    gap_ratio = float(ratios.iloc[gap_at])
    if gap_ratio < min_gap_ratio:
        return {"count": 0, "gap_ratio": gap_ratio, "threshold": None}
    threshold = float(values.iloc[gap_at])
    count = int((df[target] > threshold).sum())
    return {"count": count, "gap_ratio": round(gap_ratio, 2), "threshold": threshold}


def report(train, holdout):
    return {
        "leaking_column": find_leaking_columns(train, holdout),
        "target_typos": find_target_typos(train),
    }


if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)
    result = report(train, holdout)
    print("Columns missing from holdout:", result["leaking_column"]["missing_from_holdout"])
    print("Leaking columns:", result["leaking_column"]["leaks"])
    print("Target typos:", result["target_typos"])
