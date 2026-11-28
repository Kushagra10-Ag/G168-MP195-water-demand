
"""Drift and per-group error (owner: Tejaswita)."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from mp195.modeling import TARGET, split_xy


def psi(expected, actual, bins=10, eps=1e-4):
    """Population Stability Index of one column.

    Uses quantile bins from the old data to compare old and new distributions.
    Below 0.1: stable; 0.1 to 0.25: some drift; above 0.25: big drift.
    """
    expected = pd.Series(expected).dropna().to_numpy(dtype=float)
    actual = pd.Series(actual).dropna().to_numpy(dtype=float)

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))

    if len(edges) < 2:
        return 0.0

    edges[0], edges[-1] = -np.inf, np.inf

    old_share = np.histogram(expected, bins=edges)[0] / len(expected)
    new_share = np.histogram(actual, bins=edges)[0] / len(actual)

    old_share = np.clip(old_share, eps, None)
    new_share = np.clip(new_share, eps, None)

    return float(
        np.sum((new_share - old_share) * np.log(new_share / old_share))
    )


def psi_table(old_df, new_df, columns, bins=10):
    """Calculate PSI for every requested column."""
    return {col: psi(old_df[col], new_df[col], bins) for col in columns}


def mae_by_group(model, df, group_col="tower"):
    """Calculate the model's MAE separately for each group."""
    result = {}

    for group, part in df.groupby(group_col):
        x, y = split_xy(part)
        result[group] = float(mean_absolute_error(y, model.predict(x)))

    return result


def overall_mae(model, df):
    """Calculate the model's overall MAE."""
    x, y = split_xy(df)
    return float(mean_absolute_error(y, model.predict(x)))
