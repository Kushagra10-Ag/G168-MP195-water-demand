
import numpy as np
import pandas as pd
import pytest

from mp195.modeling import FEATURES
from mp195.monitoring import mae_by_group, psi, psi_table


def test_psi_is_zero_for_identical_data_and_large_for_a_shifted():
    rng = np.random.default_rng(0)
    old = rng.normal(30, 5, 1000)

    assert psi(old, old) == pytest.approx(0.0, abs=1e-9)
    assert psi(old, old + 10) > 0.25


def test_psi_of_a_constant_column_is_zero():
    assert psi([5] * 50, [5] * 50) == 0.0


def test_psi_table_finds_the_moved_column():
    rng = np.random.default_rng(1)

    old = pd.DataFrame({
        "a": rng.normal(0, 1, 500),
        "b": rng.normal(0, 1, 500),
    })

    new = pd.DataFrame({
        "a": rng.normal(0, 1, 500),
        "b": rng.normal(3, 1, 500),
    })

    table = psi_table(old, new, ["a", "b"])

    assert max(table, key=table.get) == "b"


class FirstColumnModel:
    """Tiny stand-in model: predicts flats_occupied * 100."""

    def predict(self, x):
        return x["flats_occupied"].to_numpy() * 100


def test_mae_by_group_is_computed_per_tower():
    df = pd.DataFrame({
        c: [1.0, 1.0, 1.0, 1.0] for c in FEATURES
    })

    df["supply_source"] = "municipal"
    df["tower"] = ["A", "A", "B", "B"]
    df["litres_per_day"] = [110.0, 90.0, 150.0, 150.0]

    result = mae_by_group(FirstColumnModel(), df)

    assert result == {
        "A": pytest.approx(10.0),
        "B": pytest.approx(50.0),
    }
