
import numpy as np
import pandas as pd
import pytest

from mp195.modeling import (
    FEATURES,
    build_pipeline,
    split_xy,
)


def make_frame(n=60):
    """Create a sample dataset for testing model pipelines."""

    rng = np.random.default_rng(0)

    df = pd.DataFrame({
        "flats_occupied": rng.integers(20, 300, n),
        "temp_c": rng.uniform(10, 40, n),
        "garden_area_m2": rng.uniform(0, 2000, n),
        "tank_level_morning_pct": rng.integers(10, 100, n),
        "leak_complaints": rng.integers(0, 15, n),
        "supply_hours": rng.uniform(1, 12, n),
        "supply_source": rng.choice(
            ["municipal", "borewell", "tanker"],
            n
        ),
        "tower": rng.choice(
            ["Tower A", "Tower B", "Tower C"],
            n
        ),
    })

    df["litres_per_day"] = (
        df["flats_occupied"] * 150
        + df["temp_c"] * 100
    )

    return df


@pytest.mark.parametrize(
    "kind",
    ["ridge", "random_forest", "gradient_boosting"]
)
def test_every_pipeline_fits_and_handles_missing_values(kind):
    """Test that every model pipeline fits and handles missing values."""

    df = make_frame()

    # Introduce a missing value
    df.loc[0, "supply_hours"] = np.nan

    # Split features and target
    x, y = split_xy(df)

    # Build and train the model pipeline
    pipe = build_pipeline(kind).fit(x, y)

    # Check predictions
    predictions = pipe.predict(x)

    assert len(predictions) == len(df)

    # Check feature columns
    assert list(x.columns) == FEATURES


def test_unknown_model_kind_is_rejected():
    """Test that an unknown model type raises a ValueError."""

    with pytest.raises(ValueError):
        build_pipeline("magic")
