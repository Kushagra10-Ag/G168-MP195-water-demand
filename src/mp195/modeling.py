
"""Features and model building (owner: Ayush)."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Target variable
TARGET = "litres_per_day"


# Numeric features
NUMERIC_FEATURES = [
    "flats_occupied",
    "temp_c",
    "garden_area_m2",
    "tank_level_morning_pct",
    "leak_complaints",
    "supply_hours",
]


# Categorical features
CATEGORICAL_FEATURES = [
    "supply_source",
    "tower",
]


# All features combined
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42


def split_xy(df):
    """Return (features, target) using only the columns known at prediction time."""
    return df[FEATURES], df[TARGET]


def build_preprocessor():
    """Build preprocessing pipelines for numeric and categorical features."""

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def build_regressor(kind, params=None):
    """Build the selected regression model."""

    params = dict(params or {})

    if kind == "ridge":
        return Ridge(**params)

    if kind == "random_forest":
        return RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params
        )

    if kind == "gradient_boosting":
        return GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            **params
        )

    raise ValueError(f"Unknown model kind: {kind}")


def build_pipeline(kind, params=None):
    """Preprocessing and model in one object, so the API can use it as it is."""

    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", build_regressor(kind, params)),
    ])
