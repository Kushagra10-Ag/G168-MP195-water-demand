"""MLflow registry helpers (owner: Tejaswita)."""

import os

import joblib
import mlflow
from mlflow import MlflowClient


def register_champion(
    run_id,
    model_name,
    alias="champion",
    artifact_path="model"
):
    """Register the model of a run as a new version and point the alias to it."""

    version = mlflow.register_model(
        f"runs:/{run_id}/{artifact_path}",
        model_name
    )

    MlflowClient().set_registered_model_alias(
        model_name,
        alias,
        version.version
    )

    return version.version


def load_champion(model_name, alias="champion"):
    """Load the model that the alias points at."""

    return mlflow.sklearn.load_model(
        f"models:/{model_name}@{alias}"
    )


def export_model(model, path="models/model.joblib"):
    """Save the model as a file for the API."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    joblib.dump(model, path)

    return path