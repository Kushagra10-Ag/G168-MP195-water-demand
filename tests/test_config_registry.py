import hashlib

import mlflow
from mlflow import MlflowClient
from sklearn.linear_model import LinearRegression

from mp195.config import load_params, sha256_of_file
from mp195.registry import export_model, register_champion


def test_load_params_reads_yaml(tmp_path):
    path = tmp_path / "params.yaml"

    path.write_text(
        "data:\n  target: litres_per_day\n",
        encoding="utf-8"
    )

    assert load_params(str(path))["data"]["target"] == "litres_per_day"


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "x.csv"

    path.write_bytes(b"a,b\n1,2\n")

    assert sha256_of_file(str(path)) == hashlib.sha256(
        b"a,b\n1,2\n"
    ).hexdigest()


def test_register_champion_sets_alias_and_export(tmp_path):
    mlflow.set_tracking_uri(
        f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    )

    exp_id = mlflow.create_experiment(
        "t",
        artifact_location=(tmp_path / "art").as_uri()
    )

    model = LinearRegression().fit(
        [[0], [1], [2]],
        [0, 1, 2]
    )

    with mlflow.start_run(experiment_id=exp_id) as run:
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle"
        )

    version = register_champion(run.info.run_id, "demo-model")

    champion = MlflowClient().get_model_version_by_alias(
        "demo-model",
        "champion"
    )

    assert champion.version == version

    path = export_model(
        model,
        str(tmp_path / "out" / "model.joblib")
    )

    assert (
        (tmp_path / "out" / "model.joblib").exists()
        and path.endswith("model.joblib")
    )