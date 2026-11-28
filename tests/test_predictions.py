import pandas as pd

from mp195.modeling import FEATURES
from mp195.predictions import write_predictions


class Constant:
    def predict(self, x):
        return [1234.5] * len(x)


def test_prediction_file_has_exactly_two_columns_and_every_row(tmp_path):
    holdout = pd.DataFrame({c: [1.0, 2.0] for c in FEATURES})
    holdout["record_id"] = ["R1", "R2"]

    path = tmp_path / "out" / "preds.csv"

    write_predictions(Constant(), holdout, str(path))

    saved = pd.read_csv(path)

    assert list(saved.columns) == ["record_id", "prediction"]
    assert saved["record_id"].tolist() == ["R1", "R2"]