"""Write the holdout predictions file (owner: Kushagra)."""

import os
import pandas as pd

from mp195.modeling import FEATURES


def write_predictions(model, holdout, path="predictions/holdout_predictions.csv"):
    """Predict every holdout row and save exactly two columns: record_id,prediction."""

    out = pd.DataFrame({
        "record_id": holdout["record_id"].to_numpy(),
        "prediction": model.predict(holdout[FEATURES]),
    })

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    out.to_csv(path, index=False)

    return out