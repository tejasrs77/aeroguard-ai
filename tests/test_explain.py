from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from aeroguard.ml.explain import calculate_tree_shap, local_shap_table


def test_tree_shap_returns_global_and_local_contributions() -> None:
    frame = pd.DataFrame(
        {
            "sensor_a": [0.0, 1.0, 2.0, 3.0, 4.0],
            "sensor_b": [4.0, 3.0, 2.0, 1.0, 0.0],
        }
    )
    target = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", RandomForestRegressor(n_estimators=10, random_state=4)),
        ]
    ).fit(frame, target)

    global_table, values, base_values = calculate_tree_shap(
        model,
        frame,
        list(frame.columns),
    )
    local_table = local_shap_table(list(frame.columns), frame.iloc[0], values[0])

    assert set(global_table["feature"]) == {"sensor_a", "sensor_b"}
    assert values.shape == frame.shape
    assert base_values.shape == (len(frame),)
    assert set(local_table["direction"]) <= {
        "increases predicted RUL",
        "decreases predicted RUL",
    }
