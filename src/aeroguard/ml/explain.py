"""SHAP explanations for the fitted Day 2 tree champion."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

with warnings.catch_warnings():
    warnings.simplefilter("ignore", PendingDeprecationWarning)
    import shap


def calculate_tree_shap(
    model: Pipeline,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return global importance, per-row SHAP values, and base values."""

    if "regressor" not in model.named_steps:
        raise ValueError("Expected a pipeline with a regressor step")
    transformed = model[:-1].transform(frame[feature_columns])
    regressor = model.named_steps["regressor"]
    explainer = shap.TreeExplainer(regressor)
    explanation = explainer(transformed, check_additivity=False)
    values = np.asarray(explanation.values, dtype="float64")
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]
    if values.shape != transformed.shape:
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; expected {transformed.shape}"
        )

    base_values = np.asarray(explanation.base_values, dtype="float64").reshape(-1)
    if len(base_values) == 1:
        base_values = np.repeat(base_values, len(frame))
    global_importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "mean_absolute_shap": np.abs(values).mean(axis=0),
            }
        )
        .sort_values("mean_absolute_shap", ascending=False)
        .reset_index(drop=True)
    )
    return global_importance, values, base_values


def local_shap_table(
    feature_columns: list[str],
    feature_values: pd.Series,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    """Make one explanation readable without requiring the SHAP UI."""

    table = pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_value": [float(feature_values[column]) for column in feature_columns],
            "shap_contribution_cycles": shap_values,
        }
    )
    table["direction"] = np.where(
        table["shap_contribution_cycles"] >= 0,
        "increases predicted RUL",
        "decreases predicted RUL",
    )
    table["absolute_contribution"] = table["shap_contribution_cycles"].abs()
    return table.sort_values("absolute_contribution", ascending=False).reset_index(drop=True)
