"""Regression metrics, including NASA's asymmetric RUL score."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def nasa_asymmetric_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Penalize late maintenance predictions more than early predictions."""

    actual_array = np.asarray(actual, dtype="float64")
    predicted_array = np.asarray(predicted, dtype="float64")
    if actual_array.shape != predicted_array.shape:
        raise ValueError("Actual and predicted arrays must have the same shape")

    error = predicted_array - actual_array
    exponent = np.where(error < 0, -error / 13.0, error / 10.0)
    return float(np.expm1(np.clip(exponent, 0, 50)).sum())


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return complementary measures of RUL prediction quality."""

    actual_array = np.asarray(actual, dtype="float64")
    predicted_array = np.asarray(predicted, dtype="float64")
    if actual_array.shape != predicted_array.shape:
        raise ValueError("Actual and predicted arrays must have the same shape")
    if actual_array.size == 0:
        raise ValueError("Metrics require at least one prediction")

    return {
        "mae": float(mean_absolute_error(actual_array, predicted_array)),
        "rmse": float(mean_squared_error(actual_array, predicted_array) ** 0.5),
        "r2": float(r2_score(actual_array, predicted_array)),
        "nasa_score": nasa_asymmetric_score(actual_array, predicted_array),
    }
