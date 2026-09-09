from __future__ import annotations

import numpy as np
import pytest

from aeroguard.ml.metrics import nasa_asymmetric_score, regression_metrics


def test_perfect_predictions_have_zero_error() -> None:
    values = np.array([10.0, 20.0, 30.0])

    metrics = regression_metrics(values, values)

    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["r2"] == 1
    assert metrics["nasa_score"] == 0


def test_nasa_score_penalizes_late_warning_more_than_early_warning() -> None:
    actual = np.array([50.0])
    early_warning = nasa_asymmetric_score(actual, np.array([40.0]))
    late_warning = nasa_asymmetric_score(actual, np.array([60.0]))

    assert late_warning > early_warning


def test_metric_shapes_must_match() -> None:
    with pytest.raises(ValueError, match="same shape"):
        regression_metrics(np.array([1.0]), np.array([1.0, 2.0]))
