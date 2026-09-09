from __future__ import annotations

import pandas as pd
import pytest

from aeroguard.ml.features import capped_rul, select_model_features


def test_feature_selection_excludes_engine_identity_and_targets(sensor_frame) -> None:
    labeled = sensor_frame.assign(rul=5, failure_within_30_cycles=1)

    selected, _ = select_model_features(labeled)

    assert "engine_id" not in selected
    assert "rul" not in selected
    assert "failure_within_30_cycles" not in selected
    assert "cycle" in selected


def test_constant_features_are_learned_from_training_partition(sensor_frame) -> None:
    sensor_frame["sensor_1"] = 7.0

    selected, constant = select_model_features(sensor_frame)

    assert "sensor_1" in constant
    assert "sensor_1" not in selected


def test_rul_cap_flattens_only_early_life_values() -> None:
    frame = pd.DataFrame({"rul": [200, 125, 30, 0]})

    assert capped_rul(frame, 125).tolist() == [125.0, 125.0, 30.0, 0.0]


def test_negative_rul_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        capped_rul(pd.DataFrame({"rul": [2, -1]}))
