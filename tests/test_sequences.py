from __future__ import annotations

import numpy as np

from aeroguard.data.rul import add_train_rul
from aeroguard.ml.sequences import (
    apply_feature_scaler,
    build_last_sequences,
    build_sequences,
    fit_feature_scaler,
)


def test_sequences_do_not_cross_engine_boundaries(sensor_frame) -> None:
    labeled = add_train_rul(sensor_frame)
    batch = build_sequences(
        labeled,
        ["cycle", "sensor_1"],
        sequence_length=2,
    )

    assert batch.values.shape == (3, 2, 2)
    assert batch.engine_ids.tolist() == [1, 1, 2]
    assert batch.cycles.tolist() == [2, 3, 2]
    assert batch.values[0, :, 0].tolist() == [1.0, 2.0]
    assert batch.values[-1, :, 0].tolist() == [1.0, 2.0]


def test_last_sequences_return_one_window_per_engine(sensor_frame) -> None:
    labeled = add_train_rul(sensor_frame)
    batch = build_last_sequences(
        labeled,
        ["cycle", "sensor_1"],
        sequence_length=2,
    )

    assert batch.values.shape == (2, 2, 2)
    assert batch.engine_ids.tolist() == [1, 2]
    assert batch.cycles.tolist() == [3, 2]


def test_scaler_is_fit_only_on_supplied_training_rows(sensor_frame) -> None:
    features = ["cycle", "sensor_1"]
    training = sensor_frame.loc[sensor_frame["engine_id"] == 1]
    validation = sensor_frame.loc[sensor_frame["engine_id"] == 2].copy()
    validation["sensor_1"] += 1_000

    scaler = fit_feature_scaler(training, features)
    scaled_validation = apply_feature_scaler(validation, features, scaler)

    assert np.isclose(scaler.mean_[1], training["sensor_1"].mean())
    assert scaled_validation["sensor_1"].mean() > 100


def test_scaled_cycle_feature_does_not_replace_cycle_identity(sensor_frame) -> None:
    labeled = add_train_rul(sensor_frame)
    features = ["cycle", "sensor_1"]
    scaler = fit_feature_scaler(labeled, features)
    scaled = apply_feature_scaler(labeled, features, scaler)

    batch = build_sequences(scaled, features, sequence_length=2)

    assert batch.cycles.tolist() == [2, 3, 2]
    assert not np.array_equal(scaled["cycle"].to_numpy(), labeled["cycle"].to_numpy())
