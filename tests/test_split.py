from __future__ import annotations

import pandas as pd

from aeroguard.ml.split import last_observation_per_engine, split_by_engine


def test_grouped_split_has_no_engine_leakage(sensor_frame) -> None:
    training, validation = split_by_engine(
        sensor_frame,
        validation_size=0.5,
        random_state=7,
    )

    train_engines = set(training["engine_id"])
    validation_engines = set(validation["engine_id"])
    assert train_engines.isdisjoint(validation_engines)
    assert train_engines | validation_engines == {1, 2}
    assert len(training) + len(validation) == len(sensor_frame)


def test_grouped_split_is_reproducible(sensor_frame) -> None:
    first = split_by_engine(sensor_frame, validation_size=0.5, random_state=11)
    second = split_by_engine(sensor_frame, validation_size=0.5, random_state=11)

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_last_observation_returns_highest_cycle(sensor_frame) -> None:
    endpoints = last_observation_per_engine(sensor_frame)

    assert endpoints[["engine_id", "cycle"]].values.tolist() == [[1, 3], [2, 2]]
