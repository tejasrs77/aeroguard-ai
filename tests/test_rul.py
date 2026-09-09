import pandas as pd
import pytest

from aeroguard.data.rul import add_test_rul, add_train_rul


def test_train_rul_counts_down_per_engine(sensor_frame) -> None:
    labeled = add_train_rul(sensor_frame, failure_horizon=1)

    engine_one = labeled.loc[labeled["engine_id"] == 1]
    assert engine_one["rul"].tolist() == [2, 1, 0]
    assert engine_one["failure_within_30_cycles"].tolist() == [0, 1, 1]


def test_test_rul_adds_supplied_tail_life(sensor_frame) -> None:
    labeled = add_test_rul(sensor_frame, pd.Series([10, 20]), failure_horizon=15)

    engine_one = labeled.loc[labeled["engine_id"] == 1]
    engine_two = labeled.loc[labeled["engine_id"] == 2]
    assert engine_one["rul"].tolist() == [12, 11, 10]
    assert engine_two["rul"].tolist() == [21, 20]
    assert engine_one["failure_within_30_cycles"].tolist() == [1, 1, 1]
    assert engine_two["failure_within_30_cycles"].tolist() == [0, 0]


def test_test_rul_requires_one_truth_value_per_engine(sensor_frame) -> None:
    with pytest.raises(ValueError, match="one final RUL"):
        add_test_rul(sensor_frame, pd.Series([10]))

