import numpy as np

from aeroguard.data.validate import validate_sensor_frame


def test_valid_sensor_frame_passes(sensor_frame) -> None:
    report = validate_sensor_frame(sensor_frame, "fixture")

    assert report.valid
    assert report.engines == 2
    assert report.rows == 5


def test_duplicate_engine_cycle_fails(sensor_frame) -> None:
    duplicated = sensor_frame.copy()
    duplicated.loc[len(duplicated)] = duplicated.iloc[0]

    report = validate_sensor_frame(duplicated, "fixture")

    assert "unique_engine_cycle" in {issue.rule for issue in report.issues}


def test_gap_in_cycle_sequence_fails(sensor_frame) -> None:
    with_gap = sensor_frame.loc[
        ~((sensor_frame["engine_id"] == 1) & (sensor_frame["cycle"] == 2))
    ]

    report = validate_sensor_frame(with_gap, "fixture")

    assert "consecutive_cycles" in {issue.rule for issue in report.issues}


def test_non_finite_sensor_fails(sensor_frame) -> None:
    invalid = sensor_frame.copy()
    invalid.loc[0, "sensor_4"] = np.inf

    report = validate_sensor_frame(invalid, "fixture")

    assert "finite_values" in {issue.rule for issue in report.issues}

