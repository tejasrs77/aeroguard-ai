from pathlib import Path

import pandas as pd

from aeroguard.analysis.statistics import (
    bootstrap_mean_ci,
    build_day1_reports,
    build_engine_summary,
)
from aeroguard.data.rul import add_train_rul


def test_engine_summary_uses_one_row_per_engine(sensor_frame) -> None:
    summary = build_engine_summary(sensor_frame)

    assert summary["observed_cycles"].tolist() == [3, 2]


def test_bootstrap_interval_is_deterministic() -> None:
    first = bootstrap_mean_ci(pd.Series([2, 3, 4]), samples=100, seed=7)
    second = bootstrap_mean_ci(pd.Series([2, 3, 4]), samples=100, seed=7)
    assert first == second


def test_day1_report_writes_tables_and_figures(sensor_frame, tmp_path: Path) -> None:
    labeled = add_train_rul(sensor_frame)
    reports = tmp_path / "reports"
    figures = reports / "figures"

    summary = build_day1_reports(labeled, reports_dir=reports, figures_dir=figures)

    assert summary["training_engines"] == 2
    assert (reports / "day1_summary.json").exists()
    assert (reports / "engine_summary.csv").exists()
    assert len(list(figures.glob("*.png"))) == 4


def test_constant_sensor_detection_uses_unique_values(sensor_frame, tmp_path: Path) -> None:
    sensor_frame["sensor_1"] = 0.1
    labeled = add_train_rul(sensor_frame)

    summary = build_day1_reports(
        labeled,
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "figures",
    )

    assert "sensor_1" in summary["constant_sensors"]
