"""Create reproducible Day 1 statistics and static visualizations."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[3] / ".cache" / "matplotlib"),
)

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aeroguard.config import FIGURES_DIR, REPORTS_DIR, SENSOR_COLUMNS


def bootstrap_mean_ci(
    values: pd.Series,
    *,
    confidence: float = 0.95,
    samples: int = 2_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimate a confidence interval for the mean using resampling."""

    data = values.dropna().to_numpy(dtype=float)
    if not len(data):
        raise ValueError("Cannot bootstrap an empty series")
    generator = np.random.default_rng(seed)
    means = generator.choice(data, size=(samples, len(data)), replace=True).mean(axis=1)
    tail = (1 - confidence) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def build_engine_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per engine, the correct unit for lifecycle statistics."""

    return (
        frame.groupby("engine_id", as_index=False)
        .agg(observed_cycles=("cycle", "max"), rows=("cycle", "size"))
        .sort_values("engine_id")
    )


def build_sensor_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize scale, spread, and missingness for every sensor."""

    records = []
    for sensor in SENSOR_COLUMNS:
        series = frame[sensor]
        records.append(
            {
                "sensor": sensor,
                "mean": float(series.mean()),
                "standard_deviation": float(series.std()),
                "minimum": float(series.min()),
                "maximum": float(series.max()),
                "unique_values": int(series.nunique()),
                "missing_values": int(series.isna().sum()),
            }
        )
    return pd.DataFrame(records).sort_values("sensor")


def _sensors_ranked_by_rul(frame: pd.DataFrame) -> list[str]:
    """Rank non-constant sensors by absolute Pearson correlation with RUL."""

    non_constant = [
        sensor for sensor in SENSOR_COLUMNS if frame[sensor].nunique(dropna=True) > 1
    ]
    return (
        frame[non_constant]
        .corrwith(frame["rul"])
        .abs()
        .sort_values(ascending=False)
        .index.tolist()
    )


def _save_lifecycle_histogram(engine_summary: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(engine_summary["observed_cycles"], bins=15, color="#13b8a6", edgecolor="white")
    axis.set(title="FD001 engine lifecycle distribution", xlabel="Cycles until failure", ylabel="Number of engines")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_rul_histogram(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(frame["rul"], bins=25, color="#2563eb", edgecolor="white")
    axis.axvline(30, color="#dc2626", linestyle="--", label="30-cycle risk threshold")
    axis.set(title="Training-row Remaining Useful Life", xlabel="Remaining cycles", ylabel="Sensor snapshots")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_sensor_trends(frame: pd.DataFrame, path: Path) -> list[str]:
    variable_sensors = _sensors_ranked_by_rul(frame)[:3]
    longest_engine = int(frame.groupby("engine_id")["cycle"].max().idxmax())
    engine = frame.loc[frame["engine_id"] == longest_engine].sort_values("cycle")

    figure, axes = plt.subplots(len(variable_sensors), 1, figsize=(9, 7), sharex=True)
    for axis, sensor in zip(np.atleast_1d(axes), variable_sensors, strict=True):
        standardized = (engine[sensor] - engine[sensor].mean()) / engine[sensor].std()
        axis.plot(engine["cycle"], standardized, color="#0f766e", linewidth=1.25)
        axis.set_ylabel(f"{sensor}\n(z-score)")
        axis.grid(alpha=0.2)
    axes[0].set_title(f"Top RUL-associated sensor trends - engine {longest_engine}")
    axes[-1].set_xlabel("Operating cycle")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return variable_sensors


def _save_correlation(frame: pd.DataFrame, path: Path) -> list[str]:
    selected = _sensors_ranked_by_rul(frame)[:10]
    correlation = frame[selected].corr()
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(selected)), selected, rotation=45, ha="right")
    axis.set_yticks(range(len(selected)), selected)
    axis.set_title("Correlation among the 10 sensors most associated with RUL")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return selected


def build_day1_reports(
    train_labeled: pd.DataFrame,
    *,
    reports_dir: Path = REPORTS_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> dict[str, object]:
    """Write Day 1 tables, figures, and a machine-readable summary."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    engine_summary = build_engine_summary(train_labeled)
    sensor_summary = build_sensor_summary(train_labeled)
    engine_summary.to_csv(reports_dir / "engine_summary.csv", index=False)
    sensor_summary.to_csv(reports_dir / "sensor_summary.csv", index=False)

    _save_lifecycle_histogram(engine_summary, figures_dir / "engine_lifecycle_distribution.png")
    _save_rul_histogram(train_labeled, figures_dir / "rul_distribution.png")
    trend_sensors = _save_sensor_trends(train_labeled, figures_dir / "sensor_degradation.png")
    correlation_sensors = _save_correlation(train_labeled, figures_dir / "sensor_correlation.png")

    lower, upper = bootstrap_mean_ci(engine_summary["observed_cycles"])
    constant_sensors = sensor_summary.loc[
        sensor_summary["unique_values"] <= 1, "sensor"
    ].tolist()
    summary: dict[str, object] = {
        "training_rows": int(len(train_labeled)),
        "training_engines": int(train_labeled["engine_id"].nunique()),
        "minimum_lifecycle_cycles": int(engine_summary["observed_cycles"].min()),
        "maximum_lifecycle_cycles": int(engine_summary["observed_cycles"].max()),
        "mean_lifecycle_cycles": float(engine_summary["observed_cycles"].mean()),
        "median_lifecycle_cycles": float(engine_summary["observed_cycles"].median()),
        "mean_lifecycle_95_percent_bootstrap_ci": [lower, upper],
        "near_failure_rows": int(train_labeled["failure_within_30_cycles"].sum()),
        "constant_sensors": constant_sensors,
        "trend_plot_sensors": trend_sensors,
        "correlation_plot_sensors": correlation_sensors,
    }
    (reports_dir / "day1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
