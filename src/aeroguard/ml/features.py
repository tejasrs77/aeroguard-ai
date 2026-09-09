"""Feature and target preparation for tabular RUL models."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from aeroguard.config import CMAPSS_COLUMNS, DEFAULT_RUL_CAP


TARGET_COLUMNS = ("rul", "failure_within_30_cycles")


def candidate_feature_columns() -> list[str]:
    """Return source measurements that are safe to present to a model."""

    return [column for column in CMAPSS_COLUMNS if column != "engine_id"]


def find_constant_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> list[str]:
    """Find features with only one observed value in the training partition."""

    return [column for column in columns if frame[column].nunique(dropna=False) <= 1]


def select_model_features(
    training_frame: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Choose useful columns using training data only to avoid leakage."""

    candidates = candidate_feature_columns()
    missing = sorted(set(candidates) - set(training_frame.columns))
    if missing:
        raise ValueError(f"Training data is missing model features: {missing}")

    constant = find_constant_features(training_frame, candidates)
    selected = [column for column in candidates if column not in constant]
    if not selected:
        raise ValueError("No non-constant model features remain")
    return selected, constant


def capped_rul(
    frame: pd.DataFrame,
    cap: int = DEFAULT_RUL_CAP,
) -> pd.Series:
    """Cap early-life RUL while preserving the important degradation region."""

    if cap <= 0:
        raise ValueError("RUL cap must be positive")
    if "rul" not in frame:
        raise ValueError("Input data does not contain an RUL target")
    if (frame["rul"] < 0).any():
        raise ValueError("RUL cannot be negative")
    return frame["rul"].clip(upper=cap).astype("float64")
