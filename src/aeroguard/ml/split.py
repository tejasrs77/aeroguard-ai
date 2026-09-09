"""Engine-aware validation utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from aeroguard.config import RANDOM_STATE


def split_by_engine(
    frame: pd.DataFrame,
    *,
    validation_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create disjoint train/validation partitions grouped by engine ID."""

    if "engine_id" not in frame:
        raise ValueError("Input data must contain engine_id")
    if frame["engine_id"].nunique() < 2:
        raise ValueError("At least two engines are required for a grouped split")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=random_state,
    )
    train_index, validation_index = next(
        splitter.split(frame, groups=frame["engine_id"])
    )
    training = frame.iloc[train_index].sort_values(["engine_id", "cycle"])
    validation = frame.iloc[validation_index].sort_values(["engine_id", "cycle"])

    overlap = set(training["engine_id"]) & set(validation["engine_id"])
    if overlap:
        raise RuntimeError(f"Engine leakage detected: {sorted(overlap)}")
    return training.reset_index(drop=True), validation.reset_index(drop=True)


def last_observation_per_engine(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent available sensor row for every engine."""

    required = {"engine_id", "cycle"}
    if not required.issubset(frame.columns):
        raise ValueError("Input data must contain engine_id and cycle")
    return (
        frame.sort_values(["engine_id", "cycle"])
        .groupby("engine_id", as_index=False, sort=True)
        .tail(1)
        .sort_values("engine_id")
        .reset_index(drop=True)
    )
