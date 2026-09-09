"""Create leakage-safe, time-ordered engine sequences for deep learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from aeroguard.config import DEFAULT_RUL_CAP
from aeroguard.ml.features import capped_rul


@dataclass(frozen=True)
class SequenceBatch:
    """Model inputs and the engine-cycle endpoint represented by each sequence."""

    values: np.ndarray
    targets: np.ndarray
    engine_ids: np.ndarray
    cycles: np.ndarray

    def __post_init__(self) -> None:
        sample_count = len(self.values)
        if self.values.ndim != 3:
            raise ValueError("Sequence values must have shape [samples, time, features]")
        if not all(
            len(item) == sample_count
            for item in (self.targets, self.engine_ids, self.cycles)
        ):
            raise ValueError("Sequence metadata must have one value per sample")


def fit_feature_scaler(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
) -> StandardScaler:
    """Fit scaling statistics on training engines only."""

    if training_frame[feature_columns].isna().any().any():
        raise ValueError("Sequence features contain missing values")
    return StandardScaler().fit(training_frame[feature_columns])


def apply_feature_scaler(
    frame: pd.DataFrame,
    feature_columns: list[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Apply pre-fitted scaling while retaining IDs, cycles, and targets."""

    transformed = frame.copy()
    transformed["_sequence_cycle"] = frame["cycle"].astype("int64")
    transformed[feature_columns] = scaler.transform(frame[feature_columns])
    return transformed


def build_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    sequence_length: int = 30,
    stride: int = 1,
    rul_cap: int = DEFAULT_RUL_CAP,
) -> SequenceBatch:
    """Build past-to-present windows that never cross an engine boundary."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    values: list[np.ndarray] = []
    targets: list[float] = []
    engine_ids: list[int] = []
    cycles: list[int] = []

    cycle_metadata_column = (
        "_sequence_cycle" if "_sequence_cycle" in frame.columns else "cycle"
    )
    ordered = frame.sort_values(["engine_id", cycle_metadata_column])
    for engine_id, engine in ordered.groupby("engine_id", sort=True):
        engine = engine.reset_index(drop=True)
        if len(engine) < sequence_length:
            continue
        engine_values = engine[feature_columns].to_numpy(dtype="float32")
        engine_targets = capped_rul(engine, rul_cap).to_numpy(dtype="float32")
        engine_cycles = engine[cycle_metadata_column].to_numpy(dtype="int64")
        for end in range(sequence_length - 1, len(engine), stride):
            start = end - sequence_length + 1
            values.append(engine_values[start : end + 1])
            targets.append(float(engine_targets[end]))
            engine_ids.append(int(engine_id))
            cycles.append(int(engine_cycles[end]))

    if not values:
        raise ValueError("No engine contains enough rows to build a sequence")
    return SequenceBatch(
        values=np.stack(values).astype("float32"),
        targets=np.asarray(targets, dtype="float32"),
        engine_ids=np.asarray(engine_ids, dtype="int64"),
        cycles=np.asarray(cycles, dtype="int64"),
    )


def build_last_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    sequence_length: int = 30,
    rul_cap: int = DEFAULT_RUL_CAP,
) -> SequenceBatch:
    """Build exactly one final sequence for every engine."""

    values: list[np.ndarray] = []
    targets: list[float] = []
    engine_ids: list[int] = []
    cycles: list[int] = []

    cycle_metadata_column = (
        "_sequence_cycle" if "_sequence_cycle" in frame.columns else "cycle"
    )
    ordered = frame.sort_values(["engine_id", cycle_metadata_column])
    for engine_id, engine in ordered.groupby("engine_id", sort=True):
        if len(engine) < sequence_length:
            raise ValueError(
                f"Engine {engine_id} has {len(engine)} rows; {sequence_length} required"
            )
        engine = engine.tail(sequence_length)
        values.append(engine[feature_columns].to_numpy(dtype="float32"))
        targets.append(float(capped_rul(engine, rul_cap).iloc[-1]))
        engine_ids.append(int(engine_id))
        cycles.append(int(engine[cycle_metadata_column].iloc[-1]))

    if not values:
        raise ValueError("No engines were supplied")
    return SequenceBatch(
        values=np.stack(values).astype("float32"),
        targets=np.asarray(targets, dtype="float32"),
        engine_ids=np.asarray(engine_ids, dtype="int64"),
        cycles=np.asarray(cycles, dtype="int64"),
    )
