"""Parse the whitespace-separated C-MAPSS source files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aeroguard.config import CMAPSS_COLUMNS, RAW_DIR


def read_cmapss(path: Path) -> pd.DataFrame:
    """Read one 26-column sensor file and enforce numeric source types."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMNS,
        engine="python",
        on_bad_lines="error",
    )
    if frame.empty:
        raise ValueError(f"Dataset file is empty: {path}")
    if frame.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(
            f"Expected {len(CMAPSS_COLUMNS)} columns, found {frame.shape[1]}"
        )

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any(axis=None):
        bad_cells = int(numeric.isna().sum().sum())
        raise ValueError(f"Source contains {bad_cells} missing or non-numeric values")

    for column in ("engine_id", "cycle"):
        values = numeric[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain whole numbers")
        numeric[column] = numeric[column].astype("int64")

    return numeric


def read_rul_truth(path: Path) -> pd.Series:
    """Read the test-set RUL value supplied for each engine."""

    if not path.exists():
        raise FileNotFoundError(f"RUL truth file does not exist: {path}")
    truth = pd.read_csv(path, sep=r"\s+", header=None, names=["final_rul"])[
        "final_rul"
    ]
    truth = pd.to_numeric(truth, errors="coerce")
    if truth.empty or truth.isna().any():
        raise ValueError("RUL truth must contain only numeric values")
    return truth.astype("int64")


def load_fd001(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the FD001 train, test, and truth sources as one logical bundle."""

    train = read_cmapss(raw_dir / "train_FD001.txt")
    test = read_cmapss(raw_dir / "test_FD001.txt")
    truth = read_rul_truth(raw_dir / "RUL_FD001.txt")
    return train, test, truth

