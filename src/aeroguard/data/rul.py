"""Construct Remaining Useful Life targets without leaking future engines."""

from __future__ import annotations

import pandas as pd


def add_train_rul(frame: pd.DataFrame, failure_horizon: int = 30) -> pd.DataFrame:
    """Label training rows, whose final recorded cycle is the failure cycle."""

    labeled = frame.copy()
    last_cycle = labeled.groupby("engine_id")["cycle"].transform("max")
    labeled["rul"] = (last_cycle - labeled["cycle"]).astype("int64")
    labeled["failure_within_30_cycles"] = (labeled["rul"] <= failure_horizon).astype(
        "int8"
    )
    return labeled


def add_test_rul(
    frame: pd.DataFrame,
    final_rul: pd.Series,
    failure_horizon: int = 30,
) -> pd.DataFrame:
    """Label test rows using the supplied RUL at each engine's last observation."""

    engine_ids = sorted(frame["engine_id"].unique())
    if len(engine_ids) != len(final_rul):
        raise ValueError(
            f"Expected one final RUL for {len(engine_ids)} engines, found {len(final_rul)}"
        )

    final_rul_by_engine = dict(zip(engine_ids, final_rul.astype(int), strict=True))
    labeled = frame.copy()
    max_observed = labeled.groupby("engine_id")["cycle"].transform("max")
    observed_tail_rul = labeled["engine_id"].map(final_rul_by_engine)
    labeled["rul"] = (observed_tail_rul + max_observed - labeled["cycle"]).astype(
        "int64"
    )
    labeled["failure_within_30_cycles"] = (labeled["rul"] <= failure_horizon).astype(
        "int8"
    )
    return labeled

