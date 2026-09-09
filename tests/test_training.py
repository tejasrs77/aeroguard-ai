from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from aeroguard.config import CMAPSS_COLUMNS
from aeroguard.data.rul import add_train_rul
from aeroguard.ml.features import select_model_features
from aeroguard.ml.split import split_by_engine
from aeroguard.ml.train import evaluate_candidates, select_champion, train_day2_models


def _small_engine_frame(engine_count: int, cycles: int) -> pd.DataFrame:
    rows = []
    for engine_id in range(1, engine_count + 1):
        for cycle in range(1, cycles + 1):
            row = [engine_id, cycle, 0.0, 0.0, 100.0]
            row.extend(
                float(sensor + cycle * 0.2 + engine_id * 0.05)
                for sensor in range(1, 22)
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=CMAPSS_COLUMNS)


def _baseline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", DummyRegressor(strategy="mean")),
        ]
    )


def test_candidate_evaluation_persists_model_without_tracking(tmp_path: Path) -> None:
    labeled = add_train_rul(_small_engine_frame(6, 7))
    training, validation = split_by_engine(labeled, validation_size=0.33)
    features, _ = select_model_features(training)

    fitted, metrics, run_ids = evaluate_candidates(
        training,
        validation,
        features,
        {"baseline": _baseline()},
        models_dir=tmp_path / "models",
        tracking_dir=None,
    )

    assert list(fitted) == ["baseline"]
    assert metrics.loc[0, "evaluation"] == "validation_all_cycles"
    assert (tmp_path / "models" / "candidate_baseline.joblib").exists()
    assert run_ids == {}


def test_champion_is_selected_by_lowest_rmse() -> None:
    metrics = pd.DataFrame(
        {"model": ["a", "b"], "rmse": [12.0, 9.0], "mae": [8.0, 8.5]}
    )

    assert select_champion(metrics) == "b"


def test_complete_day2_workflow_writes_reproducible_artifacts(tmp_path: Path) -> None:
    train_labeled = add_train_rul(_small_engine_frame(8, 8))
    test_labeled = add_train_rul(_small_engine_frame(4, 5))

    summary = train_day2_models(
        train_labeled,
        test_labeled,
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "figures",
        models_dir=tmp_path / "models",
        tracking_dir=None,
        models={"baseline": _baseline()},
        validation_size=0.25,
    )

    assert summary["engine_overlap"] == 0
    assert summary["champion_model"] == "baseline"
    assert (tmp_path / "models" / "champion_rul_model.joblib").exists()
    assert (tmp_path / "models" / "champion_metadata.json").exists()
    assert (tmp_path / "reports" / "day2_metrics.csv").exists()
    assert (tmp_path / "reports" / "day2_test_predictions.csv").exists()
    assert len(list((tmp_path / "figures").glob("day2_*.png"))) == 4
