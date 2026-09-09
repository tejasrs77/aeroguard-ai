"""Train, compare, track, and persist Day 2 RUL regression models."""

from __future__ import annotations

import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from mlflow.tracking import MlflowClient

from aeroguard.config import (
    DEFAULT_RUL_CAP,
    FIGURES_DIR,
    MLFLOW_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from aeroguard.ml.features import capped_rul, select_model_features
from aeroguard.ml.metrics import regression_metrics
from aeroguard.ml.split import last_observation_per_engine, split_by_engine


EXPERIMENT_NAME = "AeroGuard FD001 RUL"


def _configure_mlflow(tracking_dir: Path) -> None:
    """Use a supported local SQLite store with a separate artifact directory."""

    tracking_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = tracking_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    database_path = (tracking_dir / "mlflow.db").resolve().as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{database_path}")

    client = MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=artifact_dir.resolve().as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)


def build_candidate_models(
    random_state: int = RANDOM_STATE,
) -> dict[str, RegressorMixin]:
    """Create an interpretable baseline and progressively stronger models."""

    return {
        "mean_baseline": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", DummyRegressor(strategy="mean")),
            ]
        ),
        "ridge_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=10.0)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=220,
                        max_depth=14,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    XGBRegressor(
                        n_estimators=450,
                        max_depth=5,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=4,
                        reg_lambda=1.0,
                        objective="reg:squarederror",
                        tree_method="hist",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _scalar_model_parameters(model: RegressorMixin) -> dict[str, Any]:
    """Select MLflow-safe scalar parameters from the final estimator."""

    estimator = model.named_steps["regressor"] if isinstance(model, Pipeline) else model
    parameters: dict[str, Any] = {}
    for key, value in estimator.get_params(deep=False).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            parameters[f"model__{key}"] = "None" if value is None else value
    return parameters


def evaluate_candidates(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    models: Mapping[str, RegressorMixin],
    *,
    rul_cap: int = DEFAULT_RUL_CAP,
    models_dir: Path = MODELS_DIR,
    tracking_dir: Path | None = MLFLOW_DIR,
) -> tuple[dict[str, RegressorMixin], pd.DataFrame, dict[str, str]]:
    """Fit candidate models and evaluate them on unseen engines."""

    if not models:
        raise ValueError("At least one candidate model is required")

    models_dir.mkdir(parents=True, exist_ok=True)
    x_train = training[feature_columns]
    y_train = capped_rul(training, rul_cap)
    x_validation = validation[feature_columns]
    y_validation = capped_rul(validation, rul_cap)

    tracking_enabled = tracking_dir is not None
    if tracking_enabled:
        _configure_mlflow(tracking_dir)

    fitted: dict[str, RegressorMixin] = {}
    rows: list[dict[str, Any]] = []
    run_ids: dict[str, str] = {}

    for name, candidate in models.items():
        model = clone(candidate)
        start = perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = perf_counter() - start
        predictions = np.clip(model.predict(x_validation), 0, rul_cap)
        metrics = regression_metrics(y_validation.to_numpy(), predictions)
        model_path = models_dir / f"candidate_{name}.joblib"
        joblib.dump(model, model_path)

        row = {
            "model": name,
            "evaluation": "validation_all_cycles",
            "records": int(len(validation)),
            "fit_seconds": float(fit_seconds),
            **metrics,
        }
        rows.append(row)
        fitted[name] = model

        if tracking_enabled:
            with mlflow.start_run(run_name=f"validation_{name}") as run:
                mlflow.set_tags(
                    {
                        "dataset": "NASA C-MAPSS FD001",
                        "evaluation": "unseen_validation_engines_all_cycles",
                        "stage": "candidate_selection",
                    }
                )
                mlflow.log_params(
                    {
                        "feature_count": len(feature_columns),
                        "rul_cap": rul_cap,
                        "training_engines": training["engine_id"].nunique(),
                        "validation_engines": validation["engine_id"].nunique(),
                        **_scalar_model_parameters(model),
                    }
                )
                mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
                mlflow.log_metric("fit_seconds", fit_seconds)
                mlflow.log_artifact(str(model_path), artifact_path="model")
                run_ids[name] = run.info.run_id

    metrics_frame = pd.DataFrame(rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    return fitted, metrics_frame, run_ids


def select_champion(metrics: pd.DataFrame) -> str:
    """Select the model with the lowest validation RMSE only."""

    required = {"model", "rmse"}
    if not required.issubset(metrics.columns) or metrics.empty:
        raise ValueError("Model metrics must contain at least one model and RMSE")
    return str(metrics.sort_values(["rmse", "mae"]).iloc[0]["model"])


def _feature_importance(
    model: RegressorMixin,
    feature_columns: list[str],
) -> pd.DataFrame:
    estimator = model.named_steps["regressor"] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype="float64")
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.ravel(estimator.coef_)).astype("float64")
    else:
        importance = np.zeros(len(feature_columns), dtype="float64")
    return (
        pd.DataFrame({"feature": feature_columns, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _save_day2_figures(
    validation_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    *,
    figures_dir: Path,
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    comparison_path = figures_dir / "day2_model_comparison.png"
    comparison = validation_metrics.sort_values("rmse", ascending=False)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    positions = np.arange(len(comparison))
    width = 0.36
    axis.bar(positions - width / 2, comparison["mae"], width, label="MAE", color="#14b8a6")
    axis.bar(positions + width / 2, comparison["rmse"], width, label="RMSE", color="#2563eb")
    axis.set_xticks(positions, comparison["model"], rotation=15, ha="right")
    axis.set_ylabel("Error in cycles (lower is better)")
    axis.set_title("Validation performance on completely unseen engines")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(comparison_path, dpi=160)
    plt.close(figure)
    saved.append(comparison_path.name)

    prediction_path = figures_dir / "day2_test_predicted_vs_actual.png"
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        predictions["actual_rul_capped"],
        predictions["predicted_rul"],
        alpha=0.72,
        color="#2563eb",
        edgecolors="white",
        linewidths=0.4,
    )
    limit = max(
        float(predictions["actual_rul_capped"].max()),
        float(predictions["predicted_rul"].max()),
    )
    axis.plot([0, limit], [0, limit], linestyle="--", color="#dc2626", label="Perfect prediction")
    axis.set_xlabel("Actual RUL (cycles, capped)")
    axis.set_ylabel("Predicted RUL (cycles)")
    axis.set_title("Champion model on official test-engine endpoints")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(prediction_path, dpi=160)
    plt.close(figure)
    saved.append(prediction_path.name)

    residual_path = figures_dir / "day2_test_residuals.png"
    residuals = predictions["predicted_rul"] - predictions["actual_rul_capped"]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.hist(residuals, bins=18, color="#14b8a6", edgecolor="white")
    axis.axvline(0, color="#dc2626", linestyle="--", label="Zero error")
    axis.set_xlabel("Prediction error: predicted − actual (cycles)")
    axis.set_ylabel("Test engines")
    axis.set_title("Champion-model residual distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(residual_path, dpi=160)
    plt.close(figure)
    saved.append(residual_path.name)

    importance_path = figures_dir / "day2_feature_importance.png"
    top = importance.head(15).sort_values("importance")
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top["feature"], top["importance"], color="#2563eb")
    axis.set_xlabel("Model importance (relative, not causal)")
    axis.set_title("Top inputs used by the champion model")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(importance_path, dpi=160)
    plt.close(figure)
    saved.append(importance_path.name)
    return saved


def train_day2_models(
    train_labeled: pd.DataFrame,
    test_labeled: pd.DataFrame,
    *,
    reports_dir: Path = REPORTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    models_dir: Path = MODELS_DIR,
    tracking_dir: Path | None = MLFLOW_DIR,
    models: Mapping[str, RegressorMixin] | None = None,
    validation_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    rul_cap: int = DEFAULT_RUL_CAP,
) -> dict[str, Any]:
    """Run the complete leakage-safe Day 2 model-development workflow."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    training, validation = split_by_engine(
        train_labeled,
        validation_size=validation_size,
        random_state=random_state,
    )
    feature_columns, constant_features = select_model_features(training)
    candidates = dict(models) if models is not None else build_candidate_models(random_state)

    _, validation_metrics, run_ids = evaluate_candidates(
        training,
        validation,
        feature_columns,
        candidates,
        rul_cap=rul_cap,
        models_dir=models_dir,
        tracking_dir=tracking_dir,
    )
    champion_name = select_champion(validation_metrics)

    champion = clone(candidates[champion_name])
    champion.fit(train_labeled[feature_columns], capped_rul(train_labeled, rul_cap))
    test_endpoints = last_observation_per_engine(test_labeled)
    predicted = np.clip(champion.predict(test_endpoints[feature_columns]), 0, rul_cap)
    actual_uncapped = test_endpoints["rul"].to_numpy(dtype="float64")
    actual_capped = capped_rul(test_endpoints, rul_cap).to_numpy()
    test_metrics_capped = regression_metrics(actual_capped, predicted)
    test_metrics_uncapped = regression_metrics(actual_uncapped, predicted)

    champion_path = models_dir / "champion_rul_model.joblib"
    joblib.dump(champion, champion_path)
    importance = _feature_importance(champion, feature_columns)
    importance.to_csv(reports_dir / "day2_feature_importance.csv", index=False)

    test_predictions = test_endpoints[["engine_id", "cycle", "rul"]].rename(
        columns={"rul": "actual_rul"}
    )
    test_predictions["actual_rul_capped"] = actual_capped
    test_predictions["predicted_rul"] = predicted
    test_predictions["residual"] = predicted - actual_capped
    test_predictions.to_csv(reports_dir / "day2_test_predictions.csv", index=False)

    test_rows = [
        {
            "model": champion_name,
            "evaluation": "official_test_endpoints_capped",
            "records": int(len(test_endpoints)),
            "fit_seconds": np.nan,
            **test_metrics_capped,
        },
        {
            "model": champion_name,
            "evaluation": "official_test_endpoints_uncapped",
            "records": int(len(test_endpoints)),
            "fit_seconds": np.nan,
            **test_metrics_uncapped,
        },
    ]
    all_metrics = pd.concat(
        [validation_metrics, pd.DataFrame(test_rows)], ignore_index=True
    )
    all_metrics.to_csv(reports_dir / "day2_metrics.csv", index=False)

    figures = _save_day2_figures(
        validation_metrics,
        test_predictions,
        importance,
        figures_dir=figures_dir,
    )

    champion_run_id: str | None = None
    if tracking_dir is not None:
        _configure_mlflow(tracking_dir)
        with mlflow.start_run(run_name=f"champion_refit_{champion_name}") as run:
            mlflow.set_tags(
                {
                    "dataset": "NASA C-MAPSS FD001",
                    "evaluation": "official_test_engine_endpoints",
                    "stage": "champion_refit",
                    "selected_using": "validation_rmse_only",
                }
            )
            mlflow.log_params(
                {
                    "champion_model": champion_name,
                    "feature_count": len(feature_columns),
                    "rul_cap": rul_cap,
                    "training_engines": train_labeled["engine_id"].nunique(),
                    **_scalar_model_parameters(champion),
                }
            )
            mlflow.log_metrics(
                {f"test_capped_{key}": value for key, value in test_metrics_capped.items()}
            )
            mlflow.log_artifact(str(champion_path), artifact_path="model")
            mlflow.log_artifact(
                str(reports_dir / "day2_test_predictions.csv"), artifact_path="reports"
            )
            champion_run_id = run.info.run_id

    metadata = {
        "model_name": champion_name,
        "feature_columns": feature_columns,
        "constant_features_dropped": constant_features,
        "rul_cap": rul_cap,
        "random_state": random_state,
        "selection_metric": "validation_rmse",
        "selection_scope": "all cycles from unseen validation engines",
        "final_evaluation_scope": "last observed row of each official test engine",
    }
    (models_dir / "champion_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    summary: dict[str, Any] = {
        "training_rows": int(len(training)),
        "validation_rows": int(len(validation)),
        "training_engines": int(training["engine_id"].nunique()),
        "validation_engines": int(validation["engine_id"].nunique()),
        "official_test_engines": int(len(test_endpoints)),
        "engine_overlap": 0,
        "rul_cap": rul_cap,
        "selected_feature_count": len(feature_columns),
        "constant_features_dropped": constant_features,
        "candidate_models": list(candidates),
        "champion_model": champion_name,
        "champion_validation_metrics": validation_metrics.iloc[0][
            ["mae", "rmse", "r2", "nasa_score"]
        ].to_dict(),
        "official_test_metrics_capped": test_metrics_capped,
        "official_test_metrics_uncapped": test_metrics_uncapped,
        "candidate_mlflow_run_ids": run_ids,
        "champion_mlflow_run_id": champion_run_id,
        "figures": figures,
    }
    (reports_dir / "day2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
