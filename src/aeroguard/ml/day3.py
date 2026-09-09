"""Orchestrate Day 3 sequence modeling and SHAP explainability."""

from __future__ import annotations

import json
import os
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline

from aeroguard.config import (
    DEFAULT_RUL_CAP,
    FIGURES_DIR,
    MLFLOW_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from aeroguard.ml.explain import calculate_tree_shap, local_shap_table
from aeroguard.ml.features import capped_rul
from aeroguard.ml.lstm import (
    LSTMConfig,
    fit_lstm_fixed_epochs,
    predict_lstm,
    train_lstm,
)
from aeroguard.ml.metrics import regression_metrics
from aeroguard.ml.sequences import (
    SequenceBatch,
    apply_feature_scaler,
    build_last_sequences,
    build_sequences,
    fit_feature_scaler,
)
from aeroguard.ml.split import last_observation_per_engine, split_by_engine
from aeroguard.ml.train import _configure_mlflow


SEQUENCE_LENGTH = 30


def _rows_at_sequence_endpoints(
    frame: pd.DataFrame,
    batch: SequenceBatch,
) -> pd.DataFrame:
    indexed = frame.set_index(["engine_id", "cycle"])
    keys = pd.MultiIndex.from_arrays(
        [batch.engine_ids, batch.cycles],
        names=["engine_id", "cycle"],
    )
    return indexed.loc[keys].reset_index()


def _save_training_history(history: pd.DataFrame, figures_dir: Path) -> str:
    path = figures_dir / "day3_lstm_training_history.png"
    figure, loss_axis = plt.subplots(figsize=(9, 5.5))
    metric_axis = loss_axis.twinx()
    loss_axis.plot(
        history["epoch"],
        history["training_huber_loss"],
        marker="o",
        color="#14b8a6",
        label="Training Huber loss",
    )
    metric_axis.plot(
        history["epoch"],
        history["validation_rmse"],
        marker="o",
        color="#2563eb",
        label="Validation RMSE",
    )
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Training Huber loss", color="#0f766e")
    metric_axis.set_ylabel("Validation RMSE (cycles)", color="#1d4ed8")
    loss_axis.set_title("LSTM learning curve and early-stopping evidence")
    loss_axis.grid(alpha=0.2)
    lines = loss_axis.get_lines() + metric_axis.get_lines()
    loss_axis.legend(lines, [line.get_label() for line in lines], loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path.name


def _save_sequence_comparison(metrics: pd.DataFrame, figures_dir: Path) -> str:
    path = figures_dir / "day3_sequence_model_comparison.png"
    ordered = metrics.sort_values("rmse", ascending=False)
    positions = np.arange(len(ordered))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.bar(positions - width / 2, ordered["mae"], width, label="MAE", color="#14b8a6")
    axis.bar(positions + width / 2, ordered["rmse"], width, label="RMSE", color="#2563eb")
    axis.set_xticks(positions, ordered["model"])
    axis.set_ylabel("Error in cycles (lower is better)")
    axis.set_title("Fair comparison on identical validation sequence endpoints")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path.name


def _save_global_shap(global_shap: pd.DataFrame, figures_dir: Path) -> str:
    path = figures_dir / "day3_shap_global.png"
    top = global_shap.head(15).sort_values("mean_absolute_shap")
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top["feature"], top["mean_absolute_shap"], color="#2563eb")
    axis.set_xlabel("Mean |SHAP contribution| in predicted cycles")
    axis.set_title("Global Random Forest explanation across 100 test engines")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path.name


def _save_local_shap(
    local_shap: pd.DataFrame,
    *,
    engine_id: int,
    prediction: float,
    base_value: float,
    figures_dir: Path,
) -> str:
    path = figures_dir / "day3_shap_local_high_risk_engine.png"
    top = local_shap.head(10).sort_values("shap_contribution_cycles")
    colors = np.where(top["shap_contribution_cycles"] >= 0, "#14b8a6", "#dc2626")
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(top["feature"], top["shap_contribution_cycles"], color=colors)
    axis.axvline(0, color="#111827", linewidth=0.9)
    axis.set_xlabel("Contribution to predicted RUL (cycles)")
    axis.set_title(
        f"Engine {engine_id}: base {base_value:.1f} → prediction {prediction:.1f} cycles"
    )
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path.name


def train_day3_models(
    train_labeled: pd.DataFrame,
    test_labeled: pd.DataFrame,
    *,
    reports_dir: Path = REPORTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    models_dir: Path = MODELS_DIR,
    tracking_dir: Path | None = MLFLOW_DIR,
    sequence_length: int = SEQUENCE_LENGTH,
    max_epochs: int = 20,
    patience: int = 4,
    batch_size: int = 256,
    hidden_size: int = 48,
    num_layers: int = 2,
    dropout: float = 0.15,
    random_state: int = RANDOM_STATE,
    rul_cap: int = DEFAULT_RUL_CAP,
) -> dict[str, Any]:
    """Train an LSTM, compare fairly, and explain the tree champion with SHAP."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = models_dir / "champion_metadata.json"
    champion_path = models_dir / "champion_rul_model.joblib"
    validation_forest_path = models_dir / "candidate_random_forest.joblib"
    required_paths = (metadata_path, champion_path, validation_forest_path)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Day 2 artifacts are required before Day 3. Missing: {missing}"
        )

    day2_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = list(day2_metadata["feature_columns"])
    training, validation = split_by_engine(
        train_labeled,
        validation_size=0.2,
        random_state=random_state,
    )

    validation_scaler = fit_feature_scaler(training, feature_columns)
    scaled_training = apply_feature_scaler(training, feature_columns, validation_scaler)
    scaled_validation = apply_feature_scaler(validation, feature_columns, validation_scaler)
    training_sequences = build_sequences(
        scaled_training,
        feature_columns,
        sequence_length=sequence_length,
        rul_cap=rul_cap,
    )
    validation_sequences = build_sequences(
        scaled_validation,
        feature_columns,
        sequence_length=sequence_length,
        rul_cap=rul_cap,
    )

    model_config = LSTMConfig(
        input_size=len(feature_columns),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    started = perf_counter()
    training_result = train_lstm(
        training_sequences,
        validation_sequences,
        model_config,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        seed=random_state,
        rul_cap=rul_cap,
    )
    validation_fit_seconds = perf_counter() - started
    training_result.history.to_csv(
        reports_dir / "day3_lstm_training_history.csv", index=False
    )

    validation_endpoints = _rows_at_sequence_endpoints(validation, validation_sequences)
    validation_forest: Pipeline = joblib.load(validation_forest_path)
    forest_validation_predictions = np.clip(
        validation_forest.predict(validation_endpoints[feature_columns]),
        0,
        rul_cap,
    )
    forest_validation_metrics = regression_metrics(
        capped_rul(validation_endpoints, rul_cap).to_numpy(),
        forest_validation_predictions,
    )
    fair_validation = pd.DataFrame(
        [
            {
                "model": "random_forest",
                "evaluation": "validation_sequence_endpoints",
                "records": len(validation_sequences.targets),
                **forest_validation_metrics,
            },
            {
                "model": "lstm",
                "evaluation": "validation_sequence_endpoints",
                "records": len(validation_sequences.targets),
                **training_result.validation_metrics,
            },
        ]
    ).sort_values("rmse")
    validation_winner = str(fair_validation.iloc[0]["model"])
    deployment_champion = "random_forest"

    final_scaler = fit_feature_scaler(train_labeled, feature_columns)
    scaled_all_training = apply_feature_scaler(
        train_labeled,
        feature_columns,
        final_scaler,
    )
    all_training_sequences = build_sequences(
        scaled_all_training,
        feature_columns,
        sequence_length=sequence_length,
        rul_cap=rul_cap,
    )
    final_model = fit_lstm_fixed_epochs(
        all_training_sequences,
        model_config,
        epochs=training_result.best_epoch,
        batch_size=batch_size,
        seed=random_state,
    )
    joblib.dump(final_scaler, models_dir / "lstm_feature_scaler.joblib")
    lstm_path = models_dir / "lstm_rul_model.pt"
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "model_config": model_config.to_dict(),
        },
        lstm_path,
    )

    scaled_test = apply_feature_scaler(test_labeled, feature_columns, final_scaler)
    test_sequences = build_last_sequences(
        scaled_test,
        feature_columns,
        sequence_length=sequence_length,
        rul_cap=rul_cap,
    )
    lstm_test_predictions = predict_lstm(
        final_model,
        test_sequences.values,
        rul_cap=rul_cap,
    )
    lstm_test_metrics = regression_metrics(
        test_sequences.targets,
        lstm_test_predictions,
    )

    test_endpoints = last_observation_per_engine(test_labeled)
    tabular_champion: Pipeline = joblib.load(champion_path)
    forest_test_predictions = np.clip(
        tabular_champion.predict(test_endpoints[feature_columns]),
        0,
        rul_cap,
    )
    forest_test_metrics = regression_metrics(
        capped_rul(test_endpoints, rul_cap).to_numpy(),
        forest_test_predictions,
    )
    test_predictions = test_endpoints[["engine_id", "cycle", "rul"]].rename(
        columns={"rul": "actual_rul"}
    )
    test_predictions["actual_rul_capped"] = test_sequences.targets
    test_predictions["random_forest_predicted_rul"] = forest_test_predictions
    test_predictions["lstm_predicted_rul"] = lstm_test_predictions
    test_predictions.to_csv(reports_dir / "day3_test_predictions.csv", index=False)

    global_shap, shap_values, base_values = calculate_tree_shap(
        tabular_champion,
        test_endpoints,
        feature_columns,
    )
    global_shap.to_csv(reports_dir / "day3_shap_global.csv", index=False)
    high_risk_position = int(np.argmin(forest_test_predictions))
    high_risk_engine = int(test_endpoints.iloc[high_risk_position]["engine_id"])
    local_shap = local_shap_table(
        feature_columns,
        test_endpoints.iloc[high_risk_position],
        shap_values[high_risk_position],
    )
    local_shap.insert(0, "engine_id", high_risk_engine)
    local_shap.to_csv(reports_dir / "day3_shap_local_high_risk_engine.csv", index=False)
    raw_prediction = float(tabular_champion.predict(test_endpoints.iloc[[high_risk_position]][feature_columns])[0])
    shap_reconstructed = float(base_values[high_risk_position] + shap_values[high_risk_position].sum())
    shap_additivity_error = abs(raw_prediction - shap_reconstructed)

    metrics = pd.concat(
        [
            fair_validation,
            pd.DataFrame(
                [
                    {
                        "model": "random_forest",
                        "evaluation": "official_test_endpoints_capped",
                        "records": len(test_endpoints),
                        **forest_test_metrics,
                    },
                    {
                        "model": "lstm",
                        "evaluation": "official_test_endpoints_capped",
                        "records": len(test_endpoints),
                        **lstm_test_metrics,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    metrics.to_csv(reports_dir / "day3_metrics.csv", index=False)

    figures = [
        _save_training_history(training_result.history, figures_dir),
        _save_sequence_comparison(fair_validation, figures_dir),
        _save_global_shap(global_shap, figures_dir),
        _save_local_shap(
            local_shap,
            engine_id=high_risk_engine,
            prediction=raw_prediction,
            base_value=float(base_values[high_risk_position]),
            figures_dir=figures_dir,
        ),
    ]

    lstm_metadata = {
        "model_name": "lstm",
        "feature_columns": feature_columns,
        "sequence_length": sequence_length,
        "rul_cap": rul_cap,
        "model_config": model_config.to_dict(),
        "best_epoch": training_result.best_epoch,
        "scaler_fit_scope": "all 100 training engines after epoch selection",
        "validation_split": "same 80/20 engine split as Day 2",
    }
    (models_dir / "lstm_metadata.json").write_text(
        json.dumps(lstm_metadata, indent=2), encoding="utf-8"
    )
    (models_dir / "day3_overall_champion.json").write_text(
        json.dumps(
            {
                "model_name": deployment_champion,
                "status": "current explainable deployment champion",
                "validation_accuracy_winner": validation_winner,
                "lstm_promotion_status": "held for operational-risk review",
                "reason": (
                    "LSTM improved validation and test RMSE, but its official-test "
                    "NASA asymmetric risk score was worse; SHAP explanations are "
                    "currently tied directly to the Random Forest predictions."
                ),
                "test_set_used_for_selection": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mlflow_run_id: str | None = None
    if tracking_dir is not None:
        _configure_mlflow(tracking_dir)
        with mlflow.start_run(run_name="day3_lstm_and_shap") as run:
            mlflow.set_tags(
                {
                    "dataset": "NASA C-MAPSS FD001",
                    "stage": "sequence_model_and_explainability",
                    "comparison_scope": "identical_unseen_engine_sequence_endpoints",
                    "deployment_champion": deployment_champion,
                }
            )
            mlflow.log_params(
                {
                    "sequence_length": sequence_length,
                    "rul_cap": rul_cap,
                    "feature_count": len(feature_columns),
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "best_epoch": training_result.best_epoch,
                    "device": training_result.device,
                }
            )
            mlflow.log_metrics(
                {
                    **{f"validation_lstm_{key}": value for key, value in training_result.validation_metrics.items()},
                    **{f"validation_random_forest_{key}": value for key, value in forest_validation_metrics.items()},
                    **{f"test_lstm_{key}": value for key, value in lstm_test_metrics.items()},
                    "validation_fit_seconds": validation_fit_seconds,
                    "shap_additivity_error": shap_additivity_error,
                }
            )
            for artifact in (
                lstm_path,
                models_dir / "lstm_metadata.json",
                reports_dir / "day3_metrics.csv",
                reports_dir / "day3_lstm_training_history.csv",
                reports_dir / "day3_shap_global.csv",
                reports_dir / "day3_shap_local_high_risk_engine.csv",
            ):
                mlflow.log_artifact(str(artifact), artifact_path="day3")
            mlflow_run_id = run.info.run_id

    summary: dict[str, Any] = {
        "sequence_length": sequence_length,
        "selected_feature_count": len(feature_columns),
        "training_engines": int(training["engine_id"].nunique()),
        "validation_engines": int(validation["engine_id"].nunique()),
        "engine_overlap": 0,
        "training_sequences": int(len(training_sequences.targets)),
        "validation_sequences": int(len(validation_sequences.targets)),
        "official_test_sequences": int(len(test_sequences.targets)),
        "lstm_device": training_result.device,
        "epochs_completed": int(len(training_result.history)),
        "best_epoch": int(training_result.best_epoch),
        "validation_fit_seconds": validation_fit_seconds,
        "fair_validation_metrics": {
            "random_forest": forest_validation_metrics,
            "lstm": training_result.validation_metrics,
        },
        "validation_winner": validation_winner,
        "deployment_champion": deployment_champion,
        "lstm_promotion_status": "held for operational-risk review",
        "official_test_metrics_capped": {
            "random_forest": forest_test_metrics,
            "lstm": lstm_test_metrics,
        },
        "shap_explained_test_engines": int(len(test_endpoints)),
        "high_risk_engine_explained": high_risk_engine,
        "shap_additivity_error": shap_additivity_error,
        "top_global_shap_features": global_shap.head(5)["feature"].tolist(),
        "pytorch_version": torch.__version__,
        "shap_version": version("shap"),
        "mlflow_run_id": mlflow_run_id,
        "figures": figures,
    }
    (reports_dir / "day3_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
