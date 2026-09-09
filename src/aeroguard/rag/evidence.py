"""Create a trustworthy engine evidence packet from persisted model artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from aeroguard.config import DEFAULT_RUL_CAP, MODELS_DIR, PROCESSED_DIR
from aeroguard.ml.explain import calculate_tree_shap, local_shap_table
from aeroguard.ml.lstm import load_lstm_checkpoint, predict_lstm
from aeroguard.ml.sequences import apply_feature_scaler, build_last_sequences


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: float
    contribution_cycles: float
    direction: str


@dataclass(frozen=True)
class EngineEvidence:
    """Only verified numerical facts that a text generator is allowed to describe."""

    engine_id: int
    observed_through_cycle: int
    deployment_model: str
    predicted_rul_cycles: float
    risk_band: str
    advisory_lstm_rul_cycles: float | None
    model_disagreement_cycles: float | None
    shap_base_value_cycles: float
    shap_additivity_error: float
    top_feature_contributions: tuple[FeatureContribution, ...]
    evidence_boundary: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_risk(predicted_rul: float) -> str:
    if predicted_rul <= 15:
        return "critical"
    if predicted_rul <= 30:
        return "high"
    if predicted_rul <= 60:
        return "elevated"
    return "routine"


def build_engine_evidence(
    engine_id: int,
    *,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    top_features: int = 5,
) -> EngineEvidence:
    """Recompute deployment prediction, advisory prediction, and local TreeSHAP."""

    if engine_id <= 0:
        raise ValueError("engine_id must be positive")
    if top_features <= 0:
        raise ValueError("top_features must be positive")

    frame = pd.read_csv(processed_dir / "test_fd001_labeled.csv")
    engine = frame.loc[frame["engine_id"] == engine_id].sort_values("cycle")
    if engine.empty:
        raise ValueError(f"Engine {engine_id} is not present in the test dataset")

    champion_metadata = json.loads(
        (models_dir / "champion_metadata.json").read_text(encoding="utf-8")
    )
    feature_columns = list(champion_metadata["feature_columns"])
    endpoint = engine.tail(1)
    forest: Pipeline = joblib.load(models_dir / "champion_rul_model.joblib")
    raw_forest_prediction = float(forest.predict(endpoint[feature_columns])[0])
    forest_prediction = float(np.clip(raw_forest_prediction, 0, DEFAULT_RUL_CAP))

    _, shap_values, base_values = calculate_tree_shap(
        forest,
        endpoint,
        feature_columns,
    )
    shap_table = local_shap_table(
        feature_columns,
        endpoint.iloc[0],
        shap_values[0],
    ).head(top_features)
    contributions = tuple(
        FeatureContribution(
            feature=str(row.feature),
            value=float(row.feature_value),
            contribution_cycles=float(row.shap_contribution_cycles),
            direction=str(row.direction),
        )
        for row in shap_table.itertuples(index=False)
    )
    reconstructed = float(base_values[0] + shap_values[0].sum())

    advisory_lstm: float | None = None
    lstm_metadata_path = models_dir / "lstm_metadata.json"
    if lstm_metadata_path.exists():
        lstm_metadata = json.loads(lstm_metadata_path.read_text(encoding="utf-8"))
        lstm_features = list(lstm_metadata["feature_columns"])
        scaler = joblib.load(models_dir / "lstm_feature_scaler.joblib")
        scaled = apply_feature_scaler(engine, lstm_features, scaler)
        sequence = build_last_sequences(
            scaled,
            lstm_features,
            sequence_length=int(lstm_metadata["sequence_length"]),
            rul_cap=int(lstm_metadata["rul_cap"]),
        )
        lstm = load_lstm_checkpoint(models_dir / "lstm_rul_model.pt")
        advisory_lstm = float(
            predict_lstm(
                lstm,
                sequence.values,
                rul_cap=int(lstm_metadata["rul_cap"]),
            )[0]
        )

    disagreement = (
        None if advisory_lstm is None else abs(forest_prediction - advisory_lstm)
    )
    return EngineEvidence(
        engine_id=int(engine_id),
        observed_through_cycle=int(endpoint.iloc[0]["cycle"]),
        deployment_model="random_forest",
        predicted_rul_cycles=forest_prediction,
        risk_band=classify_risk(forest_prediction),
        advisory_lstm_rul_cycles=advisory_lstm,
        model_disagreement_cycles=disagreement,
        shap_base_value_cycles=float(base_values[0]),
        shap_additivity_error=abs(raw_forest_prediction - reconstructed),
        top_feature_contributions=contributions,
        evidence_boundary=(
            "Model decision support from simulated NASA C-MAPSS data; not an "
            "airworthiness determination or an approved maintenance instruction."
        ),
    )
