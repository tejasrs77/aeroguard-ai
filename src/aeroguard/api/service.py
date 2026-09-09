"""Read generated artifacts and expose stable dashboard-ready records."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from aeroguard.config import (
    KNOWLEDGE_DIR,
    MODELS_DIR,
    RAG_DIR,
    REPORTS_DIR,
)
from aeroguard.rag.documents import load_knowledge_chunks
from aeroguard.rag.evidence import build_engine_evidence, classify_risk
from aeroguard.rag.generator import build_grounded_prompt, generate_brief
from aeroguard.rag.retriever import KnowledgeIndex


class ArtifactUnavailable(RuntimeError):
    """Raised when an endpoint needs a generated artifact that is absent."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtifactUnavailable(f"Required artifact is missing: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_health(
    *,
    reports_dir: Path = REPORTS_DIR,
    models_dir: Path = MODELS_DIR,
    rag_dir: Path = RAG_DIR,
) -> dict[str, Any]:
    required = {
        "day1_report": reports_dir / "day1_summary.json",
        "day2_report": reports_dir / "day2_summary.json",
        "day3_report": reports_dir / "day3_summary.json",
        "day4_report": reports_dir / "day4_summary.json",
        "test_predictions": reports_dir / "day3_test_predictions.csv",
        "deployment_model": models_dir / "champion_rul_model.joblib",
        "lstm_model": models_dir / "lstm_rul_model.pt",
        "knowledge_index": rag_dir / "knowledge_index.joblib",
    }
    components = {name: path.exists() for name, path in required.items()}
    return {
        "status": "healthy" if all(components.values()) else "degraded",
        "service": "aeroguard-api",
        "components": components,
        "ready": all(components.values()),
    }


def _engine_frame(reports_dir: Path = REPORTS_DIR) -> pd.DataFrame:
    path = reports_dir / "day3_test_predictions.csv"
    if not path.exists():
        raise ArtifactUnavailable("Run Day 3 to create test predictions")
    frame = pd.read_csv(path)
    frame["risk_band"] = frame["random_forest_predicted_rul"].map(classify_risk)
    frame["model_disagreement_cycles"] = (
        frame["random_forest_predicted_rul"] - frame["lstm_predicted_rul"]
    ).abs()
    return frame


def project_overview(*, reports_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    day1 = _read_json(reports_dir / "day1_summary.json")
    day2 = _read_json(reports_dir / "day2_summary.json")
    day3 = _read_json(reports_dir / "day3_summary.json")
    day4 = _read_json(reports_dir / "day4_summary.json")
    engines = _engine_frame(reports_dir)
    risk_order = ["critical", "high", "elevated", "routine"]
    risk_counts = engines["risk_band"].value_counts().to_dict()
    lowest = engines.sort_values("random_forest_predicted_rul").iloc[0]
    return {
        "dataset": {
            "name": "NASA C-MAPSS FD001",
            "training_rows": day1["training_rows"],
            "training_engines": day1["training_engines"],
            "test_engines": day2["official_test_engines"],
        },
        "fleet": {
            "monitored_engines": int(len(engines)),
            "risk_counts": {band: int(risk_counts.get(band, 0)) for band in risk_order},
            "average_deployment_rul": float(engines["random_forest_predicted_rul"].mean()),
            "highest_priority_engine": int(lowest["engine_id"]),
            "lowest_predicted_rul": float(lowest["random_forest_predicted_rul"]),
        },
        "model": {
            "deployment_champion": day3["deployment_champion"],
            "validation_accuracy_winner": day3["validation_winner"],
            "random_forest_test_rmse": day3["official_test_metrics_capped"]["random_forest"]["rmse"],
            "lstm_test_rmse": day3["official_test_metrics_capped"]["lstm"]["rmse"],
            "lstm_promotion_status": day3["lstm_promotion_status"],
        },
        "retrieval": {
            "knowledge_documents": day4["knowledge_documents"],
            "knowledge_chunks": day4["knowledge_chunks"],
            "hit_rate_at_3": day4["hit_rate_at_3"],
            "mean_reciprocal_rank": day4["mean_reciprocal_rank"],
            "generator_provider": day4["generator_provider"],
        },
    }


def model_metrics(*, reports_dir: Path = REPORTS_DIR) -> list[dict[str, Any]]:
    path = reports_dir / "day3_metrics.csv"
    if not path.exists():
        raise ArtifactUnavailable("Run Day 3 to create model metrics")
    return pd.read_csv(path).to_dict(orient="records")


def retrieval_quality(*, reports_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    summary = _read_json(reports_dir / "day4_summary.json")
    path = reports_dir / "day4_retrieval_evaluation.csv"
    if not path.exists():
        raise ArtifactUnavailable("Run Day 4 to create retrieval evaluation")
    return {
        "summary": {
            "knowledge_documents": summary["knowledge_documents"],
            "knowledge_chunks": summary["knowledge_chunks"],
            "evaluation_queries": summary["evaluation_queries"],
            "hit_rate_at_3": summary["hit_rate_at_3"],
            "mean_reciprocal_rank": summary["mean_reciprocal_rank"],
            "knowledge_sha256": summary["knowledge_sha256"],
        },
        "cases": pd.read_csv(path).to_dict(orient="records"),
    }


def list_engines(
    *,
    page: int = 1,
    page_size: int = 10,
    risk_band: str | None = None,
    search: str | None = None,
    sort_by: Literal["engine_id", "predicted_rul", "disagreement"] = "predicted_rul",
    order: Literal["asc", "desc"] = "asc",
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    frame = _engine_frame(reports_dir)
    if risk_band:
        frame = frame.loc[frame["risk_band"] == risk_band]
    if search:
        normalized = search.strip()
        if normalized:
            frame = frame.loc[frame["engine_id"].astype(str).str.contains(normalized, regex=False)]
    sort_columns = {
        "engine_id": "engine_id",
        "predicted_rul": "random_forest_predicted_rul",
        "disagreement": "model_disagreement_cycles",
    }
    frame = frame.sort_values(sort_columns[sort_by], ascending=order == "asc")
    total_items = len(frame)
    total_pages = max(1, math.ceil(total_items / page_size))
    start = (page - 1) * page_size
    selected = frame.iloc[start : start + page_size]
    items = [
        {
            "engine_id": int(row.engine_id),
            "observed_cycle": int(row.cycle),
            "predicted_rul": float(row.random_forest_predicted_rul),
            "advisory_lstm_rul": float(row.lstm_predicted_rul),
            "benchmark_actual_rul": float(row.actual_rul_capped),
            "model_disagreement": float(row.model_disagreement_cycles),
            "risk_band": str(row.risk_band),
        }
        for row in selected.itertuples(index=False)
    ]
    return {
        "items": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
        },
    }


def _knowledge_index() -> KnowledgeIndex:
    index_path = RAG_DIR / "knowledge_index.joblib"
    if index_path.exists():
        return KnowledgeIndex.load(index_path)
    return KnowledgeIndex(load_knowledge_chunks(KNOWLEDGE_DIR))


def engine_brief(
    engine_id: int,
    *,
    question: str | None = None,
    generator_mode: str = "local",
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    precomputed = reports_dir / "day4_reliability_brief.json"
    if question is None and generator_mode == "local" and precomputed.exists():
        payload = _read_json(precomputed)
        if int(payload["brief"]["engine_id"]) == engine_id:
            return payload

    evidence = build_engine_evidence(engine_id)
    feature_names = ", ".join(
        item.feature for item in evidence.top_feature_contributions
    )
    query = question or (
        f"How should a reliability engineer review a {evidence.risk_band} engine "
        f"with predicted RUL {evidence.predicted_rul_cycles:.1f}, disagreement "
        f"between models, and influential features {feature_names}?"
    )
    index = _knowledge_index()
    hits = index.search(query, top_k=4)
    brief = generate_brief(evidence, hits, mode=generator_mode)
    return {
        "brief": brief.to_dict(),
        "verified_evidence": evidence.to_dict(),
        "retrieval_query": query,
        "retrieved_sources": [hit.to_dict() for hit in hits],
        "grounded_prompt": build_grounded_prompt(evidence, hits),
    }
