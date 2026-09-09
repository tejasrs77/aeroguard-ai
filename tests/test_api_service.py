from __future__ import annotations

from pathlib import Path

import pandas as pd

from aeroguard.api.service import artifact_health, list_engines


def _write_predictions(directory: Path) -> None:
    pd.DataFrame(
        {
            "engine_id": [1, 2, 3, 4],
            "cycle": [100, 110, 120, 130],
            "actual_rul": [8, 22, 55, 92],
            "actual_rul_capped": [8.0, 22.0, 55.0, 92.0],
            "random_forest_predicted_rul": [7.0, 24.0, 52.0, 96.0],
            "lstm_predicted_rul": [5.0, 31.0, 51.0, 80.0],
        }
    ).to_csv(directory / "day3_test_predictions.csv", index=False)


def test_artifact_health_reports_missing_and_ready_components(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    rag = tmp_path / "rag"
    reports.mkdir()
    models.mkdir()
    rag.mkdir()

    degraded = artifact_health(reports_dir=reports, models_dir=models, rag_dir=rag)
    assert degraded["status"] == "degraded"
    assert degraded["ready"] is False

    for name in ("day1_summary.json", "day2_summary.json", "day3_summary.json", "day4_summary.json"):
        (reports / name).write_text("{}", encoding="utf-8")
    _write_predictions(reports)
    for name in ("champion_rul_model.joblib", "lstm_rul_model.pt"):
        (models / name).touch()
    (rag / "knowledge_index.joblib").touch()

    healthy = artifact_health(reports_dir=reports, models_dir=models, rag_dir=rag)
    assert healthy["status"] == "healthy"
    assert healthy["ready"] is True
    assert all(healthy["components"].values())


def test_engine_queue_filters_sorts_and_paginates(tmp_path: Path) -> None:
    _write_predictions(tmp_path)

    result = list_engines(
        page=1,
        page_size=2,
        sort_by="predicted_rul",
        order="asc",
        reports_dir=tmp_path,
    )
    assert [item["engine_id"] for item in result["items"]] == [1, 2]
    assert result["pagination"] == {
        "page": 1,
        "page_size": 2,
        "total_items": 4,
        "total_pages": 2,
    }

    critical = list_engines(risk_band="critical", reports_dir=tmp_path)
    assert [item["engine_id"] for item in critical["items"]] == [1]
    assert critical["items"][0]["model_disagreement"] == 2.0


def test_engine_queue_searches_natural_engine_id(tmp_path: Path) -> None:
    _write_predictions(tmp_path)
    result = list_engines(search="3", reports_dir=tmp_path)
    assert [item["engine_id"] for item in result["items"]] == [3]
