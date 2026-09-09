"""Create the final Day 5 operational-readiness report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aeroguard.api import service
from aeroguard.config import REPORTS_DIR


API_ENDPOINTS = [
    "GET /api/health",
    "GET /api/overview",
    "GET /api/models",
    "GET /api/retrieval",
    "GET /api/engines",
    "GET /api/engines/{engine_id}/brief",
    "POST /api/copilot",
]


def build_day5_summary(*, reports_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    """Verify prior artifacts and persist a truthful dashboard snapshot."""

    health = service.artifact_health(reports_dir=reports_dir)
    if not health["ready"]:
        missing = [name for name, ready in health["components"].items() if not ready]
        raise service.ArtifactUnavailable(
            "Day 5 needs the completed Day 1-4 artifacts. Missing: " + ", ".join(missing)
        )

    overview = service.project_overview(reports_dir=reports_dir)
    engine_queue = service.list_engines(
        page=1,
        page_size=10,
        sort_by="predicted_rul",
        order="asc",
        reports_dir=reports_dir,
    )
    retrieval = service.retrieval_quality(reports_dir=reports_dir)
    models = service.model_metrics(reports_dir=reports_dir)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready",
        "application": "AeroGuard AI Operations Console",
        "api_endpoints": API_ENDPOINTS,
        "dashboard_data_source": "Generated Day 1-4 artifacts served through FastAPI",
        "monitored_engines": overview["fleet"]["monitored_engines"],
        "highest_priority_engine": overview["fleet"]["highest_priority_engine"],
        "risk_counts": overview["fleet"]["risk_counts"],
        "deployment_champion": overview["model"]["deployment_champion"],
        "validation_accuracy_winner": overview["model"]["validation_accuracy_winner"],
        "retrieval_hit_rate_at_3": retrieval["summary"]["hit_rate_at_3"],
        "model_metric_rows": len(models),
        "priority_queue_preview": engine_queue["items"],
        "safety_controls": [
            "The API returns model predictions from persisted artifacts, not the LLM.",
            "The local generator is the default; OpenAI generation is explicit opt-in.",
            "Citations are limited to retrieved knowledge-base chunks.",
            "The dashboard is decision support and not an airworthiness system.",
        ],
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    destination = reports_dir / "day5_summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return summary
