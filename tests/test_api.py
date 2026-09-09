from __future__ import annotations

from fastapi.testclient import TestClient

from aeroguard.api import service
from aeroguard.api.main import app


client = TestClient(app)


def test_health_has_operational_headers() -> None:
    response = client.get("/api/health", headers={"X-Request-ID": "test-request"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request"
    assert float(response.headers["X-Process-Time-Ms"]) >= 0
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_dashboard_is_served() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "AeroGuard AI" in response.text
    assert "Fleet reliability control" in response.text
    for control_id in (
        "refresh-data",
        "review-critical",
        "engine-search",
        "engine-sort",
        "previous-page",
        "next-page",
        "copilot-form",
        "generate-brief",
    ):
        assert f'id="{control_id}"' in response.text


def test_engine_parameters_are_validated_before_service_call() -> None:
    assert client.get("/api/engines?page_size=51").status_code == 422
    assert client.get("/api/engines?risk_band=unknown").status_code == 422
    assert client.get("/api/engines?page=0").status_code == 422


def test_missing_engine_returns_404(monkeypatch) -> None:
    def missing_engine(_: int, **__) -> dict:
        raise ValueError("Engine 999 does not exist")

    monkeypatch.setattr(service, "engine_brief", missing_engine)
    response = client.get("/api/engines/999/brief")
    assert response.status_code == 404
    assert response.json()["detail"] == "Engine 999 does not exist"


def test_copilot_request_shape_and_service_boundary(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_brief(engine_id: int, *, question: str | None, generator_mode: str) -> dict:
        captured.update(
            engine_id=engine_id,
            question=question,
            generator_mode=generator_mode,
        )
        return {"brief": {"engine_id": engine_id}}

    monkeypatch.setattr(service, "engine_brief", fake_brief)
    response = client.post(
        "/api/copilot",
        json={"engine_id": 34, "question": "What should I review?", "generator_mode": "local"},
    )
    assert response.status_code == 200
    assert response.json()["brief"]["engine_id"] == 34
    assert captured == {
        "engine_id": 34,
        "question": "What should I review?",
        "generator_mode": "local",
    }
    assert client.post("/api/copilot", json={"engine_id": 0}).status_code == 422


def test_missing_artifact_becomes_clear_503(monkeypatch) -> None:
    def unavailable() -> dict:
        raise service.ArtifactUnavailable("Run Day 3 first")

    monkeypatch.setattr(service, "project_overview", unavailable)
    response = client.get("/api/overview")
    assert response.status_code == 503
    assert response.json() == {"detail": "Run Day 3 first"}
