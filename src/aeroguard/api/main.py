"""FastAPI application serving real AeroGuard artifacts and its dashboard."""

from __future__ import annotations

import os
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from aeroguard.api import service
from aeroguard.api.schemas import CopilotRequest
from aeroguard.config import DASHBOARD_DIR, FIGURES_DIR


app = FastAPI(
    title="AeroGuard AI API",
    description="Predictive-maintenance evidence and grounded reliability briefs.",
    version="1.0.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "AEROGUARD_CORS_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def request_metadata(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))[:64]
    started = perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{(perf_counter() - started) * 1000:.2f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.exception_handler(service.ArtifactUnavailable)
async def artifact_unavailable(_: Request, error: service.ArtifactUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.get("/api/health", tags=["system"])
def health() -> dict:
    return service.artifact_health()


@app.get("/api/overview", tags=["dashboard"])
def overview() -> dict:
    return service.project_overview()


@app.get("/api/models", tags=["dashboard"])
def models() -> list[dict]:
    return service.model_metrics()


@app.get("/api/retrieval", tags=["dashboard"])
def retrieval() -> dict:
    return service.retrieval_quality()


@app.get("/api/engines", tags=["engines"])
def engines(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    risk_band: str | None = Query(None, pattern="^(critical|high|elevated|routine)$"),
    search: str | None = Query(None, max_length=20),
    sort_by: str = Query("predicted_rul", pattern="^(engine_id|predicted_rul|disagreement)$"),
    order: str = Query("asc", pattern="^(asc|desc)$"),
) -> dict:
    return service.list_engines(
        page=page,
        page_size=page_size,
        risk_band=risk_band,
        search=search,
        sort_by=sort_by,
        order=order,
    )


@app.get("/api/engines/{engine_id}/brief", tags=["engines"])
def get_engine_brief(engine_id: int = Path(..., gt=0)) -> dict:
    try:
        return service.engine_brief(engine_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/copilot", tags=["copilot"])
def copilot(request: CopilotRequest) -> dict:
    try:
        return service.engine_brief(
            request.engine_id,
            question=request.question,
            generator_mode=request.generator_mode,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")
app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
