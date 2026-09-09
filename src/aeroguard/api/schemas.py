"""Validated request shapes for the AeroGuard API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CopilotRequest(BaseModel):
    engine_id: int = Field(gt=0, le=100_000)
    question: str | None = Field(default=None, max_length=500)
    generator_mode: Literal["local", "openai"] = "local"
