"""Retrieval evaluation with labeled questions and rank-based metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from aeroguard.rag.retriever import KnowledgeIndex


def load_evaluation_cases(path: Path) -> list[dict[str, str]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Retrieval evaluation must contain a non-empty list")
    for case in cases:
        if not isinstance(case, dict) or not {"query", "expected_chunk_id"} <= case.keys():
            raise ValueError("Each evaluation case needs query and expected_chunk_id")
    return cases


def evaluate_retrieval(
    index: KnowledgeIndex,
    cases: list[dict[str, str]],
    *,
    top_k: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Measure hit rate and reciprocal rank for the expected source section."""

    rows: list[dict[str, Any]] = []
    for case in cases:
        hits = index.search(case["query"], top_k=top_k)
        identifiers = [hit.chunk_id for hit in hits]
        expected = case["expected_chunk_id"]
        rank = identifiers.index(expected) + 1 if expected in identifiers else None
        rows.append(
            {
                "query": case["query"],
                "expected_chunk_id": expected,
                "expected_rank": rank,
                f"hit_at_{top_k}": int(rank is not None),
                "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
                "retrieved_chunk_ids": " | ".join(identifiers),
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "evaluation_queries": len(frame),
        "top_k": top_k,
        f"hit_rate_at_{top_k}": float(frame[f"hit_at_{top_k}"].mean()),
        "mean_reciprocal_rank": float(frame["reciprocal_rank"].mean()),
    }
    return frame, summary
