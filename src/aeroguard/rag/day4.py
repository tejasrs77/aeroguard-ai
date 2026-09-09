"""Orchestrate the grounded Day 4 retrieval and generation workflow."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from aeroguard.config import KNOWLEDGE_DIR, RAG_DIR, REPORTS_DIR
from aeroguard.rag.documents import load_knowledge_chunks
from aeroguard.rag.evaluation import evaluate_retrieval, load_evaluation_cases
from aeroguard.rag.evidence import build_engine_evidence
from aeroguard.rag.generator import build_grounded_prompt, generate_brief
from aeroguard.rag.retriever import KnowledgeIndex


def _knowledge_fingerprint(checksums: list[str]) -> str:
    return hashlib.sha256("".join(checksums).encode("utf-8")).hexdigest()


def run_day4_copilot(
    *,
    engine_id: int | None = None,
    generator_mode: str | None = None,
    knowledge_dir: Path = KNOWLEDGE_DIR,
    rag_dir: Path = RAG_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Index sources, evaluate retrieval, and generate one real engine brief."""

    rag_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    chunks = load_knowledge_chunks(knowledge_dir)
    index = KnowledgeIndex(chunks)
    index_path = rag_dir / "knowledge_index.joblib"
    index.save(index_path)

    manifest = {
        "index_type": "TF-IDF word unigram and bigram cosine similarity",
        "knowledge_documents": len({chunk.source for chunk in chunks}),
        "knowledge_chunks": len(chunks),
        "knowledge_sha256": _knowledge_fingerprint(
            [chunk.sha256 for chunk in chunks]
        ),
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    (rag_dir / "knowledge_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    cases = load_evaluation_cases(knowledge_dir / "evaluation_queries.json")
    evaluation, evaluation_summary = evaluate_retrieval(index, cases, top_k=3)
    evaluation.to_csv(reports_dir / "day4_retrieval_evaluation.csv", index=False)

    if engine_id is None:
        day3_summary = json.loads(
            (reports_dir / "day3_summary.json").read_text(encoding="utf-8")
        )
        engine_id = int(day3_summary["high_risk_engine_explained"])
    evidence = build_engine_evidence(engine_id)
    feature_names = ", ".join(
        item.feature for item in evidence.top_feature_contributions
    )
    query = (
        f"How should a reliability engineer triage a {evidence.risk_band} engine "
        f"with predicted RUL {evidence.predicted_rul_cycles:.1f}, model disagreement, "
        f"and influential signals {feature_names}? Include data quality, SHAP limits, "
        "human authority, and late-warning risk."
    )
    hits = index.search(query, top_k=4)
    brief = generate_brief(evidence, hits, mode=generator_mode)
    prompt = build_grounded_prompt(evidence, hits)

    brief_payload = {
        "brief": brief.to_dict(),
        "verified_evidence": evidence.to_dict(),
        "retrieval_query": query,
        "retrieved_sources": [hit.to_dict() for hit in hits],
    }
    (reports_dir / "day4_reliability_brief.json").write_text(
        json.dumps(brief_payload, indent=2), encoding="utf-8"
    )
    (rag_dir / "last_grounded_prompt.json").write_text(prompt, encoding="utf-8")

    summary: dict[str, Any] = {
        "knowledge_documents": manifest["knowledge_documents"],
        "knowledge_chunks": manifest["knowledge_chunks"],
        "knowledge_sha256": manifest["knowledge_sha256"],
        **evaluation_summary,
        "engine_id": evidence.engine_id,
        "observed_through_cycle": evidence.observed_through_cycle,
        "deployment_model": evidence.deployment_model,
        "predicted_rul_cycles": evidence.predicted_rul_cycles,
        "advisory_lstm_rul_cycles": evidence.advisory_lstm_rul_cycles,
        "risk_band": evidence.risk_band,
        "shap_additivity_error": evidence.shap_additivity_error,
        "retrieved_citations": list(brief.citations),
        "generator_provider": brief.provider,
        "openai_enabled": brief.provider.startswith("openai_responses:"),
        "rul_source_of_truth": "persisted random_forest model",
        "generator_can_modify_rul": False,
        "index_artifact": str(index_path),
    }
    (reports_dir / "day4_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
