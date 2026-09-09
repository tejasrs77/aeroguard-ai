from __future__ import annotations

from aeroguard.rag.documents import KnowledgeChunk
from aeroguard.rag.evaluation import evaluate_retrieval
from aeroguard.rag.retriever import KnowledgeIndex


def _chunk(identifier: str, text: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=identifier,
        source=f"{identifier}.md",
        document_title=identifier.title(),
        section="Section",
        text=text,
        sha256="0" * 64,
    )


def test_retriever_ranks_lexically_relevant_source_first(tmp_path) -> None:
    index = KnowledgeIndex(
        [
            _chunk("quality", "Validate missing sensor values and cycle order."),
            _chunk("risk", "Late warning overestimates remaining useful life."),
        ]
    )

    hits = index.search("How do I check missing sensor data?", top_k=1)
    index.save(tmp_path / "index.joblib")
    restored = KnowledgeIndex.load(tmp_path / "index.joblib")

    assert hits[0].chunk_id == "quality"
    assert hits[0].score > 0
    assert restored.search("late RUL warning", top_k=1)[0].chunk_id == "risk"


def test_retrieval_evaluation_calculates_hit_rate_and_mrr() -> None:
    index = KnowledgeIndex(
        [
            _chunk("quality", "Validate missing values and duplicates."),
            _chunk("authority", "Qualified humans make maintenance decisions."),
        ]
    )
    cases = [
        {"query": "Who makes maintenance decisions?", "expected_chunk_id": "authority"},
        {"query": "How are duplicates validated?", "expected_chunk_id": "quality"},
    ]

    frame, summary = evaluate_retrieval(index, cases, top_k=1)

    assert frame["hit_at_1"].tolist() == [1, 1]
    assert summary["hit_rate_at_1"] == 1.0
    assert summary["mean_reciprocal_rank"] == 1.0
