"""Small, inspectable TF-IDF retriever for the local reliability handbook."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from aeroguard.rag.documents import KnowledgeChunk


@dataclass(frozen=True)
class RetrievalHit:
    """One ranked knowledge result and its lexical similarity score."""

    rank: int
    score: float
    chunk_id: str
    source: str
    document_title: str
    section: str
    text: str

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class KnowledgeIndex:
    """Fit and search a deterministic local TF-IDF matrix."""

    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        if not chunks:
            raise ValueError("At least one knowledge chunk is required")
        self.chunks = list(chunks)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix: csr_matrix = self.vectorizer.fit_transform(
            chunk.text for chunk in chunks
        ).tocsr()

    def search(self, query: str, *, top_k: int = 4) -> list[RetrievalHit]:
        if not query or not query.strip():
            raise ValueError("Retrieval query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        ordered = np.argsort(-scores, kind="stable")[: min(top_k, len(self.chunks))]
        return [
            RetrievalHit(
                rank=rank,
                score=float(scores[position]),
                chunk_id=self.chunks[position].chunk_id,
                source=self.chunks[position].source,
                document_title=self.chunks[position].document_title,
                section=self.chunks[position].section,
                text=self.chunks[position].text,
            )
            for rank, position in enumerate(ordered, start=1)
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "KnowledgeIndex":
        index = joblib.load(path)
        if not isinstance(index, cls):
            raise TypeError(f"Unexpected object in knowledge index: {type(index)!r}")
        return index
