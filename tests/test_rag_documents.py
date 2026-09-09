from __future__ import annotations

from aeroguard.rag.documents import load_knowledge_chunks


def test_markdown_sections_become_stable_citable_chunks(tmp_path) -> None:
    (tmp_path / "guide.md").write_text(
        "# Reliability Guide\n\nIntro text.\n\n## Data Quality\n\nCheck cycles.\n",
        encoding="utf-8",
    )

    chunks = load_knowledge_chunks(tmp_path)

    assert [chunk.chunk_id for chunk in chunks] == [
        "guide#overview",
        "guide#data-quality",
    ]
    assert chunks[1].source == "guide.md"
    assert chunks[1].document_title == "Reliability Guide"
    assert len(chunks[1].sha256) == 64


def test_empty_knowledge_directory_is_rejected(tmp_path) -> None:
    try:
        load_knowledge_chunks(tmp_path)
    except FileNotFoundError as error:
        assert "No Markdown knowledge files" in str(error)
    else:
        raise AssertionError("Expected an empty knowledge directory to fail")
