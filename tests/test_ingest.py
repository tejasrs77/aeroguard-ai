from pathlib import Path

import pytest

from aeroguard.data.ingest import read_cmapss, read_rul_truth


def test_read_cmapss_parses_whitespace_file(tmp_path: Path) -> None:
    values = [1, 1, 0.1, 0.2, 100.0, *range(1, 22)]
    source = tmp_path / "train.txt"
    source.write_text("   ".join(map(str, values)) + "\n", encoding="utf-8")

    frame = read_cmapss(source)

    assert frame.shape == (1, 26)
    assert frame.loc[0, "engine_id"] == 1
    assert frame.loc[0, "sensor_21"] == 21


def test_read_cmapss_rejects_non_numeric_value(tmp_path: Path) -> None:
    values = [1, 1, 0.1, 0.2, 100.0, *range(1, 21), "broken"]
    source = tmp_path / "train.txt"
    source.write_text(" ".join(map(str, values)) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing or non-numeric"):
        read_cmapss(source)


def test_read_rul_truth(tmp_path: Path) -> None:
    source = tmp_path / "truth.txt"
    source.write_text("12\n31\n", encoding="utf-8")

    truth = read_rul_truth(source)

    assert truth.tolist() == [12, 31]

