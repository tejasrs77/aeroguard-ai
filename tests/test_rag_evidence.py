from __future__ import annotations

import pytest

from aeroguard.rag.evidence import classify_risk


@pytest.mark.parametrize(
    ("rul", "expected"),
    [(0, "critical"), (15, "critical"), (15.1, "high"), (30, "high"), (60, "elevated"), (61, "routine")],
)
def test_risk_bands_have_explicit_boundaries(rul, expected) -> None:
    assert classify_risk(rul) == expected
