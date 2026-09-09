from __future__ import annotations

import pandas as pd
import pytest

from aeroguard.config import CMAPSS_COLUMNS


@pytest.fixture
def sensor_frame() -> pd.DataFrame:
    rows = []
    for engine_id, cycles in ((1, 3), (2, 2)):
        for cycle in range(1, cycles + 1):
            values = [engine_id, cycle, 0.1, 0.2, 100.0]
            values.extend(float(number + cycle) for number in range(1, 22))
            rows.append(values)
    return pd.DataFrame(rows, columns=CMAPSS_COLUMNS)

