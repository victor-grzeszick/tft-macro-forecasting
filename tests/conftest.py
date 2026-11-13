"""Shared fixtures. The tests use small synthetic data and never read the test period of the real panel."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402


@pytest.fixture
def synthetic_panel():
    """A stationary 5-variable VAR(1)-like panel on the project's 1990-2024 monthly index."""
    rng = np.random.default_rng(0)
    index = pd.date_range(config.SAMPLE_START, config.SAMPLE_END, freq="MS", name="date")
    values = np.zeros((len(index), len(config.VAR_SYSTEM)))
    for t in range(1, len(index)):
        values[t] = 0.5 * values[t - 1] + rng.normal(size=values.shape[1])
    return pd.DataFrame(values, index=index, columns=config.VAR_SYSTEM)
