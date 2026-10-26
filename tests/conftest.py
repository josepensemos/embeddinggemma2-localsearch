"""Test path setup and fixtures.

Nothing here needs the model: `search` imports torch lazily, inside Model._load().
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


@pytest.fixture
def query_vector() -> np.ndarray:
    """The query [1, 0, 0]: a stored vector's first component is its cosine with it."""
    return np.array([1.0, 0.0, 0.0], dtype="float32")
