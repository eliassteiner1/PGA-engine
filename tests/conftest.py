"""
conftest.py is loaded automatically by pytest before any test runs.
Fixtures defined here are available in every test file without importing them.
"""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random generator, so 'random' tests are reproducible. Fresh for every test."""
    return np.random.default_rng(42)
