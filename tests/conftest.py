"""
conftest.py is loaded automatically by pytest before any test runs.
Fixtures defined here are available in every test file without importing them.
"""

import numpy as np
import pytest

from pga_engine.layer1 import BLADE_2_ENUM, DIM


def name_to_idx(name: str) -> int:
    """'e12' -> index of e12 in the multivector array. 's' is the scalar."""
    if name == "s":
        return 0
    bitmask = sum(1 << int(k) for k in name[1:])
    return BLADE_2_ENUM[bitmask]


def make_mv(**coeffs: float) -> np.ndarray:
    """mv(s=1, e12=0.5) -> 16-element coefficient array."""
    out = np.zeros(DIM)
    for name, value in coeffs.items():
        out[name_to_idx(name)] = value
    return out


@pytest.fixture
def mv():
    """Factory fixture: tests receive the make_mv function as the argument `mv`."""
    return make_mv


@pytest.fixture
def rng():
    """Seeded random generator, so 'random' tests are reproducible."""
    return np.random.default_rng(42)
