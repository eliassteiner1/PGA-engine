"""Tests for the construction helpers in utils.py."""

import numpy as np
import pytest

from pga_engine.algebra import PGA3D
from pga_engine.utils import blade, mv, name_to_enum


def test_names_cover_every_basis_blade():
    names = ["s", "e0", "e1", "e2", "e3", "e01", "e02", "e03", "e12", "e13", "e23",
             "e012", "e013", "e023", "e123", "e0123"]
    assert [name_to_enum(n) for n in names] == list(range(PGA3D.dim))


def test_mv_places_coefficients():
    A = mv(s=1, e12=0.5, e0123=-2)
    expected = np.zeros(16)
    expected[[0, 8, 15]] = [1, 0.5, -2]
    assert np.array_equal(A, expected)


# pytest.raises: the test passes only if the code inside raises that exception
@pytest.mark.parametrize("indices", [(2, 1), (1, 1), (4,)])   # unsorted, duplicate, out of range
def test_blade_rejects_invalid_indices(indices):
    with pytest.raises(ValueError):
        blade(*indices)


@pytest.mark.parametrize("name", ["x12", "e"])   # not 'e' + digits
def test_mv_rejects_invalid_names(name):
    with pytest.raises(ValueError):
        mv(**{name: 1})
