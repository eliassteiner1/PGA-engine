"""Tests for the Algebra dataclass and its precomputed lookup tables."""

import dataclasses

import numpy as np
import pytest

from pga_engine.algebra import PGA3D, Algebra

# a few signatures to check that nothing silently assumes 3D PGA
METRICS = [
    (0, 1, 1, 1),     # 3D PGA
    (1, 1, 1),        # 3D Euclidean (VGA)
    (1, 1, 1, -1),    # spacetime-like, has a negative square
    (0, 1, 1, 1, 1),  # 4D PGA, 5 unit vectors
]


@pytest.fixture(params=METRICS, ids=str, scope="module")
def alg(request):
    """Runs a test once per metric in METRICS (built once per metric, Algebra is immutable)."""
    return Algebra.from_metric(request.param)


# --- 1. PGA3D specifics -----------------------------------------------------------------

def test_pga3d_blade_order():
    # 1, e0 e1 e2 e3, e01 e02 e03 e12 e13 e23, e012 e013 e023 e123, e0123
    expected = [0b0000,
                0b0001, 0b0010, 0b0100, 0b1000,
                0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100,
                0b0111, 0b1011, 0b1101, 0b1110,
                0b1111]
    assert list(PGA3D.blades) == expected


def test_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        PGA3D.metric = (1, 1, 1)
    with pytest.raises(ValueError):
        PGA3D.gp_table[0, 0, 0] = 5


# --- 2. structural properties, for every metric -----------------------------------------

def test_sizes(alg):
    n = len(alg.metric)
    assert alg.nr_unitvecs == n
    assert alg.dim == 2 ** n
    assert alg.gp_table.shape == (alg.dim,) * 3
    assert sorted(alg.blades) == list(range(alg.dim))  # every subset exactly once


def test_blades_sorted_by_grade(alg):
    assert np.all(np.diff(alg.grade) >= 0)
    assert alg.grade[0] == 0 and alg.grade[-1] == alg.nr_unitvecs


def test_table_values(alg):
    assert set(np.unique(alg.gp_table)) <= {-1, 0, 1}
    # blade i times blade j gives at most one blade k
    assert np.all(np.count_nonzero(alg.gp_table, axis=2) <= 1)


def test_scalar_is_identity(alg):
    eye = np.eye(alg.dim)
    assert np.array_equal(alg.gp_table[0], eye)      # 1 * X = X
    assert np.array_equal(alg.gp_table[:, 0], eye)   # X * 1 = X


def test_vector_squares_are_metric(alg):
    for k, m in enumerate(alg.metric):
        i = alg.blade_index[1 << k]
        assert alg.gp_table[i, i, 0] == m


def test_associativity(alg):
    """(ij)k == i(jk) for all triples of basis blades, written as two tensor contractions."""
    T = alg.gp_table
    left = np.einsum("ijm, mkn -> ijkn", T, T)   # (blade_i blade_j) blade_k
    right = np.einsum("jkm, imn -> ijkn", T, T)  # blade_i (blade_j blade_k)
    assert np.array_equal(left, right)
