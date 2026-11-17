"""Tests for the precomputed Cayley table (CAYLEY_TABLE + CAYLEY_SIGNS)."""

import numpy as np

from pga_engine.layer1 import CAYLEY_SIGNS, CAYLEY_TABLE, DIM


def test_shape_and_dtype():
    assert CAYLEY_TABLE.shape == (DIM, DIM)
    assert CAYLEY_SIGNS.shape == (DIM, DIM)
    # entries are used as array indices, so they must be integers
    assert np.issubdtype(CAYLEY_TABLE.dtype, np.integer)
    assert np.issubdtype(CAYLEY_SIGNS.dtype, np.integer)


def test_value_ranges():
    assert CAYLEY_TABLE.min() >= 0 and CAYLEY_TABLE.max() < DIM
    assert set(np.unique(CAYLEY_SIGNS)) <= {-1, 0, 1}


def test_scalar_is_identity():
    # row 0 and column 0: 1 * X = X * 1 = X
    assert np.array_equal(CAYLEY_TABLE[0, :], np.arange(DIM))
    assert np.array_equal(CAYLEY_TABLE[:, 0], np.arange(DIM))
    assert np.all(CAYLEY_SIGNS[0, :] == 1)
    assert np.all(CAYLEY_SIGNS[:, 0] == 1)


def test_squares_of_basis_blades():
    # order: 1, e0 e1 e2 e3, e01 e02 e03 e12 e13 e23, e012 e013 e023 e123, e0123
    expected = [1, 0, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, -1, 0]
    assert CAYLEY_SIGNS.diagonal().tolist() == expected


def test_nonzero_squares_give_scalar():
    nonzero = CAYLEY_SIGNS.diagonal() != 0
    assert np.all(CAYLEY_TABLE.diagonal()[nonzero] == 0)


def test_associativity():
    """(i*j)*k == i*(j*k) for all 16^3 triples of basis blades."""

    def mul(i, j):
        return CAYLEY_SIGNS[i, j], CAYLEY_TABLE[i, j]

    failures = []
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                s1, m = mul(i, j)
                s2, left_idx = mul(m, k)
                s3, n = mul(j, k)
                s4, right_idx = mul(i, n)
                left_sign, right_sign = s1 * s2, s3 * s4
                # all zero results are equal, no matter which dummy index they carry
                left = (left_sign, left_idx if left_sign else -1)
                right = (right_sign, right_idx if right_sign else -1)
                if left != right:
                    failures.append((i, j, k))

    assert failures == []
