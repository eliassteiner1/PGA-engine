"""Tests for _resolve_bb_mult_bitmask: the product of two single basis blades."""

import pytest

from pga_engine.algebra import PGA3D, _bb_bitmask_resolve_multiplication
from pga_engine.utils import blade as b  # short alias keeps the tables below readable

PGA = PGA3D.metric  # (0, 1, 1, 1)


# Each row: (left blade, right blade, expected sign, expected result blade).
# Every row becomes its own test case, so a failure tells you exactly which one broke.
@pytest.mark.parametrize(
    "bb1, bb2, sign, result",
    [
        (b(1), b(2), 1, b(1, 2)),                # e1 e2 = e12   (already sorted)
        (b(2), b(1), -1, b(1, 2)),               # e2 e1 = -e12  (one swap)
        (b(1), b(1), 1, b()),                    # e1 e1 = 1
        (b(1, 2), b(1, 2), -1, b()),             # e12 e12 = -1
        (b(1, 2, 3), b(1, 2, 3), -1, b()),       # e123 e123 = -1
        (b(0, 1, 2, 3), b(1, 2), -1, b(0, 3)),   # e0123 e12 = -e03
        (b(1, 2, 3), b(0), -1, b(0, 1, 2, 3)),   # e123 e0 = -e0123
        (b(0), b(1, 2, 3), 1, b(0, 1, 2, 3)),    # e0 e123 = e0123
    ],
)
def test_known_products(bb1, bb2, sign, result):
    assert _bb_bitmask_resolve_multiplication(bb1, bb2, PGA) == (sign, result)


@pytest.mark.parametrize(
    "bb1, bb2",
    [
        (b(0), b(0)),          # e0 e0 = 0
        (b(0, 2), b(0, 1, 3)), # any shared e0 kills the product
    ],
)
def test_shared_e0_is_zero(bb1, bb2):
    sign, _ = _bb_bitmask_resolve_multiplication(bb1, bb2, PGA)
    assert sign == 0


def test_distinct_vectors_anticommute():
    for i in range(4):
        for j in range(4):
            if i != j:
                s_ij, r_ij = _bb_bitmask_resolve_multiplication(b(i), b(j), PGA)
                s_ji, r_ji = _bb_bitmask_resolve_multiplication(b(j), b(i), PGA)
                assert r_ij == r_ji
                assert s_ij == -s_ji


# --- other metrics ------------------------------------------------------------------------

def test_all_shared_vectors_contribute():
    # e12 e12 = -e1 e1 e2 e2: one swap, then metric[1] * metric[2]
    assert _bb_bitmask_resolve_multiplication(b(1, 2), b(1, 2), (1, -1, -1, 1)) == (-1, b())
    assert _bb_bitmask_resolve_multiplication(b(1, 2), b(1, 2), (1, -1, 1, 1)) == (1, b())


def test_metric_longer_than_four():
    # e4 only exists in a 5D algebra: guards against code that silently assumes 4 unit vectors.
    # blade() only knows PGA3D's 4 unit vectors, so raw bitmasks are used here.
    e4 = 1 << 4
    assert _bb_bitmask_resolve_multiplication(e4, e4, (0, 1, 1, 1, -1)) == (-1, 0)
