"""Tests for resolve_bb_mult_bitmask: the product of two single basis blades."""

import pytest

from pga_engine.layer1 import resolve_bb_mult_bitmask
from pga_engine.utils import blade as b  # short alias keeps the tables below readable


# Each row: (left blade, right blade, expected sign, expected result blade).
# Every row becomes its own test case, so a failure tells you exactly which one broke.
@pytest.mark.parametrize(
    "bb1, bb2, sign, result",
    [
        (b(), b(), 1, b()),                      # 1 * 1 = 1
        (b(), b(1, 2), 1, b(1, 2)),              # 1 * e12 = e12
        (b(1), b(2), 1, b(1, 2)),                # e1 e2 = e12   (already sorted)
        (b(2), b(1), -1, b(1, 2)),               # e2 e1 = -e12  (one swap)
        (b(1), b(1), 1, b()),                    # e1 e1 = 1
        (b(3), b(3), 1, b()),                    # e3 e3 = 1
        (b(1, 2), b(1, 2), -1, b()),             # e12 e12 = -1
        (b(1, 2, 3), b(1, 2, 3), -1, b()),       # e123 e123 = -1
        (b(0, 1, 2, 3), b(1, 2), -1, b(0, 3)),   # e0123 e12 = -e03
        (b(0, 1, 2, 3), b(1), 1, b(0, 2, 3)),    # e0123 e1 = e023
        (b(1, 2, 3), b(0), -1, b(0, 1, 2, 3)),   # e123 e0 = -e0123
        (b(0), b(1, 2, 3), 1, b(0, 1, 2, 3)),    # e0 e123 = e0123
    ],
)
def test_known_products(bb1, bb2, sign, result):
    assert resolve_bb_mult_bitmask(bb1, bb2) == (sign, result)


@pytest.mark.parametrize(
    "bb1, bb2",
    [
        (b(0), b(0)),                  # e0 e0 = 0
        (b(0, 1), b(0, 1)),            # e01 e01 = 0
        (b(0, 1, 2, 3), b(0, 1, 2, 3)),  # I I = 0
        (b(0, 2), b(0, 1, 3)),         # any shared e0 kills the product
    ],
)
def test_shared_e0_is_zero(bb1, bb2):
    sign, _ = resolve_bb_mult_bitmask(bb1, bb2)
    assert sign == 0


def test_distinct_vectors_anticommute():
    for i in range(4):
        for j in range(4):
            if i != j:
                s_ij, r_ij = resolve_bb_mult_bitmask(b(i), b(j))
                s_ji, r_ji = resolve_bb_mult_bitmask(b(j), b(i))
                assert r_ij == r_ji
                assert s_ij == -s_ji
