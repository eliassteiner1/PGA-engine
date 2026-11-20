"""Tests for the geometric product gp() on full multivectors."""

import numpy as np
import pytest

from pga_engine.algebra import PGA3D, Algebra
from pga_engine.products import gp
from pga_engine.utils import mv


# --- 1. implementation: loop reference, batching, other algebras --------------------------
# (identity, associativity etc. are already checked on the table itself in test_algebra.py)

def test_batched(rng):
    A, B = rng.normal(size=(2, 7, PGA3D.dim))   # 7 pairs of multivectors
    out = gp(A, B)
    assert out.shape == (7, PGA3D.dim)
    assert np.allclose(out, [gp(a, b) for a, b in zip(A, B)])
    # a batch against a single multivector broadcasts
    assert np.allclose(gp(A, B[0]), [gp(a, B[0]) for a in A])


def test_other_algebra():
    vga = Algebra.from_metric((1, 1, 1))
    # (1, 1, 1) has unit vectors e0, e1, e2: its bivectors behave like quaternions
    i, j, k = mv(alg=vga, e01=1), mv(alg=vga, e02=1), mv(alg=vga, e12=1)
    for q in (i, j, k):
        assert np.allclose(gp(q, q, alg=vga), mv(alg=vga, s=-1))


# --- 2. vectors ------------------------------------------------------------------------

def test_euclidean_vector_squares_to_length():
    u = mv(e1=1, e2=2, e3=3)
    assert np.allclose(gp(u, u), mv(s=14))


def test_degenerate_direction_drops_out():
    u = mv(e0=1, e1=1)
    assert np.allclose(gp(u, u), mv(s=1))


def test_symmetric_part_is_scalar(rng):
    a = mv(e1=rng.normal(), e2=rng.normal(), e3=rng.normal())
    b = mv(e1=rng.normal(), e2=rng.normal(), e3=rng.normal())
    sym = gp(a, b) + gp(b, a)
    assert np.allclose(sym[1:], 0)                  # nothing but scalar
    assert np.isclose(sym[0], 2 * np.dot(a, b))     # and it equals 2 a.b


# --- 3. bivectors (quaternion structure) ----------------------------------------------

def test_bivector_triple_product():
    assert np.allclose(gp(mv(e23=1), mv(e13=1)), mv(e12=1))
    triple = gp(gp(mv(e23=1), mv(e13=1)), mv(e12=1))
    assert np.allclose(triple, mv(s=-1))


# --- 4. rotors --------------------------------------------------------------------------

def rotor(theta):
    return mv(s=np.cos(theta / 2), e12=np.sin(theta / 2))


@pytest.mark.parametrize("theta", [0.3, np.pi / 2, 2.7])
def test_rotor_times_reverse_is_one(theta):
    R = rotor(theta)
    R_rev = mv(s=np.cos(theta / 2), e12=-np.sin(theta / 2))
    assert np.allclose(gp(R, R_rev), mv(s=1))


def test_rotors_compose():
    alpha, beta = 0.4, 1.1
    assert np.allclose(gp(rotor(alpha), rotor(beta)), rotor(alpha + beta))


# --- 5. pseudoscalar --------------------------------------------------------------------

def test_pseudoscalar_products():
    assert np.allclose(gp(mv(e0123=1), mv(e1=1)), mv(e023=1))
    assert np.allclose(gp(mv(e0123=1), mv(e0=1)), 0)
    assert np.allclose(gp(mv(e0123=1), mv(e0123=1)), 0)
