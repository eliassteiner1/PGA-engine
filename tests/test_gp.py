"""Tests for the geometric product gp() on full multivectors."""

import numpy as np
import pytest

from pga_engine.layer1 import DIM, gp


# --- 1. algebraic properties on random multivectors ------------------------------------

def test_scalar_one_is_identity(mv, rng):
    A = rng.normal(size=DIM)
    one = mv(s=1)
    assert np.allclose(gp(one, A), A)
    assert np.allclose(gp(A, one), A)


def test_associative(rng):
    A, B, C = rng.normal(size=(3, DIM))
    assert np.allclose(gp(gp(A, B), C), gp(A, gp(B, C)))


def test_distributive(rng):
    A, B, C = rng.normal(size=(3, DIM))
    assert np.allclose(gp(A, B + C), gp(A, B) + gp(A, C))
    assert np.allclose(gp(A + B, C), gp(A, C) + gp(B, C))


def test_linear_in_scalars(rng):
    A, B = rng.normal(size=(2, DIM))
    assert np.allclose(gp(2.5 * A, B), 2.5 * gp(A, B))
    assert np.allclose(gp(A, -3 * B), -3 * gp(A, B))


def test_not_commutative(rng):
    A, B = rng.normal(size=(2, DIM))
    assert not np.allclose(gp(A, B), gp(B, A))


# --- 2. vectors -----------------------------------------------------------------------

def test_euclidean_vector_squares_to_length(mv):
    u = mv(e1=1, e2=2, e3=3)
    assert np.allclose(gp(u, u), mv(s=14))


def test_degenerate_direction_drops_out(mv):
    u = mv(e0=1, e1=1)
    assert np.allclose(gp(u, u), mv(s=1))


def test_orthogonal_vectors(mv):
    a = mv(e1=1, e2=1)
    b = mv(e1=1, e2=-1)
    assert np.allclose(gp(a, b), mv(e12=-2))
    # symmetric part = 2 * (a . b) = 0 for orthogonal vectors
    assert np.allclose(gp(a, b) + gp(b, a), 0)


def test_symmetric_part_is_scalar(mv, rng):
    a = mv(e1=rng.normal(), e2=rng.normal(), e3=rng.normal())
    b = mv(e1=rng.normal(), e2=rng.normal(), e3=rng.normal())
    sym = gp(a, b) + gp(b, a)
    assert np.allclose(sym[1:], 0)                  # nothing but scalar
    assert np.isclose(sym[0], 2 * np.dot(a, b))     # and it equals 2 a.b


# --- 3. bivectors (quaternion structure) ---------------------------------------------

def test_bivectors_anticommute(mv):
    assert np.allclose(gp(mv(e12=1), mv(e23=1)), mv(e13=1))
    assert np.allclose(gp(mv(e23=1), mv(e12=1)), mv(e13=-1))


def test_bivector_triple_product(mv):
    assert np.allclose(gp(mv(e23=1), mv(e13=1)), mv(e12=1))
    triple = gp(gp(mv(e23=1), mv(e13=1)), mv(e12=1))
    assert np.allclose(triple, mv(s=-1))


# --- 4. rotors -------------------------------------------------------------------------

def rotor(mv, theta):
    return mv(s=np.cos(theta / 2), e12=np.sin(theta / 2))


@pytest.mark.parametrize("theta", [0.0, 0.3, np.pi / 2, np.pi, 2.7])
def test_rotor_times_reverse_is_one(mv, theta):
    R = rotor(mv, theta)
    R_rev = mv(s=np.cos(theta / 2), e12=-np.sin(theta / 2))
    assert np.allclose(gp(R, R_rev), mv(s=1))


def test_rotors_compose(mv):
    alpha, beta = 0.4, 1.1
    assert np.allclose(gp(rotor(mv, alpha), rotor(mv, beta)), rotor(mv, alpha + beta))


# --- 5. pseudoscalar and e0 ------------------------------------------------------------

def test_pseudoscalar_products(mv):
    assert np.allclose(gp(mv(e0123=1), mv(e1=1)), mv(e023=1))
    assert np.allclose(gp(mv(e0123=1), mv(e0=1)), 0)
    assert np.allclose(gp(mv(e0123=1), mv(e0123=1)), 0)


def test_e123_anticommutes_with_e0(mv):
    assert np.allclose(gp(mv(e123=1), mv(e0=1)), mv(e0123=-1))
    assert np.allclose(gp(mv(e0=1), mv(e123=1)), mv(e0123=1))


@pytest.mark.parametrize("e", ["e1", "e2", "e3"])
def test_e123_commutes_with_euclidean_vectors(mv, e):
    I3, v = mv(e123=1), mv(**{e: 1})
    assert np.allclose(gp(I3, v), gp(v, I3))
