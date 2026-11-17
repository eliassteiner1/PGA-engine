"""Tests for the geometric product gp() on full multivectors."""

import numpy as np
import pytest

from pga_engine.layer1 import DIM, _gp_loop
from pga_engine.utils import mv


# --- 1. algebraic properties on random multivectors ------------------------------------

def test_scalar_one_is_identity(rng):
    A = rng.normal(size=DIM)
    one = mv(s=1)
    assert np.allclose(_gp_loop(one, A), A)
    assert np.allclose(_gp_loop(A, one), A)


def test_associative(rng):
    A, B, C = rng.normal(size=(3, DIM))
    assert np.allclose(_gp_loop(_gp_loop(A, B), C), _gp_loop(A, _gp_loop(B, C)))


def test_distributive(rng):
    A, B, C = rng.normal(size=(3, DIM))
    assert np.allclose(_gp_loop(A, B + C), _gp_loop(A, B) + _gp_loop(A, C))
    assert np.allclose(_gp_loop(A + B, C), _gp_loop(A, C) + _gp_loop(B, C))


def test_linear_in_scalars(rng):
    A, B = rng.normal(size=(2, DIM))
    assert np.allclose(_gp_loop(2.5 * A, B), 2.5 * _gp_loop(A, B))
    assert np.allclose(_gp_loop(A, -3 * B), -3 * _gp_loop(A, B))


def test_not_commutative(rng):
    A, B = rng.normal(size=(2, DIM))
    assert not np.allclose(_gp_loop(A, B), _gp_loop(B, A))


# --- 2. vectors -----------------------------------------------------------------------

def test_euclidean_vector_squares_to_length():
    u = mv(e1=1, e2=2, e3=3)
    assert np.allclose(_gp_loop(u, u), mv(s=14))


def test_degenerate_direction_drops_out():
    u = mv(e0=1, e1=1)
    assert np.allclose(_gp_loop(u, u), mv(s=1))


def test_orthogonal_vectors():
    a = mv(e1=1, e2=1)
    b = mv(e1=1, e2=-1)
    assert np.allclose(_gp_loop(a, b), mv(e12=-2))
    # symmetric part = 2 * (a . b) = 0 for orthogonal vectors
    assert np.allclose(_gp_loop(a, b) + _gp_loop(b, a), 0)


def test_symmetric_part_is_scalar(rng):
    a = mv(e1=rng.normal(), e2=rng.normal(), e3=rng.normal())
    b = mv(e1=rng.normal(), e2=rng.normal(), e3=rng.normal())
    sym = _gp_loop(a, b) + _gp_loop(b, a)
    assert np.allclose(sym[1:], 0)                  # nothing but scalar
    assert np.isclose(sym[0], 2 * np.dot(a, b))     # and it equals 2 a.b


# --- 3. bivectors (quaternion structure) ---------------------------------------------

def test_bivectors_anticommute():
    assert np.allclose(_gp_loop(mv(e12=1), mv(e23=1)), mv(e13=1))
    assert np.allclose(_gp_loop(mv(e23=1), mv(e12=1)), mv(e13=-1))


def test_bivector_triple_product():
    assert np.allclose(_gp_loop(mv(e23=1), mv(e13=1)), mv(e12=1))
    triple = _gp_loop(_gp_loop(mv(e23=1), mv(e13=1)), mv(e12=1))
    assert np.allclose(triple, mv(s=-1))


# --- 4. rotors -------------------------------------------------------------------------

def rotor(theta):
    return mv(s=np.cos(theta / 2), e12=np.sin(theta / 2))


@pytest.mark.parametrize("theta", [0.0, 0.3, np.pi / 2, np.pi, 2.7])
def test_rotor_times_reverse_is_one(theta):
    R = rotor(theta)
    R_rev = mv(s=np.cos(theta / 2), e12=-np.sin(theta / 2))
    assert np.allclose(_gp_loop(R, R_rev), mv(s=1))


def test_rotors_compose():
    alpha, beta = 0.4, 1.1
    assert np.allclose(_gp_loop(rotor(alpha), rotor(beta)), rotor(alpha + beta))


# --- 5. pseudoscalar and e0 ------------------------------------------------------------

def test_pseudoscalar_products():
    assert np.allclose(_gp_loop(mv(e0123=1), mv(e1=1)), mv(e023=1))
    assert np.allclose(_gp_loop(mv(e0123=1), mv(e0=1)), 0)
    assert np.allclose(_gp_loop(mv(e0123=1), mv(e0123=1)), 0)


def test_e123_anticommutes_with_e0():
    assert np.allclose(_gp_loop(mv(e123=1), mv(e0=1)), mv(e0123=-1))
    assert np.allclose(_gp_loop(mv(e0=1), mv(e123=1)), mv(e0123=1))


@pytest.mark.parametrize("e", ["e1", "e2", "e3"])
def test_e123_commutes_with_euclidean_vectors(e):
    I3, v = mv(e123=1), mv(**{e: 1})
    assert np.allclose(_gp_loop(I3, v), _gp_loop(v, I3))
