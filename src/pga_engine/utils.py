import numpy as np

from pga_engine.algebra import PGA3D, Algebra


def blade(*indices: int, alg: Algebra = PGA3D) -> int:
    """ builds the bitmask of a basis blade from its unit vector indices: blade(0, 2) -> 0b_0101 (e02). blade() is the scalar. """

    # indices must be strictly increasing: e21 is a valid product but NOT one of our basis blades (it's -e12),
    # silently mapping it to e12 would lose that sign. duplicates (e11) would also give a wrong bitmask.
    if any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError(f"indices must be strictly increasing, got {indices}")
    if any(not 0 <= k < alg.nr_unitvecs for k in indices):
        raise ValueError(f"indices must be in 0..{alg.nr_unitvecs - 1}, got {indices}")

    return sum(1 << k for k in indices)


def name_to_enum(name: str, alg: Algebra = PGA3D) -> int:
    """ converts a blade name to its position in the multivector array: 'e12' -> 8. 's' is the scalar. """

    if name == "s":
        return 0
    if not (name.startswith("e") and name[1:].isdigit()):
        raise ValueError(f"blade name must be 's' or 'e' followed by indices (e.g. 'e12'), got {name!r}")

    return alg.blade_index[blade(*(int(c) for c in name[1:]), alg=alg)]


def mv(alg: Algebra = PGA3D, **coeffs: float) -> np.ndarray:
    """ builds a multivector from named coefficients: mv(s=1, e12=0.5) -> coefficient array of length alg.dim. """

    out = np.zeros(alg.dim)
    for name, value in coeffs.items():
        out[name_to_enum(name, alg)] = value
    return out
