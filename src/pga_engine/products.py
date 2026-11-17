""" products of full multivectors. all of them read the lookup tables of an Algebra (PGA3D by default). """

import numpy as np

from pga_engine.algebra import PGA3D, Algebra


def gp_loop(v1: np.ndarray, v2: np.ndarray, alg: Algebra = PGA3D) -> np.ndarray:
    """ computes the geometric product for arbitrary multivectors. this is the "more understandable" version that does the calculation through a loop. only works for single multivectors, no batches. used as reference for gp in the tests. """

    v1v2 = np.zeros(alg.dim)

    for i in range(alg.dim):
        for j in range(alg.dim):
            # iterating through each possible multiplication of basis elements. gp_table[i, j] is a whole multivector: zero everywhere except at the resulting blade, where it holds the sign (or all zero if the product vanishes). so it just has to be scaled by both coefficients and added up
            v1v2 += v1[i] * v2[j] * alg.gp_table[i, j]

    return v1v2

def gp(v1: np.ndarray, v2: np.ndarray, alg: Algebra = PGA3D) -> np.ndarray:
    """ computes the geometric product v1 v2 as a single tensor contraction: result_k = sum over i, j of v1_i v2_j gp_table[i, j, k].
    the ... in the einsum string stands for any leading batch dimensions, so v1 and v2 can be single multivectors of shape (dim,) or batches of shape (N, dim), which also broadcast against each other. """
    return np.einsum("...i, ...j, ijk -> ...k", v1, v2, alg.gp_table)
