""" products of full multivectors. all of them read the lookup tables of an Algebra (PGA3D by default). """

import numpy as np

from pga_engine.algebra import PGA3D, Algebra


def gp(v1: np.ndarray, v2: np.ndarray, alg: Algebra = PGA3D) -> np.ndarray:
    """ computes the geometric product v1 v2 as a single tensor contraction: result_k = sum over i, j of v1_i v2_j gp_table[i, j, k]. the ... in the einsum string stands for any leading batch dimensions, so v1 and v2 can be single multivectors of shape (dim,) or batches of shape (N, dim), which also broadcast against each other. """
    
    # TODO: this is the first function that's actually used. add some input sanitization, at least squeeze and assert dim on vecs...
    
    return np.einsum("...i, ...j, ijk -> ...k", v1, v2, alg.gp_table)


