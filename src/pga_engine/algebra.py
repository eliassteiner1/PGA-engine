""" the algebra core: basis blades as bitmasks, their multiplication, and the Algebra dataclass that holds all precomputed lookup tables for a given metric. """

import numpy as np
from dataclasses import dataclass
from typing import Self


def _bb_bitmask_2_index(bitmask: int, n_unitvecs: int) -> tuple[int, ...]:
    """ returns the indices of the unit vectors contained in a blade bitmask: 0b_1010 -> (1, 3) (e13). nvecs is the number of unit vectors (bits) to check. """
    return tuple(i for i in range(n_unitvecs) if (bitmask >> i) & 1)

def _bb_bitmask_resolve_multiplication(bb1: int, bb2: int, metric: tuple[int, ...]) -> tuple[int, int]:
    """ resolves the multiplication of two basis blades given as bitmasks. returns (sign, resulting blade bitmask).
    XOR on the bitmasks directly yields the resulting blade. the sign comes from two parts: the swaps needed to sort the product, and the squares of all unit vectors shared by both blades, given by the metric (e_k e_k = metric[k]).
    for PGA this means any shared e0 makes the whole product 0, as metric[0] = 0. in that case the sign is 0 and the returned blade is meaningless. """

    # use bitwise XOR for the sign free result of the multiplication: if both inputs contain a unitvector it reduces, if neither of them contains it, the result won't contain it either.
    resulting_bb_bitmask = bb1 ^ bb2   
    
    # determine the sign of the multiplication: each swap of two unitvecs yields a (-1) factor. to determine directly, how many swaps are necessary to sort the new product, counting the number of pairs (bb1_i, bb2_j) where bb1_i is LARGER than bb2_j is enough. the parity of that number determines the sign.
    swaps_counter = 0
    for i in range(1, len(metric)):
        pairs_bb1_greater_bb2 = (bb1 >> i) & bb2
        swaps_counter += pairs_bb1_greater_bb2.bit_count()
    resulting_sign = (-1)**swaps_counter
        
    # find all shared unitvecs: after sorting, each of them meets its twin and the pair collapses to its square metric[k]. a single 0 in there (e.g. e0 in PGA) kills the whole product
    shared_unitvecs_bitmask = bb1 & bb2
    for i in _bb_bitmask_2_index(shared_unitvecs_bitmask, len(metric)):
        resulting_sign *= metric[i]

    return resulting_sign, resulting_bb_bitmask

@dataclass(frozen=True)
class Algebra:
    """ holds everything needed to compute in the geometric algebra with the given metric. don't construct directly, use Algebra.from_metric(metric), which computes all the lookup tables.
    frozen and the arrays are read-only, so one instance can safely be shared everywhere (e.g. as default argument). """

    metric: tuple[int, ...]         # square of each unit vector: (0, 1, 1, 1) means e0² = 0, e1² = e2² = e3² = +1
    blades: tuple[int, ...]         # array index -> blade bitmask, sorted by grade, then by indices
    blade_index: dict[int, int]     # blade bitmask -> array index, the reverse of blades
    blades_grades: np.ndarray       # (dim,) grade of the blade at each array index
    gp_table: np.ndarray            # (dim, dim, dim) geometric product: blade_i * blade_j = gp_table[i, j, k] * blade_k

    @property
    def nr_unitvecs(self) -> int:
        """ number of unit vectors e_k, derived from the metric """
        return len(self.metric)

    @property
    def dim(self) -> int:
        """ number of basis blades, i.e. the length of a multivector array: 2 ** nr_unitvecs """
        return len(self.blades)

    @classmethod
    def from_metric(cls, metric) -> Self:
        """ builds an algebra and all its lookup tables from a metric (one entry per unit vector, any iterable). e.g. (0, 1, 1, 1) for 3D PGA or (1, 1, 1) for plain 3D euclidean GA. """

        #TODO: add some input sanitization, just assert on metric?

        # normalize input and setup params
        metric = tuple(metric)
        n_unitvecs = len(metric)
        dim = 2 ** n_unitvecs

        # bit k of a bitmask stands for e_k, e.g. for 4 unit vectors: [e3|e2|e1|e0]. this maps each basis blade "number/index" to its corresponding bitmask representation. sorted by grade first for convenience, then lexicographically (ascending indices)
        blades = tuple(sorted(range(dim), key=lambda b: (b.bit_count(), _bb_bitmask_2_index(b, n_unitvecs))))

        # this is the reverse mapping to go from a BB in bitmask representation (which is also just a number! but not the index) -> its index (sorted by grade)
        blade_index = {blade: i for i, blade in enumerate(blades)}
        
        # grade of each basis blade
        blades_grades = np.array([b.bit_count() for b in blades])

        # builds the lookup table for arbitrary multivector multiplication, the geometric product. this is a scattered version of the typical 2D cayley table, as it lends itself better to einsum tensor contraction.
        # gp_table[i, j, k] = sign means blade_i * blade_j = sign * blade_k. for each (i, j) at most one k is nonzero (none if the product is 0)
        gp_table = np.zeros((dim, dim, dim))
        for i in range(dim):
            for j in range(dim):
                bb_sign, bb_bitmask = _bb_bitmask_resolve_multiplication(blades[i], blades[j], metric)
                gp_table[i, j, blade_index[bb_bitmask]] = bb_sign

        # lock the lookup tensors
        blades_grades.flags.writeable = False
        gp_table.flags.writeable = False

        # construct the class with all the necessary lookup tables
        return cls(
            metric=metric,
            blades=blades,
            blade_index=blade_index,
            blades_grades=blades_grades,
            gp_table=gp_table
        )

# the default algebra used throughout the library: 3D PGA R(3,0,1), with e0² = 0 and e1² = e2² = e3² = +1
PGA3D = Algebra.from_metric(metric=(0, 1, 1, 1)) # e0, e1, e2, e3



