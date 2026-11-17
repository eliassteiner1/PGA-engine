import numpy as np


# index notation (just experiment) =====================================================================================
basis_blades_idx = [
    # this is an example definition of the basis blades with index notation. each list represents the combinations of e_i that form the basis blades
    
    # grade 0 (scalar)
    [],
    # grad 1 (vector)
    [0],
    [1],
    [2],
    [3],
    # grade 2 (bivector)
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 3],
    # grad 3 (trivector)
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3],
    # grade 4 (pseudoscalar)    
    [0, 1, 2, 3],
]

def resolve_bb_mult_index(bb1: list[int], bb2: list[int]) -> tuple[int, list[int]]:
    """ an example function for how to resolve multiplication of two basis blades in the index notation. concatenatin the two index lists is the multiplication. then the indices are sorted and doubles are eliminated, while keeping track of number of swaps for sign changes. """
    
    cat = bb1 + bb2 # these are the indices that would result from multiplying two basis blades
    sgn = 0 # this is basically the (-1)**sng exponent to keep track of sign changes
    
    while True:
        
        chg = False # keeping track if the list has changed this iteration
        rmv = [] # elements to remove from the list at the end of one iteration
        
        i = 0
        while i < len(cat)-1:
            a = cat[i]
            b = cat[i+1]
            
            # if a == b they reduce
            if a == b:
                if a == 0:
                    # in this case the entire multiplication collapses to 0 and we're finished
                    return 0, []
                else:
                    # if a = b > 0, their multiplication reduces to 1
                    # skip ahead as this pair is treated at once here. will be removed at the end of iteration
                    rmv += [i, i+1]
                    chg = True
                    i += 2
                    continue

            elif a > b:
                # as we defined the basis blades with only increasing indices, we need to swap the two!
                cat[i], cat[i+1] = cat[i+1], cat[i]
                # and add a (-1) as all swaps yield a sign change
                chg = True
                sgn += 1
                i += 1
                continue

            # if a<b all good! 
            i += 1
        
        if not chg:
            # if nothing was changed anymore we're done
            return (-1)**sgn, cat
    
        # remove all rmv elements (basically replace cat)
        cat = [cat[i] for i in range(len(cat)) if i not in rmv]
 

# Bitmask Notation =====================================================================================================
NR_UNITVECS = 4
DIM = 16
E0 = 0b_0001

ENUM_2_BLADE = [
    # bitmask is : [e3|e2|e1|e0]
    # this maps each basis blade "number/index" to it's corresponding bitmask representation. sorted by grade for convenience
    
    # grade 0 (scalar)
    0b_0000, # nr0
    # grad 1 (vector)
    0b_0001, # nr1
    0b_0010, # nr2
    0b_0100, # nr3
    0b_1000, # nr4
    # grade 2 (bivector)
    0b_0011, # nr5
    0b_0101, # nr6
    0b_1001, # nr7
    0b_0110, # nr8
    0b_1010, # nr9
    0b_1100, # nr10
    # grad 3 (trivector)
    0b_0111, # nr11
    0b_1011, # nr12
    0b_1101, # nr13
    0b_1110, # nr14
    # grade 4 (pseudoscalar)    
    0b_1111, # nr15
]

BLADE_2_ENUM = {
    # this is the reverse mapping to go from a BB in bitmask representation (which is also just a numer! but not the index) -> it's index (sorted by grade)
    blade: i for i, blade in enumerate(ENUM_2_BLADE)
}

def resolve_bb_mult_bitmask(bb1: int, bb2: int) -> tuple[int, int]:
    """ a more efficient way to resolve the multiplication of two elementary basis blades. XOR on their bitmasks directly yields the resulting bb. if a double e0 is present, the result is instantly 0. tracking the sing change needs a bit more attention... """
    
    # in case that both inputs carry the unitvector e0, the whole multiplication reduces to scalar 0
    if bb1 & bb2 & E0:
        return 0, 0
    
    # use bitwise XOR for the sign free result of the multiplication: if both inputs contain a unitvector it reduces, in neither of them contains it, the result won't contain it either. 
    res = bb1 ^ bb2
    
    # determine the sign of the multiplication: each swap of two unitvecs yields a (-1) factor. to determine directly, how many swaps are necessary to sort the new product, counting the number of pairs (bb1_i, bb2_j) where bb1_i is LARGER than bb2_j is enough. the parity of that number determines the sign.
    swaps = 0
    for i in range(1, NR_UNITVECS):
        temp = (bb1 >> i) & bb2
        swaps += temp.bit_count()
    
    return (-1)**swaps, res

def _build_cayley() -> tuple[np.ndarray, np.ndarray]:
    """ builds the lookup table for arbitrary multivector multiplication """
    
    # datatype int, because the tables entries are used as indexes again
    table = np.zeros(shape=(DIM, DIM), dtype=int)
    signs = np.zeros(shape=(DIM, DIM), dtype=int)

    for i in range(DIM):
        for j in range(DIM):
            
            # iterate through each basis blade combination and evaluate their multiplication
            a = ENUM_2_BLADE[i]
            b = ENUM_2_BLADE[j]
            sgn, res = resolve_bb_mult_bitmask(a, b)
            
            # result is saved as the cayley lookup table + resulting sign
            table[i, j] = BLADE_2_ENUM[res]
            signs[i, j] = sgn
            
    return signs, table

CAYLEY_SIGNS, CAYLEY_TABLE = _build_cayley()

def gp(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """ computes the geometric product ab = a·b + a^b """
    
    v1v2 = np.zeros(DIM)

    for i in range(DIM):
        for j in range(DIM):
            
            # iterating through each possible multiplication of basis elements. coefficients are multiplied and the basis element is converted to the result according to the cayley table. the whole result is multiplied by the sign. 
            coeff1 = v1[i]
            coeff2 = v2[j]
            
            resulting_blade_idx = CAYLEY_TABLE[i, j]
            res_sign_change = CAYLEY_SIGNS[i, j]
            
            v1v2[resulting_blade_idx] += res_sign_change * coeff1 * coeff2
    
    return v1v2