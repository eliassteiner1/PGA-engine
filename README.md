# PGA-engine

A 3D Projective Geometric Algebra engine (`R(3,0,1)`, with e0² = 0 and e1² = e2² = e3² = +1), built from scratch to understand how it works.

> **Status: work in progress.** This README is a temporary progress log, not documentation yet.

## Current state

### Layer 1: the algebra core ([`src/pga_engine/layer1.py`](src/pga_engine/layer1.py))

- **Multivectors** are 16-element numpy arrays of coefficients, ordered by grade:

  | Index | 0 | 1–4 | 5–10 | 11–14 | 15 |
  |---|---|---|---|---|---|
  | Grade | scalar | vectors (planes) | bivectors (lines) | trivectors (points) | pseudoscalar |
  | Blades | 1 | e0 e1 e2 e3 | e01 e02 e03 e12 e13 e23 | e012 e013 e023 e123 | e0123 |

  Every blade's indices are sorted ascending (e13, not e31). Many PGA references use e31, e021 and e032, so expect sign differences when comparing results.
- **Basis blades as bitmasks:** bit k stands for e_k, so e13 = `0b1010`. `ENUM_2_BLADE` and `BLADE_2_ENUM` map between bitmask and array index.
- **Blade multiplication** (`resolve_bb_mult_bitmask`):
  - the result blade is `a ^ b`
  - the product is 0 if both blades contain e0
  - the sign is (−1)^swaps, with the swaps counted by shift + popcount
- **Cayley table:** `CAYLEY_TABLE` (result index) and `CAYLEY_SIGNS` (−1, 0, +1) are precomputed once and read-only.
- **Geometric product:** `_gp_loop` is a readable double-loop reference. A vectorized `gp` (using `np.add.at`) is in progress.
- **Index-list version** (`basis_blades_idx`, `resolve_bb_mult_index`): kept for documentation only.

### Utilities ([`src/pga_engine/utils.py`](src/pga_engine/utils.py))

- `blade(0, 2)` returns the bitmask `0b0101`.
- `name_to_enum("e12")` returns the array index `8`.
- `mv(s=1, e12=0.5)` returns a 16-element multivector.
- Inputs are validated: unsorted, duplicate or out-of-range indices raise `ValueError`.

### Tests ([`tests/`](tests/))

| File | Covers |
|---|---|
| `test_blade_mult.py` | hand-checked blade products, the e0 → 0 rule, anticommutation |
| `test_cayley.py` | dtype and ranges, scalar identity, squares of the basis blades, associativity over all 16³ triples |
| `test_gp.py` | identity, associativity, distributivity, linearity, non-commutativity, vectors, bivectors (quaternion structure), rotors, pseudoscalar / e0 |
| `test_utils.py` | blade and name construction, invalid input |

Run them with:

```powershell
uv run pytest            # everything
uv run pytest -v -k gp   # verbose, only tests matching "gp"
```

## Next steps

1. **Finish the vectorized `gp`.** Build it with `np.add.at`, add a shape check that raises `ValueError`, and add a test comparing it against `_gp_loop` on random inputs.
2. **Add a grade map and a generic product.** Add a `GRADE` array (one grade per index) and a `_product(a, b, signs)` function that all products share.
3. **Outer and inner products as masked sign tables** sharing `CAYLEY_TABLE`:
   - wedge `∧` (meet): keep the pairs where `grade == r + s`
   - inner `·` (symmetric "fat dot"): keep the pairs where `grade == |r − s|`

   Tests: cross-check the wedge table against a resolver where any shared vector gives 0. For vectors, `a·b + a∧b == ab`. For bivectors, `e12·e23 = e12∧e23 = 0` while `e12 e23 = e13`.
4. **Unary operations:** grade projection `⟨A⟩ₖ`, reverse, grade involution and Clifford conjugate. Each is a fixed ±1 array per grade.
5. **Dual and regressive product `∨` (join).** The pseudoscalar can't be inverted (I² = 0), so PGA needs a Poincaré/Hodge dual defined by index permutation.
6. **Geometry layer:**
   - constructors for planes, lines and points
   - norms (Euclidean and ideal)
   - normalization
   - the sandwich product `R X R̃`
   - `exp`/`log` for building rotors and motors
7. **Performance:** batched multivectors with shape `(N, 16)`, and later sparse or generated product code.
