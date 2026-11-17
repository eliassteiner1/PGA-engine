# PGA-engine

A 3D Projective Geometric Algebra engine (`R(3,0,1)`, with e0² = 0 and e1² = e2² = e3² = +1), built from scratch to understand how it works.

> **Status: work in progress.** This README is a temporary progress log, not documentation yet.

## Current state

### The algebra core ([`src/pga_engine/algebra.py`](src/pga_engine/algebra.py))

- **Multivectors** are numpy arrays of coefficients, ordered by grade. For 3D PGA that is 16 entries:

  | Index | 0 | 1–4 | 5–10 | 11–14 | 15 |
  |---|---|---|---|---|---|
  | Grade | scalar | vectors (planes) | bivectors (lines) | trivectors (points) | pseudoscalar |
  | Blades | 1 | e0 e1 e2 e3 | e01 e02 e03 e12 e13 e23 | e012 e013 e023 e123 | e0123 |

  Every blade's indices are sorted ascending (e13, not e31). Many PGA references use e31, e021 and e032, so expect sign differences when comparing results.
- **Basis blades as bitmasks:** bit k stands for e_k, so e13 = `0b1010`. The order above isn't hardcoded: all bitmasks `0 … 2ⁿ−1` are sorted by grade, then by their index tuple.
- **Blade multiplication** (`_resolve_bb_mult_bitmask(a, b, metric)`):
  - the result blade is `a ^ b`
  - the sign is (−1)^swaps, with the swaps counted by shift + popcount
  - every unit vector shared by both blades contributes its square `metric[k]`. For PGA, a shared e0 makes the product 0.
- **`Algebra`** is a frozen dataclass holding everything for one metric. `Algebra.from_metric(metric)` computes:
  - `blades` (array index → bitmask) and `blade_index` (bitmask → array index)
  - `grade`: the grade of each array index
  - `gp_table`: the geometric product as a `(dim, dim, dim)` tensor, `blade_i blade_j = gp_table[i, j, k] blade_k`

  The arrays are read-only, so one instance can be shared everywhere. `PGA3D = Algebra.from_metric((0, 1, 1, 1))` is the default; other metrics work too, e.g. `(1, 1, 1)` for plain 3D Euclidean GA.

### Products ([`src/pga_engine/products.py`](src/pga_engine/products.py))

- `gp(a, b)`: the geometric product as one `einsum` over `gp_table`. Works on single multivectors and on batches of shape `(N, dim)`.
- `gp_loop(a, b)`: a readable double-loop version, used as the reference in the tests.
- Both take an optional `alg=` argument (default `PGA3D`).

### Utilities ([`src/pga_engine/utils.py`](src/pga_engine/utils.py))

- `blade(0, 2)` returns the bitmask `0b0101`.
- `name_to_enum("e12")` returns the array index `8`.
- `mv(s=1, e12=0.5)` returns a multivector.
- All take an optional `alg=` argument (default `PGA3D`).
- Inputs are validated: unsorted, duplicate or out-of-range indices raise `ValueError`.

### Tests ([`tests/`](tests/))

| File | Covers |
|---|---|
| `test_blade_mult.py` | hand-checked blade products, the e0 → 0 rule, anticommutation, other metrics |
| `test_algebra.py` | blade order, immutability, and for several metrics: sizes, scalar identity, vector squares, associativity of the table |
| `test_gp.py` | `gp` vs `gp_loop`, batching, vectors, quaternion structure of the bivectors, rotors, pseudoscalar |
| `test_utils.py` | blade and name construction, invalid input |

Run them with:

```powershell
uv run pytest            # everything
uv run pytest -v -k gp   # verbose, only tests matching "gp"
```

## Next steps

1. **Outer and inner products as masked tables:** `gp_table` with a grade mask applied, stored as extra fields on `Algebra`:
   - wedge `∧` (meet): keep the entries where `grade[k] == grade[i] + grade[j]`
   - inner `·` (symmetric "fat dot"): keep the entries where `grade[k] == |grade[i] − grade[j]|`

   Tests: cross-check the wedge table against a resolver where any shared vector gives 0. For vectors, `a·b + a∧b == ab`. For bivectors, `e12·e23 = e12∧e23 = 0` while `e12 e23 = e13`.
2. **Unary operations:** grade projection `⟨A⟩ₖ`, reverse, grade involution and Clifford conjugate. Each is a fixed ±1 array per grade.
3. **Dual and regressive product `∨` (join).** The pseudoscalar can't be inverted (I² = 0), so PGA needs a Poincaré/Hodge dual defined by index permutation.
4. **Geometry layer:**
   - constructors for planes, lines and points
   - norms (Euclidean and ideal)
   - normalization
   - the sandwich product `R X R̃`
   - `exp`/`log` for building rotors and motors
