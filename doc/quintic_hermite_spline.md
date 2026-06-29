# Quintic Hermite radial splines — implementation report

**Code:** `pyscf/OpenCL/hermite_spline.py`  
**Study CLI:** `expamples_prokop/hermite_radial_study.py` (thin wrapper: `plot_hermite_cubic_quintic.py`)  
**f32 GPU kernel:** `pyscf/OpenCL/hermite_spline_f32.cl` + `pyscf/OpenCL/hermite_spline_cl.py`  
**Plots / tables:** `debug/plot_hermite_cubic_quintic/`

Fitted target: **contracted** Cartesian radial  
`R(r) = Σ_k c_k exp(-α_k r²)` from libcint coefficients (not per-primitive).

---

## 1. What we implemented

### 1.1 Core spline module (`hermite_spline.py`)

| Feature | Description |
|---------|-------------|
| **Orders** | Cubic and quintic Hermite on non-uniform knots |
| **interp_space** | `u`: Hermite in mapped `u = β·log1p(r/r₀)` (production-style); `r`: same physical nodes, Hermite in `r` |
| **tangents** | `analytic`: knot `R`, `R′`, `R″` from exact GTO; `quadrature`: global LSQ on interior collocation (values at nodes fixed) |
| **Grids** | `power` (fixed N, β clusters toward origin), `uniform` (equal Δr), `log` (uniform Δu, N grows with β) |
| **origin_knot** | Prepend analytic knot at `r=0` so segment `[0, r₀]` has a left neighbour (half-line boundary fix) |
| **Eval** | `eval_radial_spline`, `eval_radial_spline_dr` for both orders and both interp spaces |

Cubic formulas match production `kernels.cl` (`hermite_eval_node`). Quintic uses standard C² Hermite basis (`H00…H21`).

### 1.2 Study tooling (`hermite_radial_study.py`)

Unified CLI subcommands:

| Command | Purpose |
|---------|---------|
| `report` | Grid plot + β-sweep table + per-shell u/r curves |
| `carbon` | Per-shell error vs β for one interp mode |
| `grid` | `r_i` vs node index, local Δr |
| `compare` | Separate u-mode vs r-mode shell plots |
| `matrix` | **4 combos per order** (u/r × analytic/quadrature); separate cubic/quintic PNG per shell |
| `f32` | f64 vs OpenCL float32 error; same combos, solid=f64 / dashed=f32 |

Plot conventions (matrix / f32):

- **Colors:** ana-u green, ana-r blue, quad-u orange, quad-r purple (`lw=0.5`)
- **Reference:** exact `|R|`, `|dR/dr|` black `lw=1.5`
- **Smoothing:** sliding maximum on error curves (`--error-smooth N`, default 5; `0` = off). Tables always use raw max errors.

Eval grid starts at **first physical knot** `r₀` (default 0.002 Å) to avoid extrapolation artifacts below the grid.

### 1.3 f32 OpenCL study kernel

Standalone kernel (not yet wired into production AO path in `kernels.cl`):

- `hermite_spline_f32.cl`: `eval_radial_spline_f32` — cubic (`float2` knots) or quintic (`float4`), u/r-mode, variable knot spacing
- Tables built in f64, **cast to f32** before GPU eval (mirrors production storage)
- `--backend cl` (GPU) or `cpu` (numpy f32 replay of same formulas)

---

## 2. Grid modes

### Power grid (default study: β=1, N=210)

```
r_i = r₀ · (r_max/r₀)^((i/(N-1))^(1/β))     β>1 packs nodes toward origin
u_i = β · log1p(r_i / r₀)
```

β controls origin clustering at **fixed N** (not `du ∝ β`).

### Log grid (production-style)

```
u_i = i · du,   r_i = r₀ · expm1(u_i/β)
```

N grows when β increases if `du` is fixed.

### Uniform grid

Equal Δr in physical space; same reference N as power grid for fair comparison.

---

## 3. Caveats and bugs found

### 3.1 Wrong `R″(r)` formula (critical — fixed)

Quintic **analytic** mode uses exact second derivative at knots. The initial implementation had the wrong sign on the `4α²r²` term:

```text
d²/dr² exp(-αr²) = (-2α + 4α²r²) exp(-αr²)     ✓
                  ≠ (-2α - 4α²r²) …             ✗ (old code)
```

**Symptom:** quintic+analytic looked catastrophically worse than cubic on 1s (`|ΔR′| ~ 0.4` vs `~10⁻⁴`).  
**After fix:** quintic+analytic is best mode for derivatives (see §5).

### 3.2 Half-line boundary at grid start

Max `|ΔR′|` often peaks at **r = r₀** (first physical knot), not because Gaussians are hard to fit globally, but because the first interval `[0, r₀]` lacked a left neighbour.

**Fix:** `origin_knot=True` (default) prepends `r=0` with analytic `R, R′, R″`.

### 3.3 Evaluating below r₀

Sampling `r < r₀` inflates errors (extrapolation). Study eval grid: `r_dense ∈ [r₀, r_max]`.

### 3.4 Quadrature fits values, not derivatives

`tangents=quadrature` LSQ matches **R** at interior Gauss points; knot `R′`, `R″` are free. Values fit to ~10⁻¹²; derivatives can still be bad at **r = r₀** (first segment), especially steep 1s:

| 1s quintic u quad | f64 `|ΔR′|` |
|-------------------|-------------|
| interior (r > 0.01 Å) | ~1.6×10⁻⁸ |
| global max at r₀ | **7.7×10⁻³** |

Quadrature is useful for value accuracy / node compression, **not** as the derivative-accuracy mode when analytic derivatives are available.

### 3.5 Quintic + analytic on valence (before R″ fix)

Was mis-attributed to “quintic can’t beat cubic” — root cause was wrong `R″`, not the quintic basis.

### 3.6 u-mode vs r-mode

- **u-mode** is production path (`kernels.cl`); chain rule `dR/dr = (dR/du)·β/(r+r₀)`.
- **r-mode** often similar or slightly better for derivatives on tight shells when using the same nodes.
- Fair cubic vs quintic comparison should use **same tangents mode** and preferably r-mode for derivative-focused tests.

### 3.7 f32 table quantization

f64 analytic quintic gives `|ΔR′| ~ 10⁻⁸` on 1s; same tables cast to **float32** and evaluated on GPU → `~10⁻³` on 1s.

Cause: steep 1s knot derivatives are O(10¹)–O(10²); `float32` storage of `dy/du` and `d²y/du²` loses precision **before** interpolation. Quadrature modes (already ~10⁻²–10⁻¹ error) are barely affected by f32.

**Open issue:** compensated packing (e.g. store `d·h`, `c·h²`) or per-shell scaling for GPU quintic.

### 3.8 Relative vs absolute error

Contracted radial sums can **cancel** at some r; `|err|/|R|` blows up while `|err|` stays ~10⁻⁷. For XC / ρ, report **absolute** `|ΔR|`, `|ΔR′|`; use `error_metrics()` rel floor for AO-level studies.

---

## 4. Knot storage (GPU target)

| Spline | GPU type | Per-node data | Bytes/node |
|--------|----------|---------------|------------|
| Cubic | `float2` | y, dy/du | 8 |
| Quintic | `float4` | y, dy/du, d²y/du², pad | 16 |

Memory-equivalent step: `du_quintic ≈ 2 · du_cubic` (half the nodes at twice the width).

Production `kernels.cl` still has **cubic only** (`hermite_eval_node`); quintic production path is not merged yet.

---

## 5. Results (benzene cc-pVDZ, carbon shells, β=1, power N=210, origin_knot)

Reference configuration for final comparison. Primary metric: **max |Δ(dR/dr)|** (GGA XC needs ∇ρ).

### 5.1 f64 matrix (`matrix_report_power_b1.txt`)

| shell | best mode | |ΔR′| | notes |
|-------|-----------|--------|-------|
| **1s** | quintic u ana | **5.8×10⁻⁸** | cubic u ana 1.0×10⁻⁴; quad modes 10⁻²–10⁻¹ |
| **1s′** | quintic u ana | **2.5×10⁻⁸** | |
| **2s** | quintic r quad | **7.9×10⁻¹¹** | ana modes ~10⁻¹⁰ |
| **2p** | quintic r quad | **4.5×10⁻⁹** | ana u 1.1×10⁻⁸; cubic u ana 1.7×10⁻⁵ |
| **3p** | quintic r quad | **5.8×10⁻¹¹** | |
| **3d** | quintic r quad | **7.4×10⁻¹⁰** | ana u 1.8×10⁻⁹ |

**Patterns:**

1. **Quintic + analytic** — best on steep core (1s): 3–4 orders better than cubic.
2. **Quintic + quadrature** — excellent **values** (`|ΔR| ~ 10⁻¹²`); derivative max still at r₀ (~10⁻³ on 1s, ~10⁻⁵ on 2p).
3. **Cubic + quadrature** — poor on 1s derivatives (u quad 0.36, r quad 0.08).
4. **Cubic + analytic** — usable on valence (2p ~10⁻⁵) but loses to quintic analytic by 3+ orders.
5. Max `|ΔR′|` for quad modes: **r′(Å) = 0.002** (= r₀). For ana modes: often interior.

Full table (all 48 rows) in `debug/plot_hermite_cubic_quintic/matrix_report_power_b1.txt`.

### 5.2 f32 OpenCL (`f32_report_power_b1.txt`, RTX 3090)

| shell | quintic ana u (f64 → f32) | cubic ana u (f64 → f32) |
|-------|---------------------------|-------------------------|
| 1s | 5.8×10⁻⁸ → **1.1×10⁻³** | 1.0×10⁻⁴ → 6.4×10⁻⁴ |
| 2p | 1.1×10⁻⁸ → **7.4×10⁻⁴** | 1.7×10⁻⁵ → 1.8×10⁻⁴ |
| 2s | 2.0×10⁻¹⁰ → 4.0×10⁻⁵ | 4.1×10⁻⁷ → 4.7×10⁻⁵ |

Quadrature rows: f32 ≈ f64 (fit error dominates).

### 5.3 Recommended modes

| Goal | Recommendation |
|------|----------------|
| **Derivative accuracy (study / future GGA)** | quintic + **analytic** tangents |
| **Fewer nodes / value fit** | quintic + quadrature (accept r₀ derivative spike or fix with derivative-aware LSQ) |
| **Production today** | cubic + midpoint in `radial_hermite.py` / `kernels.cl` |
| **GPU quintic** | fix f32 knot packing before merging to `kernels.cl` |

---

## 6. How to run

```bash
# Full matrix: cubic + quintic plots per shell (β=1, power grid)
PYTHONPATH=/home/prokop/git/pyscf python3 -u expamples_prokop/hermite_radial_study.py matrix --grid power

# f64 vs f32 (OpenCL)
PYTHONPATH=/home/prokop/git/pyscf python3 -u expamples_prokop/hermite_radial_study.py f32 --grid power

# f32 numpy replay (no GPU)
PYTHONPATH=/home/prokop/git/pyscf python3 -u expamples_prokop/hermite_radial_study.py f32 --backend cpu

# β sweep + per-shell curves
PYTHONPATH=/home/prokop/git/pyscf python3 -u expamples_prokop/hermite_radial_study.py report --grid power --order quintic --beta-list 0.5 1 2 4

# Plot smoothing off
PYTHONPATH=/home/prokop/git/pyscf python3 -u expamples_prokop/hermite_radial_study.py matrix --error-smooth 0
```

Output PNGs: `matrix_{shell}_{cubic|quintic}_power_b1.png`, `f32_{shell}_{cubic|quintic}_power_b1.png`.

---

## 7. Next steps (production GPU)

- [ ] Add `hermite_eval_quintic_node()` + `float4` path to `kernels.cl`
- [ ] f32 packing strategy for large knot derivatives (scale or store `d·h`, `c·h²`)
- [ ] Profile register / local memory vs cubic on RTX 3090
- [ ] Optional: quintic only for high-l shells or when node count dominates VRAM
- [ ] Wire analytic quintic tables into `OpenCLAOHermiteEvaluator` / XC path and e2e parity tests

---

## Appendix: tangents modes (renamed from `fit`)

| Old name | Current | Meaning |
|----------|---------|---------|
| `exact` | `analytic` | Exact GTO `R′` (and `R″` for quintic) at knots |
| `midpoint` | *(removed)* | Alias → `quadrature` in `normalize_tangents()` |
| `quadrature` | `quadrature` | LSQ on interior collocation; `n_quad` default 5, `reg` default 1e-6 |

Default when unspecified: `quadrature` (both orders).
