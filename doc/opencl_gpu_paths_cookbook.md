# OpenCL GPU/CPU execution paths — cookbook

**Purpose:** map every sub-step knob, which combinations are valid, which paths work well together, and named **profiles** for production use.

**Python profiles:** `pyscf/OpenCL/gpu_profiles.py` — `GPU_PROFILES`, `apply_gpu_profile(mf, name)`.

**Long-form benchmarks:** `doc/rho_vmat_vxc_GPU_optimization.report.md` (Parts 6–10).

---

## 1. SCF integration overview

```
mf.kernel()
  └─ get_veff()  [per SCF cycle]
       ├─ get_j()     ← DF Coulomb (J); PBE has no K
       └─ nr_rks_*()  ← XC: ρ → PBE/libxc → vmat
```

Two independent backend switches:

| Switch | Attribute | Values | Meaning |
|--------|-----------|--------|---------|
| XC backend | `mf.backend` | `1` CPU, `2` GPU, `3` both (compare) | ρ/vmat/XC path |
| DF J backend | `mf.with_df.backend` | `1` CPU, `2` GPU, `3` both | RI Coulomb GEMM |

Setup (once, before SCF):

```python
from pyscf import gto, dft
from pyscf.OpenCL.gpu_profiles import apply_gpu_profile, list_profiles

mol = gto.M(atom='...', basis='cc-pVDZ')
mf = dft.RKS(mol, xc='PBE').density_fit()
apply_gpu_profile(mf, 'production_otf')   # sets backend, DF, conv_tol, setup_gpu
mf.kernel()
```

Or shorthand:

```python
mf = dft.RKS(mol, xc='PBE').density_fit()
mf.backend = 2
mf.setup_gpu(profile='production_otf')
mf.kernel()
```

---

## 2. Sub-step variant table

Each row is one **independent knob**. In principle any combination is *attempted*; rows marked **⚠** have constraints.

### 2.1 XC path (ρ + vmat)

| Variant | `xc_path` / entry | AO on GPU at setup? | Per-cycle ρ | Per-cycle vmat | Memory (GGA, benzene cc-pVDZ grid 3) |
|---------|-------------------|---------------------|-------------|----------------|--------------------------------------|
| **Hermite OTF** | `onthefly` | No (tables ~0.2 MB) | `rho_gga_pair` (Hermite in kernel) | `vmat_gga_pair` | negligible |
| **Precomp row-major** | `precomputed`, `fused='tiled'` | Yes χ[**iG**, iAO] | `rho_gga_precomp_pair` | `vmat_gga_precomp_pair` | ~262 MB χ |
| **Precomp coalesced** | `precomputed`, `fused='coalesced'` | Yes χ[**iAO**, iG] | `rho_gga_precomp_coalesced_pair` | `vmat_gga_precomp_coalesced_pair` | ~262 MB χ |
| **Precomp radial** | `precomputed`, `fused='radial_precomp'` | R,dR only (~62 MB) | `rho_gga_radial_precomp_pair` | `vmat_gga_precomp_coalesced_pair` † | ~62 MB + χ for vmat |
| **Precomp blocked (legacy)** | `precomputed`, `fused=False` | Yes | blocked host loops | blocked | same χ as tiled |

† Radial path: ρ uses R,dR; vmat still gathers from Hermite χ (built at setup via `ao_proj`).

### 2.2 AO setup (precomputed path only)

| `ao_proj` | How χ is filled | Accuracy vs CPU GTO | Setup cost |
|-----------|-----------------|---------------------|------------|
| `'auto'` | Hermite GPU if lmax≤3, else CPU `eval_ao` | ~1e-4 rel on ∇ρ; vxc ~3e-6 | fast GPU |
| `'hermite_gpu'` | `eval_ao_hermite_cart_deriv1_tiled` + c2s | same as auto Hermite | fast GPU |
| `'cpu'` | PySCF `eval_ao` on host, upload | best (~2.6e-6 vxc) | slow (CPU bound) |

**⚠** OTF path ignores `ao_proj` and `fused` — no χ upload.

### 2.3 XC functional evaluation (PBE wv)

| `xc_eval` | Where PBE runs | ρ/wv device residency |
|-----------|----------------|----------------------|
| `'gpu'` | OpenCL `pbe.cl` (f32 default) | stays on GPU |
| `'cpu'` | libxc on host | ρ D2H, wv H2D each cycle |

| `gpu_xc` | Precision (when `xc_eval='gpu'`) |
|----------|----------------------------------|
| `'auto'` / `'pbe_f32'` | float32 PBE (production) |
| `'pbe_f64'` | float64 PBE (slower, tighter wv) |

**⚠** GPU PBE path: unmodified PBE GGA only. Meta-GGA / hybrid / range-separated → use CPU XC or extend kernels.

### 2.4 DF Coulomb (J)

| `mf.with_df.backend` | Implementation | Typical benzene (per `get_j`) |
|----------------------|----------------|-------------------------------|
| `1` | CPU `df_jk` GEMM | ~32 ms |
| `2` | GPU `df_jk` OpenCL | ~2 ms |
| `3` | both + compare | debug only |

PBE RKS uses **J only** (K not needed for closed-shell).

### 2.5 Tile / launch config (all GPU kernels)

| Env / module | Effect |
|--------------|--------|
| `OPENCL_NPTILE`, `OPENCL_NATILE`, `OPENCL_WGS_VMAT` | grid/atom tile sizes in kernels |
| `pyscf/OpenCL/tile_config.py` | defaults and sweeps |

Not a separate “path” — applies on top of any variant above.

---

## 3. Compatibility matrix

**Legend:** ✓ arbitrary OK · **→** recommended together · **⚠** invalid or wasteful

|  | Hermite OTF | coalesced | radial | tiled (legacy) |
|--|-------------|-----------|--------|----------------|
| `ao_proj` none / N/A | ✓ | **⚠** need ao_proj | **⚠** need hermite_gpu | **⚠** need ao_proj |
| `ao_proj='cpu'` | **⚠** ignored | ✓ (small mol) | **⚠** wasteful | ✓ |
| `ao_proj='hermite_gpu'` | **⚠** ignored | ✓ **→** | ✓ **→** | ✓ |
| `xc_eval='gpu'` | ✓ **→** | ✓ **→** | ✓ **→** | ✓ |
| `xc_eval='cpu'` | ✓ (debug) | ✓ (debug) | ✓ (debug) | ✓ |
| `df_backend=2` (GPU J) | ✓ | ✓ | ✓ | ✓ |
| LDA functional | ✓ | **⚠** coalesced/radial N/I | **⚠** | tiled OK |
| Meta-GGA / hybrid | CPU XC only | CPU XC only | CPU XC only | CPU XC only |

### Synergy rules (reasonable paths)

1. **Full GPU XC chain:** if ρ is on GPU, use `xc_eval='gpu'` so wv stays on device; vmat kernels expect GPU buffers.
2. **OTF:** no χ precomputation; best for **medium/large** systems where χ would exceed GPU RAM or setup dominates.
3. **Coalesced + Hermite AO:** χ[**iAO**, iG] matches vmat gather; prefer over row-major `tiled`.
4. **Radial:** low memory ρ; pair with `ao_proj='hermite_gpu'` for vmat χ at setup.
5. **Exact GTO χ:** `ao_proj='cpu'` only for **small** molecules (setup ∝ nao×ngrids on CPU).
6. **GPU DF J + tight `conv_tol_grad=1e-5`:** often **fails to converge** — f32 XC gradient floor ~1e-4; use `fast_full_gpu` tolerances or CPU J.

---

## 4. Named profiles (`GPU_PROFILES`)

| Profile | XC path | fused | ao_proj | xc_eval | DF J | conv_tol | conv_tol_grad | Typical vxc err | Energy / SCF |
|---------|---------|-------|---------|---------|------|----------|---------------|-----------------|--------------|
| `cpu_reference` | — | — | — | CPU | CPU | 1e-8 | 1e-5 | 0 (ref) | reference |
| `debug_compare` | OTF | — | — | gpu | both | 1e-8 | 1e-5 | ~3e-6 | debug, max_cycle=5 |
| `debug_xc_libxc` | precomp | coalesced | auto | **cpu** | CPU | 1e-8 | 1e-5 | ~3e-6 | converges |
| **`production_otf`** | OTF | — | — | gpu | CPU | 1e-8 | 1e-5 | ~3e-5 | default OTF (ρ+vmat Hermite) |
| **`production_otf_radial_vmat`** | OTF | — | — | gpu | CPU | 1e-8 | 1e-5 | ~3e-5 | **fastest per-cycle** — OTF ρ + radial vmat |
| `production_otf_quintic` | OTF | — | — | gpu | CPU | 1e-8 | 1e-5 | ~3e-5 | quintic spline; half setup table |
| `production_coalesced` | precomp | coalesced | auto | gpu | CPU | 1e-8 | 1e-5 | ~3e-6 | small/fixed geom |
| `production_radial` | precomp | radial | hermite_gpu | gpu | CPU | 1e-8 | 1e-5 | ~3e-6 | low χ memory |
| `production_gto_exact` | precomp | coalesced | **cpu** | gpu | CPU | 1e-8 | 1e-5 | ~2.6e-6 | small mols only |
| **`fast_full_gpu`** | OTF | — | — | gpu | **GPU** | **1e-6** | **1e-4** | ~8e-6/veff | ~7e-5 Ha (~0.04 kcal/mol) |
| `legacy_tiled_rowmajor` | precomp | tiled | auto | gpu | CPU | 1e-8 | 1e-5 | ~3e-6 | use coalesced instead |

**Default:** `production_otf` (general). **Fastest XC per cycle (benzene):** `production_otf_radial_vmat` — see `doc/GPU_benchmark.md`.

### Hybrid OTF ρ + radial vmat

```python
apply_gpu_profile(mf, 'production_otf_radial_vmat')
# equivalent manual:
mf.setup_gpu(xc_path='onthefly', xc_eval='gpu', vmat_mode='radial_precomp')
```

- ρ: `rho_gga_tiled` (OTF Hermite, same as `production_otf`)
- vmat: `vmat_gga_radial_precomp_pair` (`R,dR` gathered at setup via `build_radial_on_grid_gpu`)
- Requires GGA; radial metadata buffers must stay alive in `plan.otf` (see topical audit)

### Accuracy notes (benzene cc-pVDZ, grid level 3, PBE, DF)

| Quantity | `production_otf` / coalesced / radial | `fast_full_gpu` |
|----------|---------------------------------------|-----------------|
| Single-shot ‖vxc_gpu − vxc_cpu‖∞ | ~2.6–3.3×10⁻⁶ | ~8×10⁻⁶ |
| SCF energy vs `cpu_reference` | ~10⁻⁶ Ha when converged | ~7×10⁻⁵ Ha |
| SCF convergence @ default tol | yes | yes @ relaxed tol |
| SCF @ conv_tol_grad=1e-5 + GPU J | yes (CPU J) | **no** (use 1e-4) |

---

## 5. Decision flowchart

```mermaid
flowchart TD
    A[Start: RKS + DF + PBE] --> B{Molecule size / chi RAM?}
    B -->|large or unknown| C[production_otf]
    B -->|small, fixed geometry| D{Need exact GTO chi?}
    D -->|yes| E[production_gto_exact]
    D -->|no| F{chi fits in GPU RAM?}
    F -->|yes, want fastest precomp rho| G[production_coalesced]
    F -->|tight on RAM| H[production_radial]
    C --> I{Optimize veff XC speed?}
    G --> I
    H --> I
    I -->|yes, same accuracy| L[production_otf_radial_vmat]
    I -->|yes, relaxed SCF tol OK| J[fast_full_gpu]
    I -->|no| K[keep CPU DF J + default tol]
```

---

## 6. Manual configuration (without profiles)

```python
# Production OTF (equivalent to profile)
mf = dft.RKS(mol, xc='PBE').density_fit()
mf.backend = 2
mf.with_df.backend = 1
mf.conv_tol, mf.conv_tol_grad = 1e-8, 1e-5
mf.setup_gpu(xc_path='onthefly', xc_eval='gpu', gpu_xc='auto')
mf.kernel()

# Precomp coalesced
mf.setup_gpu(xc_path='precomputed', fused='coalesced', ao_proj='auto', xc_eval='gpu')

# Hybrid OTF rho + radial vmat (fastest per-cycle XC on benzene)
mf.setup_gpu(xc_path='onthefly', xc_eval='gpu', vmat_mode='radial_precomp')

# Full GPU with relaxed SCF
mf.with_df.backend = 2
mf.conv_tol, mf.conv_tol_grad = 1e-6, 1e-4
mf.setup_gpu(profile='fast_full_gpu')
```

### Stage timing

Per-stage wall + OpenCL event times (`plan.last_timing`):

```python
plan = mf._xc_gpu_plan
_, _, vxc = plan.nr_rks_hermite_onthefly(dm, profile=True)
print(plan.last_timing)  # gpu_rho, gpu_rho_cl, gpu_vmat, gpu_vmat_cl, gpu_total_cl, ...
```

Benzene benchmark table: `expamples_prokop/profile_xc_stages_benzene.py` → `doc/GPU_benchmark.md`.

SCF-accumulated timers (coarser):

```python
mf._gpu_profile = True
mf.kernel()
print(mf._gpu_timing_acc)  # rho, xc, vmat, sync per get_veff
```

**Profiling rules:** queue must use `PROFILING_ENABLE`; always `queue.finish()` before wall clock; use `gpu_*_cl` keys for true kernel time. cProfile under-reports GPU work.

---

## 7. Test / benchmark scripts

| Script | What it exercises |
|--------|-------------------|
| `expamples_prokop/test_opencl_xc_full_gpu_parity.py` | Step-wise ρ, wv, vmat parity |
| `expamples_prokop/test_opencl_xc_e2e_mols.py` | Speed + accuracy, arbitrary XYZ |
| `expamples_prokop/test_opencl_xc_cpu_threads.py` | CPU thread scaling vs GPU |
| `expamples_prokop/profile_gpu_scf.py` | Full converged SCF, cProfile + timers |
| `expamples_prokop/profile_xc_stages_benzene.py` | Per-stage wall vs CL timing; hybrid path comparison |
| `expamples_prokop/test_quintic_rho_otf.py` | Quintic vs cubic OTF ρ parity |

```bash
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=15 python3 -u expamples_prokop/profile_gpu_scf.py --mode cpu gpu_otf gpu_full
```

---

## 8. Known limitations

- `fused='gemm'` precomp path: possible OpenCL queue errors — avoid in production.
- `gpu_full` + `conv_tol_grad=1e-5`: DIIS stuck at ‖g‖~1e-4 (f32 XC).
- `ao_proj='cpu'` on pentacene-scale χ: multi-GB upload; tests skip >4 GB.
- Hermite ∇ρ pointwise error ~1e-4; PBE vxc integration still ~1e-6.
- cProfile under-reports GPU time; use `plan.last_timing` (profile=True) or `_gpu_timing_acc`.

---

## 9. Quick reference — path labels in reports

See **Path naming glossary** in `doc/rho_vmat_vxc_GPU_optimization.report.md` § intro table.

| Report label | Profile equivalent |
|--------------|-------------------|
| `gpu_hermite_otf` | `production_otf` |
| `gpu_otf_radial_vmat` | `production_otf_radial_vmat` |
| `gpu_otf_quintic` | `production_otf_quintic` |
| `gpu_precomp_coalesced` | `production_coalesced` |
| `gpu_precomp_radial` | `production_radial` |
| `gpu_precomp_tiled` | `legacy_tiled_rowmajor` |
| `gpu_full` (profile script) | `fast_full_gpu` |
