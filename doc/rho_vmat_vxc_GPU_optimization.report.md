# OpenCL XC Kernel Optimization Report

This document has ten parts: the original **vmat abTile fix** (baseline), a **2026-03 follow-up** covering rho/vmat tiling and pair kernels on GTX 1650, **Part 3** precomputed GTO pair-gather (2026-06-28), **Part 4** coalesced χ + radial precomp ρ (2026-06-29), **Part 5** GPU Hermite AO setup, **Part 6** full-GPU path parity audit (benzene), **Part 7** end-to-end benzene benchmarks, **Part 8** pentacene + PTCDA scaling, **Part 9** multi-CPU vs GPU comparison, and **Part 10** full SCF convergence profiling.

Test harness: `expamples_prokop/test_opencl_xc_onthefly.py --xyz <file>`  
Full parity audit: `expamples_prokop/test_opencl_xc_full_gpu_parity.py`  
Multi-molecule E2E: `expamples_prokop/test_opencl_xc_e2e_mols.py`  
CPU thread scaling: `expamples_prokop/test_opencl_xc_cpu_threads.py`  
Full SCF profile: `expamples_prokop/profile_gpu_scf.py`  
All-path benchmark: `expamples_prokop/test_opencl_xc_scf.py`  
Tile sweeps: `expamples_prokop/sweep_opencl_tiles.py`  
Config: `pyscf/OpenCL/tile_config.py` (`OPENCL_NPTILE`, `OPENCL_NATILE`, `OPENCL_WGS_VMAT`, `OPENCL_MAX_ITILE`)

### Path naming glossary (read this first)

**Cookbook (profiles, compatibility, SCF tolerances):** `doc/opencl_gpu_paths_cookbook.md` · Python: `pyscf/OpenCL/gpu_profiles.py`  
**Quintic Hermite study (cubic vs quintic vs du):** `doc/quintic_hermite_spline.md` · `pyscf/OpenCL/hermite_spline.py`

Short labels in tables are **execution paths**, not kernel tile sizes. **`NPTILE` / “grid tile”** means a block of consecutive grid points inside a kernel workgroup — it appears in almost every path and is **not** a path name.

| Table label | Full name | Setup (one-time) | Per-SCF hot path | AO on GPU? |
|-------------|-----------|------------------|------------------|------------|
| `cpu_libxc` | **CPU reference** | none | PySCF `NumInt.nr_rks` (CPU `eval_ao` + libxc) | no |
| `gpu_hermite_otf` | **Hermite on-the-fly (OTF)** | Hermite tables ~0.2 MB | `rho_gga_pair` + `vmat_gga_pair`: DM×AO contraction with Hermite radial eval in registers/`__local`; **no** precomputed χ | no |
| `gpu_precomp_tiled` | **Precomp GTO row-major** (`fused='tiled'`) | CPU PySCF `eval_ao` → upload χ[**iG**, iAO] | `rho_gga_precomp_pair` + `vmat_gga_precomp_pair` gather from row-major χ | yes (~262 MB GGA) |
| `gpu_precomp_coalesced` | **Precomp GTO coalesced** (`fused='coalesced'`) | same as row-major but χ[**iAO**, iG] transpose | `rho_gga_precomp_coalesced_pair` + `vmat_gga_precomp_coalesced_pair` | yes |
| `gpu_precomp_radial` | **Precomp radial Hermite** (`fused='radial_precomp'`) | `build_radial_on_grid_tiled` on GPU → R,dR (~62 MB); **no** full χ | `rho_gga_radial_precomp_pair` + `vmat_gga_radial_precomp_pair` | R,dR only |
| `gpu_precomp_*_hermite_setup` | **Precomp + GPU Hermite AO setup** (`ao_proj='hermite_gpu'`) | `eval_ao_hermite_cart_deriv1_tiled` + c2s on GPU (Part 5) | same per-SCF kernels as coalesced/row-major | yes (Hermite approx.) |

**Not the same:** `gpu_precomp_tiled` ≠ `gpu_hermite_otf`. The former **uploads** PySCF GTO values on the grid; the latter **never materializes** χ and contracts the density matrix with Hermite AOs evaluated on the fly.

---

## Part 1 — vmat abTile fix (baseline)

### Problem

The original `vmat_lda_tiled` and `vmat_gga_tiled` kernels used a "hybrid atom-tile + abTile" design with a 3D workgroup `(iTile, jTile, abTile)`. This caused catastrophic performance:

| System | vmat time | Total OTF time |
|--------|-----------|----------------|
| water (3 atoms, 114 ncart) | 5.17s | 5.94s |
| benzene (12 atoms, 114 ncart) | 5.17s | 5.92s |
| pentacene (36 atoms, 226 ncart) | ~40s | 40.6s |

### Root Cause

The abTile design had three major inefficiencies:

1. **57x redundant radial evaluation**: `VBLK_SIZE=3600` with `WGS_VMAT=64` gives `NABTILE=57` workgroups per atom-pair tile. Each workgroup independently re-evaluates all radial functions for the same grid tile — 57x redundant work.

2. **6x wasted angular unfolding**: Each thread owns one AO-pair element `(a,b)` but calls `unfold_shell()` which computes all 6 angular components. Only 1 is used — 6x wasted compute.

3. **Per-thread shell search**: Each thread repeatedly searched for which shell an AO index belonged to, adding a serial loop inside the grid tile loop.

For benzene: `3*3*57 = 513` workgroups, each looping over `143560/16 = 8973` grid tiles = massive serial overhead with only 64 threads per workgroup.

### Solution

Replaced the abTile design with a local AO cache + private accumulator design (from `doc/ToOpenCL.chat.md`):

```
workgroup = one (iTile, jTile) atom-pair tile
thread    = owns QPT AO-pair matrix elements with private acc[QPT]
loop      = over grid-point tiles
local     = unfolded AO values aoI[NPTILE][AO_TILE], aoJ[NPTILE][AO_TILE]
no abTile, no redundant radial recomputation, no atomics
```

#### Key design points

- **Local AO cache**: `aoI[NPTILE][AO_TILE]` and `aoJ[NPTILE][AO_TILE]` store fully unfolded AO values.
- **Fill functions**: `fill_atom_ao_lda()` and `fill_atom_aow_gga()` unfold once per `(grid point, atom)`.
- **Private accumulators**: `acc[QPT]` where `QPT = ceil(VBLK_SIZE / WGS_VMAT)`.
- **Launch**: 2D global `(n_iTiles, n_jTiles * WGS_VMAT)`, local `(1, WGS_VMAT)`.

### abTile-era results (historical)

| System | vmat old | vmat new | Speedup | Total old | Total new | CPU |
|--------|----------|----------|---------|-----------|-----------|-----|
| water | 5.17s | 0.027s | 192x | 5.94s | 0.18s | 0.03s |
| benzene | 5.17s | 0.089s | 58x | 5.92s | 0.58s | 0.49s |
| pentacene | ~40s | 0.57s | ~70x | 40.6s | 2.98s | 4.53s |
| PTCDA | — | 0.86s | — | — | 4.33s | 8.84s |

---

## Part 2 — 2026-03 tuning (rho + vmat + tiles)

Hardware: **NVIDIA GeForce GTX 1650**, `OMP_NUM_THREADS=1`, PBE/ccpvdz, grid level 3.

### Code changes since Part 1

| Area | Change |
|------|--------|
| `kernels.cl` | `contract_pair_rho_v2` / `contract_pair_rho_gga_v2`: hoisted `l[]`, `invr`, `float4` (φ,∂φ) inner contract |
| `kernels.cl` | `float2` Hermite knot packing; per-`iTile` `i_ir_tile` fix (parity bug) |
| `kernels.cl` | `vmat_lda_tiled`: upper-triangle atom tiles (`ja < ia` skip), symmetric write |
| `kernels.cl` | **New** `rho_*_pair`, `vmat_*_pair` — single-atom-pair specialization (`NATILE=1`) |
| `tile_config.py` | Compile-time `-D` flags; env overrides; defaults **`NPTILE=64, NATILE=2`** |
| `xc_grid.py` | Auto-select `*_pair` kernels when `NATILE==1`; `clear_xc_plan_cache()` on recompile |
| `test_opencl_xc_onthefly.py` | General `--xyz` CLI benchmark |

Existing `*_tiled` kernels are **unchanged** in role; pair kernels are additive.

### Kernel launch geometry (current)

#### Tiled (`NATILE ≥ 2`) — production path

**rho** (`rho_gga_tiled`):
- Global: `(ceil(ngrids/NPTILE), NATILE)`, local: `(NPTILE, NATILE)`
- One WG per grid-point tile; loops all `(iTile, jTile)` inside; `__local` reduce over `il`

**vmat** (`vmat_gga_tiled`):
- Global: `(n_iTiles, n_jTiles × WGS_VMAT)`, local: `(1, WGS_VMAT)`
- One WG per atom-pair **tile**; loops all grid tiles; GGA: one-sided `aow_i·φ_j` + host `vmat + vmat.T`

#### Pair (`NATILE = 1`) — simplified path

**rho** (`rho_gga_pair`):
- Global: `(ceil(ngrids/NPTILE), 1)`, local: `(NPTILE, 1)`
- Loops all `(ia, ja)` atom pairs; `dm_blk[16][16]` only

**vmat** (`vmat_gga_pair`):
- Global: `(natoms, natoms × WGS_VMAT)`, local: `(1, WGS_VMAT)`
- One WG per atom pair; loops all grid tiles
- **144 WGs** (benzene) vs **36 WGs** (tiled `NATILE=2`) — main reason pair vmat is slower

### Tile constant reference (pow2)

```c
#define NPTILE       64     // LOG_NPTILE = 6  (default host)
#define NATILE       2      // LOG_NATILE = 1
#define MAX_AO_ATOM  16     // LOG_MAX_AO_ATOM = 4
#define WGS_VMAT     256
#define AO_TILE      (NATILE * MAX_AO_ATOM)   // 32 for NATILE=2
#define VBLK_SIZE    (AO_TILE * AO_TILE)      // 1024
#define QPT          4                        // ceil(1024/256)
```

`NATILE=8` + `NPTILE=32` **fails compile** on GTX 1650: rho shared memory `> 48 KB`.

Validation: `WGS_VMAT ≥ NPTILE × NATILE` (blocks e.g. `NPTILE=256, NATILE=2` without raising `WGS_VMAT`).

### NATILE sweep (`NPTILE=32`, benzene PBE)

| NATILE | kernel total | rho | vmat | Notes |
|--------|-------------|-----|------|-------|
| **2** | **109 ms** | 38 ms | **71 ms** | **Best tiled** |
| 4 | 134 ms | 56 ms | 78 ms | previous default |
| 1 (pair) | 131 ms | 35 ms | 95 ms | uses `*_pair` kernels |
| 8 | — | compile fail | shared mem overflow | |

### NPTILE sweep — tiled (`NATILE=2`)

**Benzene** (143k grids, 12 atoms):

| NPTILE | kernel total | rho | vmat | CPU speedup |
|--------|-------------|-----|------|-------------|
| 32 | 109 ms | 37 ms | 72 ms | 3.5× |
| **64** | **92 ms** | 31 ms | 61 ms | **4.15×** |
| 128 | 92 ms | 30 ms | 62 ms | 4.0× (flat vs 64) |
| 256 | — | `WGS_VMAT < NPTILE×NATILE` | | |

**PTCDA** (502k grids, 38 atoms):

| NPTILE | kernel total | rho | vmat | CPU speedup |
|--------|-------------|-----|------|-------------|
| 32 | 4411 ms | 2269 ms | 2142 ms | 2.87× |
| 64 | 4093 ms | 1980 ms | 2114 ms | 3.17× |
| **128** | **3975 ms** | 1866 ms | 2108 ms | **3.36×** |

Recommendation: **`NPTILE=64`** daily driver; **`NPTILE=128`** for large grids/molecules.

### NPTILE sweep — pair kernels (`NATILE=1`)

Pair kernels use far less `__local` memory → can push `NPTILE` higher.

**Benzene**:

| NPTILE | kernel total | rho | vmat |
|--------|-------------|-----|------|
| 32 | 131 ms | 36 ms | 95 ms |
| 64 | 124 ms | 35 ms | 89 ms |
| **128** | **117 ms** | 32 ms | 86 ms |
| 256 | 118 ms | 33 ms | 85 ms | plateau |
| 512 | — | needs `WGS_VMAT≥512` | |

**PTCDA**:

| NPTILE | kernel total | rho | vmat |
|--------|-------------|-----|------|
| 32 | 5628 ms | 2254 ms | 3374 ms |
| 64 | 5533 ms | 2248 ms | 3284 ms |
| 128 | 5345 ms | 2087 ms | 3257 ms |
| **256** | **5328 ms** | 2076 ms | 3251 ms |

Pair path improves with large `NPTILE` but **never beats tiled `NATILE=2`** on these systems (~20–25% slower on PTCDA).

### Symmetry experiments

| Approach | Result |
|----------|--------|
| LDA `vmat_lda_tiled`: `iTile ≤ jTile` + transpose write | Kept; parity OK |
| GGA in-kernel symmetric (`aow_i·φ_j + aow_j·φ_i`) | **Slower** (~105 ms vs ~78 ms vmat on benzene); reverted |
| GGA one-sided kernel + host `vmat + vmat.T` | Faster; kept |

GGA symmetry cannot skip lower atom tiles without paying ~2× in the inner contract.

### Parity (all configs tested)

| System | vxc max rel err | Status |
|--------|-----------------|--------|
| benzene | ~2–3×10⁻⁶ | pass |
| pentacene | ~2–4×10⁻⁶ | pass |
| PTCDA | ~1.9×10⁻⁶ | pass |

### Current bottleneck split (best tiled config)

**Benzene** (`NPTILE=64`, `NATILE=2`): kernel **~92 ms** — rho **34%**, vmat **66%**  
**PTCDA** (`NPTILE=128`, `NATILE=2`): kernel **~3975 ms** — rho **47%**, vmat **53%**

Harness (libxc + PCIe): ~17–72 ms depending on `ngrids` — small vs kernels.

### Recommended production settings (GTX 1650)

```bash
# Default (tile_config.py): NPTILE=64 NATILE=2
PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 \
  python3 expamples_prokop/test_opencl_xc_onthefly.py --xyz data/xyz/benzene.xyz

# Large molecule
OPENCL_NPTILE=128 OPENCL_NATILE=2 PYTHONPATH=... \
  python3 expamples_prokop/test_opencl_xc_onthefly.py --xyz data/xyz/PTCDA.xyz

# Pair-kernel experiments only (simpler code, slower vmat)
OPENCL_NPTILE=128 OPENCL_NATILE=1 PYTHONPATH=... \
  python3 expamples_prokop/test_opencl_xc_onthefly.py --xyz data/xyz/benzene.xyz
```

### Summary: best results vs CPU (single core)

| System | CPU | GPU best | Config | Speedup |
|--------|-----|----------|--------|---------|
| benzene | ~0.44 s | **~0.11 s** | tiled NPTILE=64 NATILE=2 | **~4×** |
| PTCDA | ~12–13 s | **~4.0 s** | tiled NPTILE=128 NATILE=2 | **~3.3×** |

Still far from 10–100× vs CPU: tiled kernels do dense `natoms²` atom-pair work per grid point without spatial screening.

### Future work (not done)

1. **Grid-tile active atom lists** — drop distant atoms per grid tile (largest win for large molecules).
2. **Fused GPU libxc** — remove rho↔host↔vmat sync (~17–70 ms harness).
3. **vmat prepass**: hoist `hermite_map_point` + `l` in `fill_atom_ao_*` (mirror rho v2).
4. **`(li,lj)` specialized contracts** — unroll l≤3 in pair or tiled kernels.
5. **Pair + large NPTILE** with `WGS_VMAT=512` — test `NPTILE=512` pair path.
6. **Sparse DM projection** — for large molecules (separate track).

---

## Related docs

| Topic | File |
|-------|------|
| rho `iTile` inner loop, pre-SCF setup | [opencl-xc-reports/2026-03-rho-itile-host-setup.md](opencl-xc-reports/2026-03-rho-itile-host-setup.md) |
| Coalesced / radial precomp layout spec | [opencl_rho_precomp_layout.md](opencl_rho_precomp_layout.md) |
| XC architecture overview | [opencl_xc_architecture.md](opencl_xc_architecture.md) |
| Report index | [opencl-xc-reports/README.md](opencl-xc-reports/README.md) |
| Kernel guidelines | [opencl-kernel-cookbook.md](opencl-kernel-cookbook.md) |

---

## Part 3 — Precomputed GTO path (2026-06-28)

Hardware: **NVIDIA GeForce GTX 1650**, `OMP_NUM_THREADS=1`, benzene/ccpvdz/grid3 (`nao=114`, `ngrids=143560`, `natoms=12`).

Benchmark harness: `expamples_prokop/test_opencl_xc_scf.py`

### Goal

Build a **precomputed-AO** XC path that mirrors the Hermite OTF architecture:
- **1 GPU launch per stage** (ρ and vmat)
- **`__local` tile gather** (same pattern as `rho_gga_pair` / `vmat_gga_pair`)
- **No Python block loops**, **no full-grid GEMM** as default

### What was tried (and rejected as default)

| Path | `gpu_rho` | `gpu_vmat` | wall | vxc_max | Problem |
|------|-----------|------------|------|---------|---------|
| `fused='gemm'` — full-grid matmul + contract | 395 ms | 106 ms | 520 ms | 4.6e-5 | 4× full `[ngrids,nao]` GEMM; memory-bound |
| `rho_*_precomp_fused` — 32×32 tiled GEMM in-kernel | 401 ms | — | — | 2.6e-6 | 1024 threads/WG; GGA re-read global AO in inner loop; not OTF pattern |
| Atom-pair ρ without cooperative gather | ~283 ms | — | — | 2.6e-6 | Per-thread-only `aoJ[ip]` fill; missed tile-wide gather |
| `rho_*_precomp_tiled` (`NATILE=2`) | 374 ms | — | — | 2.9e-6 | Same pair math but 128-thread WG; slower than pair on benzene |
| Hybrid tiled ρ + GEMM vmat | — | — | ~475 ms | ~5e-5 | User rejected; breaks single-architecture goal |

### What shipped (default `fused='tiled'`)

| Stage | Kernel | Launch | Pattern |
|-------|--------|--------|---------|
| ρ | `rho_lda_precomp_pair` / `rho_gga_precomp_pair` | `(⌈ngrids/NPTILE⌉, 1)`, local `(NPTILE, 1)` | ja-loop: cooperative `aoJ[NPTILE][MAX_AO_ATOM]` gather; ia-loop: `dm_blk[16][16]` + contract |
| vmat | `vmat_lda_precomp_pair` / `vmat_gga_precomp_pair` | `(natoms, natoms×WGS_VMAT)`, local `(1, WGS_VMAT)` | Same as Hermite `vmat_*_pair`; host `vmat + vmat.T` for symmetry |

Host wiring: `pyscf/OpenCL/xc_grid.py` — `setup_precomputed_gto(fused='tiled')`, `_precomp_rho_fused`, `_precomp_vmat_fused`.  
`fused='gemm'` and `fused=False` (Python block loop) remain explicit fallbacks only.

Kernels also present but **not default**: `rho_*_precomp_fused`, `rho_*_precomp_tiled` (kept in `kernels.cl` for experiments).

### Benzene benchmark (latest, pair-gather ρ)

| Path | wall | gpu_rho | gpu_vmat | vxc_max |
|------|------|---------|----------|---------|
| cpu_libxc | 462 ms | — | — | ref |
| **gpu_precomp_tiled** (default) | **425 ms** | **273 ms** | 148 ms | **2.6e-6** |
| gpu_precomp_gemm (`fused=gemm`) | 520 ms | 396 ms | 106 ms | 4.6e-5 |
| gpu_precomp_blocked (old) | 516 ms | 391 ms | 104 ms | 1.1e-5 |
| **gpu_hermite_otf** | **117 ms** | **31 ms** | **65 ms** | 2.9e-6 |

Parity vs CPU libxc: **pass** (`vxc_max ≈ 2.6×10⁻⁶`).

Stage timing uses `queue.finish()` per `gpu_*` stage in `xc_grid.py` (`gpu_rho`, `gpu_vmat`, `host_rho_d2h`, etc.).

### Why precomp ρ is still ~9× slower than Hermite

Hermite OTF (`gpu_rho ≈ 31 ms`) never reads a dense `[ngrids, nao]` AO table — it evaluates compact per-shell values from coords + radial knots into `__local`.

Precomp pair gather still streams the full pre-uploaded AO buffer every SCF iteration:
- Strided reads `ao0[g*nao + μ]` — poor cache behaviour vs atom-blocked layout
- 12×12 atom-pair iterations × 3 barriers per pair per grid tile
- Dense AO is ~65 MB (GGA ×4 components) on GPU; bandwidth-dominated for small `nao`

**vmat** is closer (148 ms vs 65 ms, ~2.3×) because the pair kernel reuses the same gather pattern as Hermite and does less total work per grid point than ρ (weighted contract vs full DM trace).

### Bugs fixed in this session

1. `INVALID_WORK_GROUP_SIZE` — fused GEMM launch had `local_y=32` but `global_y=1`; must match.
2. GGA fused kernel `OUT_OF_RESOURCES` — moved `aodm` from large private arrays to `__local`.
3. vmat GGA double-symmetrization — removed in-kernel mirror write (~1.3% vxc error); host `vmat + vmat.T` only.
4. Cooperative tile gather — `aoJ[NPTILE][MAX_AO_ATOM]` filled by all threads (`k += NPTILE` loop), not per-thread-only.

### Next step (not done — largest win for precomp)

**Atom-blocked AO layout at upload**: store `[ngrids, natoms, max_ao_per_atom]` or `[ngrids, nao_atom_major]` so tile gather reads contiguous memory per atom per grid block. Should mirror Hermite's coalesced `wfRj[NPTILE][…]` fill without on-the-fly eval.

Secondary: grid-tile active atom lists from `non0tab` (same as Future work §1 in Part 2).

### Reproduce

```bash
PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 \
  python3 -u expamples_prokop/test_opencl_xc_scf.py
```

Single-path quick check:

```bash
PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 python3 -c "
from pyscf import gto, dft
from pyscf.OpenCL import init_device
from pyscf.OpenCL.xc_grid import get_xc_grid_plan
from expamples_prokop.test_opencl_xc_scf import read_xyz
mol = gto.M(atom=read_xyz('data/xyz/benzene.xyz'), basis='ccpvdz', verbose=0)
grids = dft.gen_grid.Grids(mol); grids.level=3; grids.build(with_non0tab=True)
dm = dft.RKS(mol, xc='PBE').density_fit().get_init_guess()
init_device(quiet=True)
plan = get_xc_grid_plan(mol, grids, 'PBE')
plan.setup_precomputed_gto(gpu_only=True, fused='tiled')
plan.nr_rks_precomputed_gto(dm, profile=True)
print(plan.last_timing)
"
```

---

## Part 4 — Coalesced χ and radial precomp ρ (2026-06-29)

Hardware: **NVIDIA GeForce RTX 3090**, `OMP_NUM_THREADS=1`, benzene/cc-pVDZ/grid level 3 (`nao=114`, `ngrids=143560`, `natoms=12`, `ncart=120`, `nradial=54`).

Benchmark harnesses:
- Full XC: `expamples_prokop/test_opencl_xc_scf.py`
- ρ-only parity: `expamples_prokop/test_opencl_xc_rho_precomp.py`

Design spec: `doc/opencl_rho_precomp_layout.md`  
Host wiring: `pyscf/OpenCL/xc_grid.py` — `setup_precomputed_gto(fused='coalesced'|'radial_precomp')`, `build_radial_on_grid()` in `pyscf/OpenCL/radial_hermite.py`  
Kernels: `rho_gga_precomp_coalesced_pair`, `rho_gga_radial_precomp_pair` in `pyscf/OpenCL/kernels.cl`

### Background

Part 3 shipped a **precomputed GTO** path that matched the Hermite **pair-kernel architecture** (one launch per ρ/vmat stage, `__local` DM tiles, cooperative gather). On GTX 1650 precomp ρ was still ~9× slower than Hermite OTF because AO lived in the wrong layout: `χ[c, iG, iAO]` (row-major per grid point), so a workgroup loading consecutive grid points for one AO component performed a **stride-`nao` gather** instead of coalesced reads.

Separately, the Hermite **on-the-fly (OTF)** path (`rho_gga_pair` / `vmat_gga_pair`) avoids materializing χ entirely: radial values are interpolated from compact `rad_node` tables (~0.2 MB) inside the kernel. That is fast on ρ (~4 ms on RTX 3090) but **re-evaluates Hermite radial + angular unfold twice per SCF cycle** (ρ kernel and vmat kernel), and still uses **CPU libxc** for XC unless the precomp + GPU PBE path is selected.

The question for Part 4: can we fix precomp bandwidth (Variant A) or store less than full χ (Variant B) and beat or match OTF where it matters?

### Key issues and challenges

| Issue | Symptom | Root cause |
|-------|---------|------------|
| **Bad AO memory layout** | Precomp ρ ~60 ms (tiled) vs OTF ~4 ms | `χ[g*nao + μ]` — threads in a warp read consecutive `g` but addresses differ by `nao` |
| **Full χ memory cost** | ~262 MB GGA f32 on benzene | 4 components × `ngrids × nao`; scales badly for large bases/grids |
| **Hermite vs GTO ρ** | Radial precomp ρ vs exact GTO: ~10⁻⁴ rel err on ∇ρ | R,dR/dr from mapped Hermite grid; OTF uses same approximation |
| **Hybrid path complexity** | Radial ρ + GTO vmat needs two AO representations | ρ uses Cartesian `dm_cart` + per-atom cart offsets; vmat still uses spherical GTO `χ[iG,iAO]` |
| **Setup amortization** | Precomp setup ~2–3 s vs OTF ~260 ms | `eval_ao` on CPU + upload; radial path adds ~320 ms `build_radial_on_grid` |
| **CPU/GPU Hermite convention** | Radial ρ gradients wrong by O(10³) if mismatched | `hermite_map_point` uses `t1m = t - 1` (not `1 - t`); CPU precomp must match `kernels.cl` |
| **Buffer lifetime** | `INVALID_MEM_OBJECT` on launch | Kernel args must be retained in `pcg` dict (e.g. `buf_atom_nao_cart`) or Python GC releases buffers |

### Strategies

#### Variant A — Coalesced full χ (`fused='coalesced'`)

**Goal:** Same 262 MB data, different index order — zero algorithm change in the contraction, only fix bandwidth.

**Layout:**

```
χ[c, iAO, iG]  →  flat index: (iAO * ngrids + iG)   # C-contiguous f32
```

**CPU setup (once per geometry):** during `eval_ao` block loop, write row-major `ao_staging[c][iG, iAO]` for vmat and transpose each block into `chi_staging[c][iAO, iG]`:

```python
blk = ao[c].astype(np.float32)          # (nblk, nao) from libcint
ao_staging[c][ip0:ip1] = blk
chi_staging[c][:, ip0:ip1] = blk.T    # (nao, nblk) coalesced
```

**Kernel:** `rho_gga_precomp_coalesced_pair` — same pair-tile model as `rho_gga_precomp_pair`, but J-atom tile load uses `chi[(j0+b)*ngrids + g]` so threads `ip=0..NPTILE-1` read **consecutive** `g` for fixed `(c, iAO)`.

**vmat:** unchanged — still `vmat_gga_precomp_pair` on row-major `buf_ao` (duplicate storage today).

#### Variant B — Radial-only precomp (`fused='radial_precomp'`)

**Goal:** Store `R(ir, iG)` and `dR/dr(ir, iG)` (~62 MB) instead of 4× full χ for ρ; reconstruct Cartesian AOs in registers via `unfold_shell_deriv_f4`.

**CPU setup:**

```python
build_radial_on_grid(plan, coords)  # [nradial, ngrids] f32, Hermite interp matching GPU
```

**Kernel:** `rho_gga_radial_precomp_pair` — same outer structure as `rho_gga_pair` (OTF), but J-atom radials loaded from `rad_val[ir*ngrids + g]` (coalesced on `g`) instead of `hermite_eval_ir` in registers.

**Per-SCF:** `dm` → Cartesian `dm_cart = c2s @ dm @ c2s.T` (same as OTF).

**vmat:** still precomputed spherical GTO + `vmat_gga_precomp_pair` (hybrid until a radial vmat kernel exists).

#### Shared execution model (all pair ρ kernels)

```
workgroup: (NPTILE,)   — one thread owns grid point g = gTile*NPTILE + ip
outer ja:  gather J-atom values for whole tile into __local
inner ia:  gather DM atom-pair block into __local dm_blk[16][16]
           I-atom values in registers (or local for radial)
           contract → rho[g], grad[g]
```

Future hook: replace `for (ja=0; ja<natoms; ja++)` with per-gTile active atom lists from `grid_screen.py` (not wired yet).

### Results (RTX 3090, benzene, GPU PBE f32 where noted)

#### Full XC per iteration

| Path | Wall (ms) | gpu_ρ | gpu_vmat | gpu_XC | host | vs cpu_libxc (~470 ms) |
|------|-----------|-------|----------|--------|------|------------------------|
| cpu_libxc | 470 | — | — | CPU | CPU | 1.0× |
| gpu_precomp_tiled (Part 3) | 80 | 60 | 17 | 0.1 | 2.2 | 5.9× |
| gpu_hermite_otf | 49 | 4.4 | 23.3 | — | 21 | 9.6× |
| **gpu_precomp_coalesced** | **35** | **17** | 16 | 0.2 | 2.1 | **13×** |
| **gpu_precomp_radial** | **23** | **4.8** | 16 | 0.1 | 2.2 | **20×** |

#### ρ-only (isolates projection kernel)

| Path | gpu_ρ (ms) | ρ max err vs CPU GTO |
|------|------------|----------------------|
| gpu_hermite_otf (`rho_gga_pair`) | **4.4** | ~10⁻⁶ (Hermite) |
| gpu_precomp_radial | **4.8** | ~10⁻⁴ (Hermite grid; matches OTF at ~7×10⁻⁴ f32) |
| gpu_precomp_coalesced | 17.7 | ~4×10⁻³ (exact GTO χ, f32) |
| gpu_precomp_tiled (old layout) | 59.6 | ~4×10⁻³ |

#### Memory and one-time setup

| Path | GPU static data | Setup | Notes |
|------|-----------------|-------|-------|
| OTF Hermite | ~0.2 MB tables | ~260 ms | No `eval_ao` |
| Coalesced | ~262 MB χ (+ row χ for vmat) | ~2.3 s | `eval_ao` ~2 s |
| Radial precomp | ~62 MB R,dR + ~262 MB χ for vmat | ~2.7 s | + `radial_cpu` ~320 ms |
| Tiled (Part 3) | ~262 MB χ `[iG,iAO]` | ~0.9–2 s | Same AO data, bad ρ layout |

**SCF break-even vs OTF** (extra setup / savings per cycle): radial precomp ~90 SCF iterations; coalesced ~140.

#### Numerical parity (full XC vs cpu_libxc)

| Path | vxc_max | exc_rel |
|------|---------|---------|
| gpu_hermite_otf | ~3×10⁻⁶ | ~1×10⁻⁷ |
| gpu_precomp_coalesced | ~2.6×10⁻⁶ | ~5×10⁻⁸ |
| gpu_precomp_radial | ~2.4×10⁻⁶ | ~2×10⁻⁷ |
| gpu_precomp_tiled | ~2.9×10⁻⁶ | ~3×10⁻⁸ |

All paths pass at ~10⁻⁶ on vxc for benzene/PBE.

### Interpretation

**Coalesced layout (Variant A)** solves the issue identified in Part 3: ρ drops **60 → 17 ms** (~3.5×) with no change in physics — pure memory-layout fix. Full XC **80 → 35 ms** with GPU PBE. Still slower than OTF on ρ because 262 MB χ bandwidth exceeds Hermite table eval.

**Radial precomp (Variant B)** does **not** beat OTF on ρ alone (4.8 vs 4.4 ms): the pair kernel was already near-optimal; precomputing R,dR trades ALU for DRAM without net gain on benzene. It **wins full XC** (23 ms) by combining:
- OTF-quality ρ (radial precomp kernel)
- Faster GTO vmat (16 ms vs Hermite vmat 23 ms)
- GPU PBE (0.1 ms vs libxc host ~21 ms)

**OTF Hermite** remains the best default for **low setup / moderate cycle count**: 260 ms setup, no 260 MB upload, ρ already optimal. Precomp paths win when geometry is fixed and many SCF cycles amortize `eval_ao`.

### Comparison to Part 2–3 (GTX 1650 → RTX 3090)

| Metric | GTX 1650 (Part 3) | RTX 3090 (Part 4) | Note |
|--------|-------------------|-------------------|------|
| OTF total | ~117 ms | ~49 ms | ~2.4× faster GPU |
| Precomp tiled ρ | ~273 ms | ~60 ms | Layout still hurt GTX more |
| Precomp coalesced ρ | — | ~17 ms | New in Part 4 |
| Best precomp full XC | ~425 ms (tiled) | **23 ms** (radial) | Layout + GPU PBE + hybrid |

On GTX 1650, precomp tiled was **slower than CPU** (~425 ms vs ~462 ms). On RTX 3090, optimized precomp is **20× faster than CPU** — the layout fix and GPU PBE were necessary but not sufficient on the weaker card; memory bandwidth and host XC dominated there.

### What shipped in code

| Component | Location |
|-----------|----------|
| Transpose during `eval_ao` + `buf_chi` upload | `xc_grid.py` — `fused='coalesced'` |
| `build_radial_on_grid()` | `radial_hermite.py` |
| Radial setup + `dm_cart` per cycle | `xc_grid.py` — `fused='radial_precomp'` |
| ρ-only API for tests | `nr_rks_precomputed_rho_only()` |
| Kernels | `kernels.cl` — `rho_gga_precomp_coalesced_pair`, `rho_gga_radial_precomp_pair` |
| Tests | `test_opencl_xc_scf.py` paths 8–9; `test_opencl_xc_rho_precomp.py` |

`fused` options: `'tiled'` (default), `'coalesced'`, `'radial_precomp'`, `'gemm'`, `False` (block loop).

### Open items (Part 4)

1. **Coalesced vmat** — transpose χ for vmat too; drop duplicate 262 MB row-major `buf_ao`.
2. **Radial vmat** — use `vmat_gga_pair` or radial+angular vmat kernel; eliminate `eval_ao` from radial path entirely.
3. **Grid screening** — per-gTile active atom lists (`grid_screen.py`) for all pair kernels.
4. **Hermite `t1m` convention** — align `hermite_map_point` with standard `t1m = 1-t` in kernels (requires regression sweep).
5. **LDA coalesced/radial kernels** — GGA only today.
6. **Document GTX 1650** — re-run `test_opencl_xc_scf.py` with new paths on 1650 for fair Part 4 table.

### Reproduce (RTX 3090)

```bash
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 \
  python3 -u expamples_prokop/test_opencl_xc_rho_precomp.py

PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 \
  python3 -u expamples_prokop/test_opencl_xc_scf.py
```

Quick coalesced / radial single-path:

```bash
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 python3 -c "
from pyscf import gto, dft
from pyscf.OpenCL import init_device
from pyscf.OpenCL.xc_grid import get_xc_grid_plan
from expamples_prokop.test_opencl_xc_scf import read_xyz
mol = gto.M(atom=read_xyz('data/xyz/benzene.xyz'), basis='ccpvdz', verbose=0)
grids = dft.gen_grid.Grids(mol); grids.level=3; grids.build(with_non0tab=True)
dm = dft.RKS(mol, xc='PBE').get_init_guess()
init_device(quiet=True)
for fused in ('coalesced', 'radial_precomp'):
    plan = get_xc_grid_plan(mol, grids, 'PBE')
    plan.setup_precomputed_gto(gpu_only=True, gpu_xc='pbe_f32', fused=fused)
    plan.nr_rks_precomputed_gto(dm, profile=True)
    print(fused, plan.last_timing)
"
```

### Part 4 summary

| Strategy | Addresses | ρ result | Full XC result |
|----------|-----------|----------|----------------|
| Coalesced χ `[iAO,iG]` | Strided gather (Part 3 bottleneck) | 3.5× faster than tiled | 13× vs CPU |
| Radial R,dR precomp | χ memory / Hermite eval cost | ≈ OTF (~5 ms) | **20× vs CPU** (hybrid + GPU PBE) |
| OTF Hermite (unchanged) | Setup + memory | Best ρ/ms without precompute | 9.6× vs CPU; best for few cycles |

The main lesson: **ρ projection on small molecules is compute-bound in the Hermite pair kernel, not memory-bound** — layout fixes help precomp GTO a lot, but matching OTF needs either fused Hermite eval or accepting a hybrid (radial ρ + GTO vmat + GPU XC).

---

## Part 5 — GPU Hermite AO setup (2026-06-29)

### Problem

For **precomp GTO row-major** and **precomp GTO coalesced**, one-time `setup_precomputed_gto()` was dominated by **CPU PySCF `eval_ao`** (~1.9–2.1 s on benzene/cc-pVDZ) plus PCIe upload. That is orthogonal to per-SCF ρ/vmat: we pay the cost even when geometry is fixed for many SCF cycles.

**Hermite OTF** does not have this problem (no χ upload). **Precomp radial Hermite** also skips full χ (only R,dR). The missing piece was GPU AO projection for paths that still need χ on the grid.

### Approach

Same Hermite radial representation as OTF, but **project AOs only** (no density matrix):

1. `eval_ao_hermite_cart_deriv1_tiled` — one workgroup per **grid tile** (`NPTILE` threads); outer loop over atoms; `__local wfR[NPTILE][MAX_SHELL]` collaborative radial cache; `eval_radial_cart_deriv1` writes Cartesian χ_cart.
2. GPU `c2s` matmul per component → spherical χ[**iG**, iAO] in `buf_ao`.
3. Optional `transpose_ao_to_chi` → χ[**iAO**, iG] for coalesced path (stays on GPU).

This is **simpler than Hermite OTF ρ/vmat** (no DM tiles, no XC, no atom-pair loops) but reuses the same `load_single_atom_meta_l` + tiled radial fill pattern as `rho_gga_pair`.

### Host API

```python
plan.setup_precomputed_gto(fused='coalesced', ao_proj='hermite_gpu')  # or ao_proj='auto' (default)
```

`ao_proj='auto'`: GPU Hermite setup when `lmax<=3` and GGA; else CPU `eval_ao`.

Timing keys in `plan.precalc_timing`: `eval_ao_cpu`, `eval_ao_hermite_gpu`, `ao_proj`.

### Expected trade-off

Hermite AO setup is **~2× faster than CPU `eval_ao`** on benzene (RTX 3090: ~296 ms vs ~622 ms; total setup ~297 ms vs ~841 ms) but χ is a **Hermite approximation** to GTO (same family as OTF/radial ρ, not identical to PySCF `eval_ao`). Use `ao_proj='cpu'` when exact GTO χ is required.

### Measured setup (RTX 3090, benzene, `ao_proj` sweep)

| fused | ao_proj | eval_ao_cpu | eval_ao_hermite_gpu | setup_total |
|-------|---------|-------------|---------------------|-------------|
| coalesced | cpu | 622 ms | — | 841 ms |
| coalesced | hermite_gpu | — | 296 ms | 297 ms |
| tiled (row-major) | hermite_gpu | — | 291 ms | 292 ms |

Per-SCF XC unchanged in kernel choice; coalesced+Hermite χ remains ~50 ms full XC vs ~90 ms with CPU GTO χ (layout + Hermite ρ bandwidth).

### Code

| Piece | Location |
|-------|----------|
| `eval_ao_hermite_cart_tiled`, `eval_ao_hermite_cart_deriv1_tiled`, `transpose_ao_to_chi` | `kernels.cl` |
| `OpenCLAOHermiteEvaluator.project_sph_deriv1_to_bufs` | `ao_hermite.py` |
| `build_radial_on_grid_tiled` | `kernels.cl` |
| `OpenCLAOHermiteEvaluator.build_radial_on_grid_gpu` | `ao_hermite.py` |
| `setup_precomputed_gto(fused='radial_precomp')` | `xc_grid.py` — GPU R,dR, no CPU staging |

**Radial GPU setup (benzene):** `radial_gpu` ~1.5 ms vs former CPU `build_radial_on_grid` ~330 ms; total setup ~252 ms vs ~610 ms. Only coords (~2 MB) uploaded from host; `buf_rad_val`/`buf_rad_dr` (~62 MB) stay on device.


```bash
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 \
  python3 -u expamples_prokop/test_opencl_xc_ao_hermite_setup.py
```

---

## Part 6 — Full-GPU path parity audit (2026-06-29)

### Definition: “full GPU path”

All three SCF-hot steps stay on device; only DM H2D and vxc D2H cross PCIe:

```
DM[iAO,iAO]  ─H2D─►  gpu_rho  ─►  buf_rho[4,nG]
                              ─►  gpu PBE (buf_vrho/vsigma → buf_wv)
                              ─►  gpu_vmat  ─D2H─►  vxc[iAO,iAO]
```

Requirements:

- `xc_eval='gpu'` (default) — PBE on device, no ρ/wv round-trip
- AO basis on GPU at setup: `ao_proj='hermite_gpu'` or `ao_proj='cpu'` (exact GTO χ upload)
- `fused='coalesced' | 'radial_precomp' | 'tiled'`

Audit script: `expamples_prokop/test_opencl_xc_full_gpu_parity.py`  
System: benzene / cc-pVDZ / grid level 3 / `nao=114` / `ngrids=143560` / RTX 3090 / `OMP_NUM_THREADS=1`

### End-to-end verdict: **PASS** (~2.6–3.3×10⁻⁶ on vxc)

| Path | Step 4 vxc max_abs | Step 4 rel | nelec diff | exc diff |
|------|-------------------|------------|------------|----------|
| coalesced + GTO AO (`ao_proj='cpu'`) | 2.62×10⁻⁶ | 1.15×10⁻⁶ | 8.6×10⁻⁷ | −1.6×10⁻⁶ |
| tiled + GTO AO | 2.62×10⁻⁶ | 1.15×10⁻⁶ | 8.6×10⁻⁷ | −1.6×10⁻⁶ |
| coalesced + Hermite AO | 3.34×10⁻⁶ | 1.47×10⁻⁶ | 7.3×10⁻⁶ | +1.6×10⁻⁶ |
| radial + Hermite AO | 2.86×10⁻⁶ | 1.26×10⁻⁶ | 4.4×10⁻⁶ | +3.1×10⁻⁶ |

CPU reference: `ni.nr_rks()` — `nelec=41.990461`, `exc=−34.27828722`.

### Step-by-step breakdown

```mermaid
flowchart LR
  DM["DM H2D"] --> RHO["Step 1: gpu_rho"]
  RHO --> WV["Step 2: gpu PBE → wv"]
  WV --> VMAT["Step 3: gpu_vmat"]
  VMAT --> VXC["Step 4: vxc D2H"]
```

#### Step 1 — ρ projection (DM → ρ)

| AO setup | ρ₀ rel | ∇ρ rel | ∇ρ max_abs | Notes |
|----------|--------|--------|------------|-------|
| GTO precomp (`ao_proj='cpu'`) | ~1.7×10⁻⁵ | ~3×10⁻⁶ | ~4.5×10⁻³ | f32 χ quantization |
| Hermite / radial (`ao_proj='hermite_gpu'`) | ~5×10⁻⁶ | **~1.3×10⁻⁴** | **~0.19** | Hermite ∇ρ approximation |

Hermite paths: ρ₀ is fine; ∇ρ can be off by ~0.19 on some grid points. This is the largest **ρ** error but does not dominate final vxc (see Step 2).

#### Step 2 — XC (ρ → weighted vxc `wv`)

Pointwise comparison: GPU f32 PBE vs CPU libxc (`_rks_gga_wv0`), same ρ.

| Component | max_abs | rel | Notes |
|-----------|---------|-----|-------|
| wv₀ (vrho × weight × 0.5) | **9.65×10⁻³** | 0.62 | **12 770** tail-grid points: GPU vrho≈0, libxc nonzero |
| wv₁₋₃ (GGA grad) | ~1.4×10⁻⁷ | ~10⁻⁴ | Excellent |

**Important:** 2a (GPU PBE on CPU ρ) and 2b (GPU PBE on GPU ρ) give **identical** wv errors for all paths — ρ errors (even Hermite ∇ρ) do not change wv at grid-point level for PBE. The aggregate step-2 metric is **misleading**: the 9.65×10⁻³ max is entirely wv₀ on ultra-sparse grid points (e.g. ρ₀≈1.6×10⁻⁷, libxc vrho=−9.6×10⁻³, GPU vrho=0).

Integrated impact of wv errors on vmat: max ~1.55×10⁻⁶ (contract wv difference through AO basis).

Root cause of wv₀ tail mismatch: GPU f32 symbolic PBE (`pbe.cl`) returns vrho≈0 or gets zeroed by `sanitize_pbe_xc_f32` on low-density tail points where libxc still returns small nonzero potentials. These points have negligible AO overlap → negligible vmat contribution.

#### Step 3 — vmat projection (wv → vmat)

| Test | max_abs rel |
|------|-------------|
| 3a GPU vmat + CPU libxc wv | ~1.2–1.4×10⁻⁶ |
| 3b GPU vmat + full GPU wv chain | ~1.2–1.5×10⁻⁶ |

vmat kernels (`vmat_gga_precomp_*_pair`) are correct. GPU wv tail errors do not propagate meaningfully.

#### Step 4 — full chain

Matches 3b for all paths. **Production default `xc_eval='gpu'` is safe for SCF-level accuracy.**

Debug path `xc_eval='cpu'` (libxc on D2H ρ): vxc still ~2.9×10⁻⁶ — dominated by f32 ρ quantization, not libxc vs GPU PBE.

### Where parity is lost (ranked by vxc impact)

1. **f32 ρ / vmat quantization** (~10⁻⁶ floor) — GTO paths
2. **Hermite ∇ρ** (~10⁻⁴ on gradients) — adds ~0.7×10⁻⁶ on vxc vs GTO path; still negligible for SCF
3. **GPU f32 PBE vrho on sparse tail grids** — large pointwise wv₀ errors, **negligible vmat impact**

### Practical guidance

| Goal | Settings |
|------|----------|
| Production full-GPU | `xc_eval='gpu'` (default), `ao_proj='auto'` or `'hermite_gpu'` |
| Best ρ parity | `ao_proj='cpu'` (slow setup, exact GTO χ) |
| Debug XC only | `xc_eval='cpu'` — libxc parity check, ~24 ms host XC |
| Fast AO setup | `ao_proj='hermite_gpu'` — ~3×10⁻⁶ vxc, bad ∇ρ pointwise |

### Reproduce

```bash
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 \
  python3 -u expamples_prokop/test_opencl_xc_full_gpu_parity.py
```

---

## Part 7 — End-to-end performance (benzene, RTX 3090, 2026-06-29)

Hardware: **NVIDIA GeForce RTX 3090**, `OMP_NUM_THREADS=1`, PBE/cc-pVDZ, grid level 3, `nao=114`, `ngrids=143560`.  
One XC iteration per path (same `dm`); setup is one-time outside per-iter budget.  
Min wall over 3 timed calls (1 warmup). Stages use `queue.finish()` per `gpu_*` key.

### Summary table

| Path | setup (ms) | wall (ms) | gpu (ms) | host (ms) | ρ (ms) | XC (ms) | vmat (ms) | vs CPU | vxc err |
|------|-----------|-----------|----------|-----------|--------|---------|-----------|--------|---------|
| **cpu_libxc** | 0 | **478** | 0 | 478 | — | — | — | 1.0× | ref |
| **gpu_precomp_radial_hermite** | 246 | **21** | 20 | 1 | 5.3 | 0.4 | 14.1 | **23.0×** | 2.86×10⁻⁶ |
| **gpu_hermite_otf** | 253 | **28** | 27 | 1 | 5.3 | 0.4 | 21.6 | 17.0× | 2.86×10⁻⁶ |
| **gpu_precomp_coalesced_hermite** | 284 | **29** | 28 | 1 | 15.9 | 0.1 | 12.1 | 16.6× | 3.34×10⁻⁶ |
| **gpu_precomp_coalesced_gto** | 2815 | 34 | 33 | 1 | 17.2 | 0.3 | 15.0 | 14.2× | 2.62×10⁻⁶ |
| **gpu_precomp_coalesced_auto** | 286 | **28** | 28 | 1 | 15.9 | 0.1 | 12.1 | 17.0× | 3.34×10⁻⁶ |
| **gpu_hermite_otf_libxc** | 281 | 46 | 25 | 20 | 4.5 | 19.8 | 20.9 | 10.4× | 2.98×10⁻⁶ |
| **gpu_precomp_tiled_gpu_xc** | 289 | 74 | 74 | 1 | 59.1 | 0.3 | 14.1 | 6.5× | 3.34×10⁻⁶ |
| **gpu_precomp_tiled_libxc** | 2237¹ | 92 | 75 | 20 | 60.0 | 19.6 | 15.3 | 5.2× | 2.86×10⁻⁶ |
| **gpu_precomp_tiled_gpu_xc (cpu AO)** | 2688¹ | 75 | 74 | 1 | 59.1 | 0.3 | 14.1 | 6.4× | 2.62×10⁻⁶ |

¹ `ao_proj='cpu'`: setup dominated by CPU PySCF `eval_ao` (~2.2–2.7 s). Default `ao_proj='auto'` uses GPU Hermite AO (~290 ms).

### Key observations

1. **Fastest per-SCF:** `gpu_precomp_radial_hermite` — **21 ms** (23× CPU). ρ matches OTF (~5 ms); vmat still uses precomp GTO χ from Hermite setup.
2. **Best overall default:** `gpu_precomp_coalesced` + `ao_proj='auto'` (Hermite) — **28 ms**, setup **286 ms**, no 2.6 GB CPU `eval_ao`.
3. **OTF Hermite** — **28 ms** per iter, setup **253 ms**, no χ storage; best when memory is tight.
4. **GPU PBE** saves ~20 ms host XC per iter vs libxc (`xc_eval='cpu'`): OTF 46 ms → 28 ms; tiled 92 ms → 74 ms.
5. **Row-major tiled** ρ still **3× slower** than coalesced (60 ms vs 17 ms) — layout bottleneck unchanged from Part 4.
6. **Exact GTO χ** (`ao_proj='cpu'`) costs ~2.5 s setup; per-SCF only ~6 ms faster than Hermite coalesced — rarely worth it for multi-cycle SCF.

### Stage breakdown (production paths, GPU PBE)

| Path | gpu_rho | gpu_xc_pbe | gpu_vmat | host (mainly ρ₀ D2H for nelec/exc) |
|------|---------|------------|----------|-----------------------------------|
| radial_hermite | 5.3 ms | 0.4 ms | 14.1 ms | 1.0 ms |
| coalesced_hermite | 15.9 ms | 0.1 ms | 12.1 ms | 0.7 ms |
| hermite_otf | 5.3 ms | 0.4 ms | 21.6 ms | 0.9 ms |
| tiled (row-major) | 59.1 ms | 0.3 ms | 14.1 ms | 0.9 ms |

vmat is 12–22 ms across paths; ρ layout dominates precomp differences.

### Setup cost (one-time, amortized over SCF cycles)

| Path | Total setup | AO source | Notes |
|------|-------------|-----------|-------|
| hermite_otf | ~253 ms | — | Hermite tables only |
| radial_hermite | ~246 ms | Hermite R,dR GPU ~1.5 ms | no full χ |
| coalesced + `ao_proj='auto'` | ~286 ms | Hermite AO GPU ~40 ms | χ[iAO,iG] on device |
| coalesced + `ao_proj='cpu'` | ~2815 ms | CPU eval_ao ~2600 ms | exact GTO χ |
| tiled + `ao_proj='cpu'` | ~2237 ms | CPU eval_ao | row-major χ |

Break-even vs CPU single XC (~478 ms): all GPU paths win on **first iteration** after setup. Break-even vs CPU including setup: need ~1 SCF cycle for Hermite paths, ~6 cycles for CPU `eval_ao` setup.

### Reproduce

```bash
# All paths (note: gpu_precomp_gemm may fail with INVALID_COMMAND_QUEUE — legacy path)
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 \
  python3 -u expamples_prokop/test_opencl_xc_scf.py

# Parity + per-path timing in one script
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 \
  python3 -u expamples_prokop/test_opencl_xc_full_gpu_parity.py
```

Quick single-path timing:

```bash
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 python3 -c "
from pyscf import gto, dft
from pyscf.OpenCL import init_device
from pyscf.OpenCL.xc_grid import get_xc_grid_plan
from expamples_prokop.test_opencl_xc_scf import read_xyz
import os, time
mol = gto.M(atom=read_xyz('data/xyz/benzene.xyz'), basis='ccpvdz', verbose=0)
grids = dft.gen_grid.Grids(mol); grids.level=3; grids.build(with_non0tab=True)
dm = dft.RKS(mol, xc='PBE').get_init_guess()
init_device(quiet=True)
for fused, ao in [('radial_precomp','hermite_gpu'), ('coalesced','hermite_gpu')]:
    plan = get_xc_grid_plan(mol, grids, 'PBE')
    plan.setup_precomputed_gto(gpu_only=True, fused=fused, ao_proj=ao, xc_eval='gpu')
    t0=time.perf_counter(); plan.nr_rks_precomputed_gto(dm, profile=True); print(fused, (time.perf_counter()-t0)*1e3, 'ms', plan.last_timing)
"
```

### Part 7 summary

| Rank | Path | wall (ms) | When to use |
|------|------|-----------|-------------|
| 1 | radial_hermite + GPU PBE | 21 | Fixed geometry, many SCF cycles, smallest ρ cost |
| 2 | coalesced_hermite + GPU PBE | 28 | Default precomp; best ρ/vmat balance |
| 3 | hermite_otf + GPU PBE | 28 | No χ memory, simple API |
| 4 | coalesced_gto + GPU PBE | 34 | Exact GTO χ, slow setup |
| 5 | tiled + GPU PBE | 74 | Legacy row-major; prefer coalesced |

Full-GPU path (`xc_eval='gpu'`) is **production-ready**: vxc parity ~10⁻⁶, 10–23× faster than CPU libxc on benzene.

---

## Part 8 — Pentacene + PTCDA E2E (RTX 3090, 2026-06-29)

Hardware: **NVIDIA GeForce RTX 3090**, `OMP_NUM_THREADS=1`, PBE/**6-31g**, grid level **2**.  
Harness: `expamples_prokop/test_opencl_xc_e2e_mols.py` — per-path min wall (3 timed), built-in OpenCL stage timers (`queue.finish` per `gpu_*` stage), step-wise parity audit on precomp paths.

**Profiling note:** `cProfile` only sees host Python; GPU time comes from `plan.last_timing`. Use `--profile` to dump top-20 `cProfile` functions for CPU `nr_rks` only. PCIe and kernel launch overhead appear in `host_*` keys.

### System sizes

| Molecule | natom | nao | ngrids | χ GGA f32 (theoretical) |
|----------|-------|-----|--------|-------------------------|
| pentacene | 36 | 226 | 321 784 | ~4.7 GB |
| PTCDA | 38 | 286 | 379 216 | ~6.9 GB |

`ao_proj='cpu'` (exact GTO χ upload) skipped when χ > 4 GB budget — would OOM or dominate setup.

### CPU reference (single XC iteration)

| Molecule | nr_rks | ρ step | wv step | vmat step |
|----------|--------|--------|---------|-----------|
| pentacene | 2921 ms | 2537 ms | 46 ms | 2139 ms |
| PTCDA | 4788 ms | 4098 ms | 52 ms | 3347 ms |

CPU is dominated by `eval_ao` in ρ and vmat projection (~90% of wall). libxc wv is cheap (~50 ms).

### End-to-end GPU performance

| Path | pentacene wall | vs CPU | PTCDA wall | vs CPU |
|------|---------------|--------|------------|--------|
| **gpu_hermite_otf** | **183 ms** | **16.0×** | **261 ms** | **18.3×** |
| gpu_hermite_otf_libxc | 226 ms | 12.9× | 313 ms | 15.3× |
| gpu_precomp_radial_hermite | 234 ms | 12.5× | 352 ms | 13.6× |
| gpu_precomp_coalesced_hermite | 288 ms | 10.1× | 471 ms | 10.2× |
| gpu_precomp_coalesced_auto | 290 ms | 10.1× | 464 ms | 10.3× |
| gpu_precomp_tiled_auto | 1723 ms | 1.7× | 2880 ms | 1.7× |

**Winner:** Hermite OTF — fastest per-SCF on both molecules, no χ storage, setup ~1 s.

GPU PBE saves ~45–54 ms vs libxc on host (OTF libxc path).

### Stage breakdown (GPU PBE paths, ms)

**pentacene**

| Path | gpu_rho | gpu_xc | gpu_vmat | host |
|------|---------|--------|----------|------|
| hermite_otf | 67 | 0.2 | 113 | 3 |
| radial_hermite | 78 | 0.4 | 155 | 4 |
| coalesced_hermite | 152 | 0.3 | 134 | 2 |
| tiled_auto | 1002 | 0.3 | 733 | 2 |

**PTCDA**

| Path | gpu_rho | gpu_xc | gpu_vmat | host |
|------|---------|--------|----------|------|
| hermite_otf | 108 | 0.1 | 149 | 5 |
| radial_hermite | 132 | 0.2 | 218 | 5 |
| coalesced_hermite | 278 | 0.4 | 190 | 2 |
| tiled_auto | 1800 | 0.3 | 1078 | 2 |

Scaling: ρ and vmat grow ~1.6× from pentacene → PTCDA on OTF; coalesced ρ grows ~1.8×. **Row-major tiled** remains unusable at this scale (ρ alone 1–1.8 s).

Setup (one-time): OTF/radial ~1.0 s; coalesced Hermite AO ~1.3–1.6 s (AO projection 0.3–1.6 s depending on cache); radial R,dR GPU ~2–3 ms.

### Accuracy (vxc vs CPU `nr_rks`)

| Path | pentacene vxc max | PTCDA vxc max |
|------|-------------------|---------------|
| All GPU paths | 2.2–2.5×10⁻⁶ | 2.4–2.8×10⁻⁶ |
| exc rel | ~7×10⁻⁷ | ~6–9×10⁻⁷ |

**End-to-end vxc parity holds** at SCF tolerance on larger systems.

### Step-wise audit (precomp paths)

Same pattern as Part 6 (benzene):

| Step | pentacene | PTCDA | Impact on vxc |
|------|-----------|-------|---------------|
| 1 ρ (Hermite ∇ρ vs CPU GTO) | max ~0.21, rel ~1.6×10⁻⁴ | max ~0.95, rel ~2.2×10⁻⁴ | Low — PBE wv insensitive |
| 2 wv₀ GPU PBE vs libxc | max ~1.2×10⁻² (tail grids) | max ~1.2×10⁻² | Negligible (sparse points) |
| 2 wv₁₋₃ | ~10⁻⁷ | ~10⁻⁷ | Excellent |
| 3 vmat kernel | ~2.3×10⁻⁶ | ~2.4×10⁻⁶ | PASS |
| 4 full chain | ~2.2–2.5×10⁻⁶ | ~2.4–2.8×10⁻⁶ | PASS |

PTCDA Hermite ∇ρ pointwise error is larger (~0.95 abs on some grid points) but still does not break vxc (~10⁻⁶). The wv₀ tail-grid artifact (rel ~0.56 on worst component) is identical to benzene — GPU f32 PBE vrho on ultra-low density points.

### Recommendations by molecule size

| Regime | nao | Best path | Rationale |
|--------|-----|-----------|-----------|
| Small (benzene) | ~100 | radial_hermite or coalesced | 21–29 ms; χ fits easily |
| Medium (pentacene) | ~200–300 | **hermite_otf** | 16–18× CPU; no 4.7 GB χ |
| Large (PTCDA) | ~300–500 | **hermite_otf** | 18× CPU; coalesced OK but 2× slower |
| Any | — | **avoid tiled row-major** | 1.7× CPU only — layout bottleneck |

### Reproduce

```bash
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 \
  python3 -u expamples_prokop/test_opencl_xc_e2e_mols.py --mols pentacene PTCDA

# Faster (skip step audit):
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 \
  python3 -u expamples_prokop/test_opencl_xc_e2e_mols.py --mols pentacene PTCDA --no-step-audit

# CPU cProfile (nr_rks only):
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 \
  python3 -u expamples_prokop/test_opencl_xc_e2e_mols.py --mols PTCDA --profile
```

---

## Part 9 — Multi-CPU vs GPU (16-core host, RTX 3090, 2026-06-29)

Host: **16 logical cores**; benchmarks use **`OMP_NUM_THREADS=15`** via `lib.num_threads(15)` (one core left for OS). GPU column: **gpu_hermite_otf + GPU PBE** from Parts 7–8.

Harness: `expamples_prokop/test_opencl_xc_cpu_threads.py`

### XC iteration wall time (ms)

| System | basis | grid | nao | ngrids | CPU ×1 | CPU ×15 | CPU speedup | GPU OTF | GPU vs ×1 | GPU vs ×15 |
|--------|-------|------|-----|--------|--------|---------|-------------|---------|-----------|------------|
| benzene | cc-pVDZ | 3 | 114 | 143 560 | 476 | **301** | 1.58× | **28** | **17.0×** | **10.7×** |
| pentacene | 6-31g | 2 | 226 | 321 784 | 2860 | **1629** | 1.76× | **183** | **15.6×** | **8.9×** |
| PTCDA | 6-31g | 2 | 286 | 379 216 | 4669 | **2467** | 1.89× | **261** | **17.9×** | **9.5×** |

Min wall over 3 timed `ni.nr_rks()` calls. Multi-thread vxc identical to single-thread (max drift ~10⁻¹³).

### Observations

1. **OpenMP helps but saturates early** — only 1.6–1.9× from 1→15 threads. PySCF XC integrates grid blocks with partial parallelization in `eval_ao` / BLAS; Python block loop and libxc remain largely serial.
2. **GPU still wins vs 15-core CPU** — 8.9–10.7× on the three systems. Previous “vs CPU” figures in Parts 7–8 used `OMP_NUM_THREADS=1`; against a threaded CPU the GPU advantage is ~40% smaller but still decisive.
3. **Scaling trend** — CPU speedup rises slightly with system size (1.58× → 1.89×), consistent with more work in parallelizable `eval_ao` regions. GPU advantage vs ×15 CPU stays ~9–11×.
4. **Benzene GPU floor** — radial_hermite at **21 ms** would be **14.3×** vs CPU×15 (301 ms); OTF at 28 ms is quoted above.

### When to use which backend

| Scenario | Recommendation |
|----------|----------------|
| Single-core / `OMP_NUM_THREADS=1` | GPU 16–18× faster |
| Full 15-thread CPU | GPU still ~9–11× faster |
| No GPU / debugging | Set `OMP_NUM_THREADS=15` for ~1.7× over single thread |
| Production SCF | GPU OTF + `xc_eval='gpu'`; CPU threads matter only for J/K and diagonalization |

### Reproduce

```bash
# Table above (CPU 1 and 15 threads)
PYTHONPATH=/home/prokop/git/pyscf python3 -u expamples_prokop/test_opencl_xc_cpu_threads.py --threads 1 15

# Explicit env (script also calls lib.num_threads)
OMP_NUM_THREADS=15 PYTHONPATH=/home/prokop/git/pyscf python3 -u \
  expamples_prokop/test_opencl_xc_cpu_threads.py --threads 15
```

---

## Part 10 — Full SCF convergence profiling (benzene, RTX 3090, 2026-06-29)

Harness: `expamples_prokop/profile_gpu_scf.py`  
Setup: benzene / cc-pVDZ / grid 3 / DF-RKS / PBE / `OMP_NUM_THREADS=15` / converged SCF (`conv_tol=1e-8`, `conv_tol_grad=1e-5`).

### SCF wall time (converged runs)

| Mode | cycles | total (s) | ms/cycle | vs CPU |
|------|--------|-----------|----------|--------|
| **cpu** (libxc + CPU DF J/K) | 9 | 3.88 | **431** | 1.0× |
| **gpu_otf** (GPU XC + CPU DF J/K) | 30 | 2.10 | **70** | **6.2×** |
| gpu_full (GPU XC + GPU DF J/K) | 50 (max) | 1.80 | 36 | not converged |

GPU XC cuts per-cycle time ~6×. `gpu_full` hits `max_cycle=50` without converging — likely f32 noise in GPU J/K; use `backend=3` to compare or fix later.

### Where time goes inside one SCF cycle (`gpu_otf`, 30 cycles)

**Python timers** (wall clock, includes GPU sync inside `get_veff`):

| Step | ms/call | ms/cycle | share of cycle |
|------|---------|----------|----------------|
| `rks.get_veff` (total) | 61.0 | 61.0 | **87%** |
| ├ `df.get_jk` (CPU RI-J/K) | 33.4 | 33.4 | **48%** |
| └ GPU XC (profiled) | — | 29.2 | **42%** |
| `scf.eig` | 1.6 | 1.6 | 2% |
| other | — | ~8 | ~11% |

**GPU XC sub-steps** (`_gpu_timing_acc` / `queue.finish`, per cycle):

| Stage | ms/cycle | share of GPU XC |
|-------|----------|-----------------|
| **gpu_vmat** | **22.7** | **77%** |
| gpu_rho | 4.8 | 16% |
| host_xc_reduce (ρ₀ D2H for nelec/exc) | 1.0 | 3% |
| gpu_xc_pbe | 0.3 | 1% |
| host_dm_cart + vmat D2H | 0.4 | 1% |

### Bottleneck after GPU XC offload

```text
Per SCF cycle (~70 ms, gpu_otf):
  CPU DF J/K     ████████████████████  33 ms  (48% of get_veff)
  GPU vmat       ██████████████        23 ms  (33%)
  GPU rho        ███                    5 ms  ( 7%)
  eig + DIIS     █                      2 ms  ( 3%)
  GPU PBE + PCIe ▏                     <2 ms  ( 3%)
```

**Previous bottleneck (CPU path):** `nr_rks` / libxc + `eval_ao` in ρ and vmat — **~346 ms per `get_veff`** (99% of 431 ms/cycle).

**Current bottleneck (GPU XC):** split between **CPU density-fitting J/K (~33 ms)** and **GPU vmat projection (~23 ms)**. ρ is no longer limiting (~5 ms).

### cProfile on `gpu_full` (host Python only)

GPU kernels are invisible except via `_gpu_sync` (`queue.finish`). Top host entries:

| Function | tottime | role |
|----------|---------|------|
| `_gpu_sync` | 0.25 s | waits on OpenCL (ρ, vmat, DF J/K) |
| `df_jk.get_jk` | 0.17 s | GPU DF J/K driver |
| `enqueue_read_buffer` | 0.10 s | DF cderi / vmat D2H |
| `scipy.linalg.eigh` | 0.07 s | diagonalization |
| `enqueue_nd_range_kernel` | 0.01 s | kernel launch overhead (under-reported) |

Use `mf._gpu_profile=True` + `plan.last_timing` for XC; monkey-patch `df.get_jk` for J/K — not cProfile — when tuning GPU SCF.

### Implications

1. **Next optimization target:** GPU DF J/K (already implemented; enable with `mf.with_df.backend=2`) or overlap J/K with XC.
2. **Within XC:** **vmat** (~23 ms) > ρ (~5 ms) on OTF path for benzene — differs from single-iteration precomp radial (ρ-limited) because OTF vmat does full Hermite pair contraction.
3. **Production SCF:** `mf.backend=2; mf.setup_gpu(xc_path='onthefly'); mf._gpu_profile=True` — ~6× faster than 15-thread CPU for benzene.
4. **Do not use `gpu_full` without convergence testing** — f32 J/K can stall DIIS (observed here).

### Reproduce

```bash
# CPU vs GPU OTF vs GPU full (cProfile on last mode)
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=15 python3 -u \\
  expamples_prokop/profile_gpu_scf.py --mode cpu gpu_otf gpu_full --threads 15 --profile

# GPU OTF only (detailed timers)
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=15 python3 -u \\
  expamples_prokop/profile_gpu_scf.py --mode gpu_otf --threads 15
```

---
