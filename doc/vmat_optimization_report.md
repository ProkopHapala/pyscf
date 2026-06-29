# OpenCL XC Kernel Optimization Report

This document has two parts: the original **vmat abTile fix** (baseline), and a **2026-03 follow-up** covering rho/vmat tiling, compile-time tile config, pair kernels, and benchmark sweeps on GTX 1650.

Test harness: `expamples_prokop/test_opencl_xc_onthefly.py --xyz <file>`  
Tile sweeps: `expamples_prokop/sweep_opencl_tiles.py`  
Config: `pyscf/OpenCL/tile_config.py` (`OPENCL_NPTILE`, `OPENCL_NATILE`, `OPENCL_WGS_VMAT`, `OPENCL_MAX_ITILE`)

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

