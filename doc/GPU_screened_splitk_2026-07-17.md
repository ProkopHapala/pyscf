---
type: ExperienceReport
title: Screened split-K vmat — RTX 3090 optimization session
description: Split-K screened radial vmat kernel, local-memory zeroing removal, upper-triangle revert; benzene + PTCDA benchmarks on RTX 3090
tags: [opencl, dft, xc, gpu, rtx3090, screening, split-k, vmat, optimization, lessons]
timestamp: 2026-07-17
last_updated: 2026-07-17
---

Session on the **RTX 3090** host (not the GTX 1650 laptop). Raw 3090 benchmarks: [`doc/GPU_benchmark.md`](GPU_benchmark.md). 1650 lessons: [`doc/GPU_1650_lessons_2026-07-17.md`](GPU_1650_lessons_2026-07-17.md). Optimization arc: [`doc/GPU_optimixation_experience.md`](GPU_optimixation_experience.md). Cookbook: [`doc/opencl_gpu_paths_cookbook.md`](opencl_gpu_paths_cookbook.md).

**Question of the day:** the screened radial kernels (developed for GTX 1650) already beat OTF on PTCDA — can split-K + dead-code removal push the 3090 further?

---

## Executive summary

| Finding | Consequence |
|---------|-------------|
| Upper-triangle skip **breaks vmat parity** | Reverted — vmat is one-sided GGA, not symmetric in (ia,ja) unlike rho with symmetric DM |
| **Split-K** on screened vmat gTile list | +20–30% vmat speedup; better occupancy on 82-CU 3090 |
| **Local-memory zeroing removal** | +30% vmat speedup; fill writes exactly nia/nja, guards already protect |
| Combined: new `production_radial_screened_splitk` profile | **Benzene 10.0 ms** (was 13.6 ms OTF splitK), **PTCDA 73.4 ms** (was 306.8 ms OTF splitK) |

---

## Hardware context

Same machine as prior 3090 benchmarks. See [`doc/GPU_benchmark.md`](GPU_benchmark.md) § Test machine.

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 3090 |
| Compute units | 82 |
| Global memory | 24 GB |
| Local memory / CU | 48 KiB |
| OpenCL | 3.0 (CUDA 12.4.131, driver 550.120) |
| CPU (host) | AMD Ryzen 7 5800X, 32 GiB RAM |

---

## What was tried

### 1. Upper-triangle optimization (REVERTED — parity break)

**Idea:** The `rho_gga_radial_screened` kernel already skips `ia > ja` with `scale = (ia == ja) ? 1 : 2`. The host symmetrizes with `vmat + vmat.T`. Apply the same to `vmat_gga_radial_screened_pair`.

**Implementation:**
- Early-exit: `if (ia > ja) return;`
- Off-diagonal scale: `off_diag_scale = (ia < ja) ? 2.0f : 1.0f;` on `acc[t]` before writing to `vmat`

**Result:** vmat sped up 104→64 ms, but **parity broke**: `|vxc|` went from 1.64e-03 to 1.21e-02.

**Root cause:** Unlike rho (where DM symmetry `D[i,j] = D[j,i]` allows ia≤ja with scale=2), vmat computes a **one-sided** GGA matrix:

```
V[i,j] = sum_g wv_i(g) * phi_j(g)
```

where `wv_i(g)` depends on atom i's AO derivatives and `phi_j(g)` on atom j's AOs. The (ja,ia) WG computes `V[j,i] = sum_g wv_j(g) * phi_i(g)` — a **genuinely different** block. The host `vmat + vmat.T` combines both. Skipping (ja,ia) loses the `V[j,i]` contribution entirely.

**Lesson:** Upper-triangle skip is valid only when the kernel computes a **symmetric** quantity (e.g. rho with symmetric DM). For one-sided vmat, both (ia,ja) and (ja,ia) are needed.

**Action:** Reverted both the early-exit and the scaling factor.

### 2. Split-K screened vmat (KEPT — main win)

**Problem:** PTCDA has 38² = 1444 WGs launched, but screening gives ~5.8× fewer active pairs → ~249 effective pairs. With 82 CUs, that's only ~3 WGs per CU — **under-parallelized**. Each WG processes ALL gTiles for its pair serially.

**Solution:** Split the pair's CSR gTile list across `nsplit` WGs, accumulate into partial buffers, then reduce with the existing `reduce_split_vmat` kernel.

**New kernel:** `vmat_gga_radial_screened_pair_splitk` in `pyscf/OpenCL/kernels.cl`

Key differences from `vmat_gga_radial_screened_pair`:
- 3D global: `(natoms, natoms * WGS_VMAT, nsplit)` — `isplit = get_group_id(2)`
- CSR gTile list sharded: `gt_per_split = (n_gt + nsplit - 1) / nsplit; p0 = gt0 + isplit * gt_per_split`
- Writes to `partial_vmat[isplit * ncart * ncart + ...]` instead of `vmat`
- Reuses existing `reduce_split_vmat` kernel (zero additional reduce code)

**Host wiring** (`pyscf/OpenCL/xc_grid.py`):
- `vmat_grid_splits > 1` now allowed for `vmat_mode='radial_screened'` (was only `'radial_precomp'`)
- Partial vmat buffer allocated for both `radial_precomp` and `radial_screened` when splits > 1
- Kernel arg setup branches on `vmat_grid_splits > 1` to select splitk kernel + reduce kernel
- Execution path already handled `k_vmat_reduce` generically — no hot-loop changes needed

**Sweep results (PTCDA, post-zeroing-removal):**

| splits | gpuCL (ms) | rho_cl (ms) | vmat_cl (ms) |
|--------|-----------:|------------:|-------------:|
| 1      | 87.5       | 22.6        | 64.3         |
| 2      | 82.6       | 22.9        | 58.8         |
| **4**  | **77.4**   | 22.5        | **53.9**     |
| 8      | 80.2       | 23.1        | 56.3         |
| 16     | 79.5       | 23.0        | 55.5         |
| 32     | 77.5       | 22.1        | 54.8         |

**Splits=4 is optimal** for PTCDA. Higher splits add reduce overhead without proportional vmat gain.

### 3. Local-memory zeroing removal (KEPT — big win)

**Problem:** Both screened vmat kernels zeroed `aowI[ip][a]` and `aoJ[ip][a]` for all `MAX_AO_ATOM=16` entries before the fill call:

```c
for (int a = 0; a < MAX_AO_ATOM; a++) {
    aowI[ip][a] = 0.0f;
    aoJ[ip][a] = 0.0f;
}
```

This is **2 × 16 = 32 float writes per thread per tile** — pure waste because:

1. `fill_atom_aow_gga_radial_precomp` writes exactly `nia` entries (the atom's AO count)
2. `fill_atom_ao_radial_precomp` writes exactly `nja` entries
3. The accumulation loop guards: `if (a >= nia || b >= nja) continue;` and `if (g >= ngrids) continue;`

So any uninitialized entries are never read. The zeroing was defensive but unnecessary.

**Action:** Removed the zeroing loop from both `vmat_gga_radial_screened_pair` and `vmat_gga_radial_screened_pair_splitk`.

**Impact (PTCDA, splits=8):**

| Version | vmat_cl (ms) | gpuCL (ms) |
|---------|-------------:|-----------:|
| With zeroing | 74.4 | 105.9 |
| Without zeroing | 50.6 | 82.9 |
| **Speedup** | **1.47×** | **1.28×** |

The zeroing was ~40% of vmat kernel time — a significant fraction of local memory bandwidth wasted on writes that are never read.

---

## Final benchmarks

### Benzene (cc-pVDZ, grid 3, nao=114, ngrids=143560, OMP=1)

| Method | gpuCL (ms) | rho_cl (ms) | vmat_cl (ms) | \|vxc\| |
|--------|----------:|-----------:|-------------:|---------|
| CPU libxc | — | — | — | 0 (ref) |
| OTF cubic | 25.3 | 4.2 | 20.7 | 3.15e-05 |
| OTF quintic | 26.0 | 4.2 | 21.4 | 3.19e-05 |
| Radial precomp | 22.3 | 4.8 | 17.2 | 3.15e-05 |
| OTF ρ + rad vmat | 22.8 | 4.6 | 17.5 | 3.15e-05 |
| OTF ρ + rad vmat splitK | 14.2 | 6.1 | 7.7 | 3.16e-05 |
| Radial screened | 15.6 | 3.1 | 12.1 | 3.18e-05 |
| **Radial screened splitK** | **10.0** | 3.1 | **6.3** | 3.18e-05 |

**Benzene: 10.0 ms** — new fastest path, 30% faster than OTF splitK (14.2 ms), 44× faster than CPU (446 ms).

### PTCDA (6-31g, grid 2, nao=286, ngrids=379216, OMP=1)

| Method | gpuCL (ms) | rho_cl (ms) | vmat_cl (ms) | \|vxc\| |
|--------|----------:|-----------:|-------------:|---------|
| OTF cubic | 314.1 | 119.3 | 194.2 | 1.64e-03 |
| OTF quintic | 292.3 | 115.5 | 176.2 | 1.64e-03 |
| Radial precomp | 387.9 | 143.2 | 244.4 | 1.64e-03 |
| OTF ρ + rad vmat | 361.9 | 115.5 | 245.9 | 1.64e-03 |
| OTF ρ + rad vmat splitK | 306.8 | 118.0 | 188.2 | 1.64e-03 |
| Radial screened | 94.1 | 26.7 | 66.9 | 1.64e-03 |
| **Radial screened splitK** | **73.4** | 27.7 | **45.1** | 1.64e-03 |

**PTCDA: 73.4 ms** — 4.3× faster than OTF splitK (306.8 ms), 22% faster than screened without splitK (94.1 ms).

### Speedup summary

| Metric | Benzene | PTCDA |
|--------|--------:|------:|
| vs CPU libxc | 44× | — |
| vs OTF cubic | 2.5× | 4.3× |
| vs OTF splitK (prev best) | 1.4× | 4.2× |
| vs Radial screened (no splitK) | 1.6× | 1.3× |

---

## Files touched

| File | Change |
|------|--------|
| `pyscf/OpenCL/kernels.cl` | New `vmat_gga_radial_screened_pair_splitk` kernel; removed local-mem zeroing from both screened vmat kernels; reverted upper-triangle skip |
| `pyscf/OpenCL/xc_grid.py` | `vmat_grid_splits>1` support for `radial_screened` mode; partial vmat buffer allocation; splitk kernel arg wiring |
| `pyscf/OpenCL/gpu_profiles.py` | New `production_radial_screened_splitk` profile (splits=4) |
| `expamples_prokop/profile_xc_stages_benzene.py` | Added `Radial screened splitK` stage |

---

## Strategies that worked

### 1. Split the serial dimension, not the parallel one

The screened vmat kernel has two dimensions: atom pairs (parallel, ~249 active) and gTiles per pair (serial, ~100+). With 82 CUs, ~249 pairs gives only ~3 WGs/CU. Splitting the serial gTile dimension across `nsplit` WGs multiplies the WG count by `nsplit`, filling the GPU. This is the same pattern that worked for non-screened radial vmat split-K.

### 2. Remove dead writes, not dead reads

The zeroing of `aowI`/`aoJ` was a defensive pattern — "zero everything, then fill what's needed." But the fill writes exactly the needed entries, and the accumulation guards prevent reading uninitialized data. The zeroing was pure waste: 32 float writes per thread per tile to local memory, competing with the actual AO data for bandwidth. Removing it gave a **1.47× vmat speedup** — more than any kernel-level micro-optimization could.

### 3. Reuse existing infrastructure

The `reduce_split_vmat` kernel and the `k_vmat_reduce` execution path already existed for `radial_precomp` split-K. The screened split-K required **zero new host code** in the hot loop — just different kernel arg setup in `setup_onthefly`. This is the AHA principle: avoid hasty abstractions, reuse what works.

### 4. Verify parity after every change

The upper-triangle optimization looked correct by analogy with `rho_gga_radial_screened`, but broke parity by 7×. Catching this immediately (via the `|vxc|` column in the profiling output) prevented a subtle bug from propagating. Always check parity after every kernel change, not just at the end.

---

## Strategies that didn't work

### Upper-triangle skip for one-sided vmat

The analogy between rho and vmat symmetry is tempting but wrong. Rho contracts a **symmetric** DM with AO pairs: `ρ = sum_{i,j} D[i,j] * χ_i * χ_j`, so `D[i,j] = D[j,i]` allows skipping (j,i) with scale=2. Vmat computes `V[i,j] = sum_g wv_i(g) * φ_j(g)` where `wv_i` and `φ_j` are different quantities — the matrix is **not symmetric** until the host does `V + V.T`. Both halves must be computed.

---

## Open issues / future work

- **rho is now the bottleneck for PTCDA**: 27.7 ms rho vs 45.1 ms vmat — rho was 31 ms before screening helped it to 27 ms, but vmat dropped from 104 to 45 ms. Further rho optimization (e.g. split-K for `rho_gga_radial_screened`) could help.
- **Tile sweep for screened splitK**: `NPTILE`, `WGS_VMAT`, and `vmat_grid_splits` were not jointly swept. A 1-neighborhood coordinate descent (like `sweep_splitk_tiles.py`) may find better configs.
- **GTX 1650 validation**: The zeroing removal should help the 1650 too (same local memory pressure), but split-K may not help as much (14 CUs → already over-parallelized with 249 pairs).
- **Upper-triangle for rho**: `rho_gga_radial_screened` already does upper-triangle skip correctly. No change needed there.
- **Screening tightness**: tighter `screen_eps` could reduce the number of active pairs further, amplifying the split-K benefit.
