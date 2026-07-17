# OpenCL XC report — rho iTile loop, host setup, timing (2026-03)

**Hardware:** GTX 1650, `OMP_NUM_THREADS=1`  
**Reference system:** benzene, ccpvdz, grid level 3 (`ngrids=143560`, `natoms=12`)  
**Test:** `expamples_prokop/test_opencl_xc_onthefly.py`

---

## Problem

After the vmat abTile fix ([`vmat_optimization_report.md`](../vmat_optimization_report.md)), rho dominated total time but had separate issues:

1. **`iTile` on `group_id(1)`** — each workgroup wrote partial `rho[iTile, g]`; reduction over i-atoms across tiles was done on the **CPU** (`reshape().sum(axis=0)`).
2. **Per-SCF allocation** — `cl.Buffer` / kernel setup inside the hot path.
3. **No timing split** — wall time mixed Python harness with GPU kernel execution.
4. **Wrong default GPU path** — `nr_rks_gpu` used materialized Hermite AO + GEMM instead of on-the-fly.

Architectural note: **rho and vmat should not share the same launch geometry** (see cookbook §1). vmat correctly keeps `(iTile,jTile)` on the grid and loops `gTile` inside; rho must own grid points and loop atom tiles inside.

---

## Changes

### `rho_*_tiled` (`kernels.cl`)

- Launch: `global = (ceil(ngrids/NPTILE), NATILE)`, `local = (NPTILE, NATILE)` — one workgroup per **grid tile**.
- **Inner loops:** `for (iTile) for (jTile)` — all atom-pair tiles inside the WG.
- **Final output:** `__local psum` reduce over `NATILE` → `rho[g]` (GGA: `rho[c*ngrids+g]`).
- No partial buffer, no host sum, no atomics, no second reduce kernel.

### Host (`xc_grid.py`, `rks.py`)

- `setup_onthefly()` / `setup_xc_grid_gpu()` / `mf.setup_gpu()` — compile, Hermite tables, buffers, baked kernel args **before SCF**.
- `nr_rks_hermite_onthefly(dm, profile=True)` → `plan.last_timing` with `kernel_rho`, `kernel_vmat`, `harness_*`.
- `buf_rho` size `4 × ngrids` (final density on device).

---

## Correctness

| Quantity | max rel err vs CPU |
|----------|-------------------|
| nelec | 1.8e-7 |
| exc | 2.2e-7 |
| vxc | 2.7e-6 |

---

## Performance (benzene, warm)

| Phase | time |
|-------|------|
| `setup_xc_grid_gpu` (once) | ~0.32 s |
| CPU `nr_rks` | ~0.44 s |
| **GPU total / cycle** | **~0.18 s** (~2.4× vs CPU) |
| `kernel_rho` | ~79 ms |
| `kernel_vmat` | ~87 ms |
| `kernel_total` | ~166 ms |
| `harness_total` | ~17 ms |

Compared to vmat-report era total OTF (~0.58 s benzene), end-to-end improved largely from host path + rho fix + on-the-fly default (exact attribution not re-benchmarked separately on same machine session).

---

## Still open (not in this report)

- rho: `wfRj` cooperative load reloads all `NPTILE` rows per `(iTile,jTile)`; each thread uses one row — hoist opportunity.
- Overlap libxc with GPU via events (reduce `harness_rho_xc`).
- vmat `NPTILE=32` sweep (occupancy vs inner-loop count).

---

## Files touched

- `pyscf/OpenCL/kernels.cl` — `rho_lda_tiled`, `rho_gga_tiled`
- `pyscf/OpenCL/xc_grid.py` — setup, launch dims, timing
- `pyscf/dft/rks.py` — `setup_gpu()`
- `expamples_prokop/test_opencl_xc_onthefly.py` — benzene benchmark
- `doc/opencl-kernel-cookbook.md` — guidelines (new)
