# vmat Kernel Optimization Report

## Problem

The original `vmat_lda_tiled` and `vmat_gga_tiled` kernels used a "hybrid atom-tile + abTile" design with a 3D workgroup `(iTile, jTile, abTile)`. This caused catastrophic performance:

| System | vmat time | Total OTF time |
|--------|-----------|----------------|
| water (3 atoms, 114 ncart) | 5.17s | 5.94s |
| benzene (12 atoms, 114 ncart) | 5.17s | 5.92s |
| pentacene (36 atoms, 226 ncart) | ~40s | 40.6s |

## Root Cause

The abTile design had three major inefficiencies:

1. **57x redundant radial evaluation**: `VBLK_SIZE=3600` with `WGS_VMAT=64` gives `NABTILE=57` workgroups per atom-pair tile. Each workgroup independently re-evaluates all radial functions for the same grid tile — 57x redundant work.

2. **6x wasted angular unfolding**: Each thread owns one AO-pair element `(a,b)` but calls `unfold_shell()` which computes all 6 angular components. Only 1 is used — 6x wasted compute.

3. **Per-thread shell search**: Each thread repeatedly searched for which shell an AO index belongs to, adding a serial loop inside the grid tile loop.

For benzene: `3*3*57 = 513` workgroups, each looping over `143560/16 = 8973` grid tiles = massive serial overhead with only 64 threads per workgroup.

## Solution

Replaced the abTile design with a local AO cache + private accumulator design (from `doc/ToOpenCL.chat.md` lines 3606-4057):

```
workgroup = one (iTile, jTile) atom-pair tile
thread    = owns QPT AO-pair matrix elements with private acc[QPT]
loop      = over grid-point tiles
local     = unfolded AO values aoI[NPTILE][AO_TILE], aoJ[NPTILE][AO_TILE]
no abTile, no redundant radial recomputation, no atomics
```

### Key design points

- **Local AO cache**: `aoI[NPTILE][AO_TILE]` and `aoJ[NPTILE][AO_TILE]` store fully unfolded AO values. `AO_TILE = NATILE * MAX_AO_ATOM = 4*15 = 60`. Total local memory: `2 * 16 * 60 * 4 = 7.5 KB`.

- **Fill functions**: `fill_atom_ao_lda()` and `fill_atom_aow_gga()` unfold all AO components once per `(grid point, atom)` and store them in local memory. No per-thread redundant unfolding.

- **Private accumulators**: `acc[QPT]` where `QPT = ceil(VBLK_SIZE / WGS_VMAT) = ceil(3600/256) = 15`. Each thread accumulates 15 AO-pair elements over all grid tiles.

- **Direct AO index mapping**: `iao_l = il*MAX_AO_ATOM + a`, `jao_l = jl*MAX_AO_ATOM + b`. No shell search needed.

- **Launch**: 2D global `(n_iTiles, n_jTiles * WGS_V)`, local `(1, WGS_V)` with `WGS_V=256`. For benzene: only `3*3 = 9` workgroups instead of 513.

### Constants

```c
#define NPTILE       16
#define NATILE       4
#define MAX_SHELL    6
#define MAX_AO_ATOM  15
#define AO_TILE      (NATILE * MAX_AO_ATOM)      // 60
#define VBLK_SIZE    (AO_TILE * AO_TILE)          // 3600
#define WGS_VMAT     256
#define QPT          ((VBLK_SIZE+WGS_VMAT-1)/WGS_VMAT)  // 15
#define PT_ATOM_SIZE (NPTILE * NATILE)            // 64
```

## Files Modified

- `pyscf/OpenCL/kernels.cl`: Replaced both `vmat_lda_tiled` and `vmat_gga_tiled` kernels. Added `fill_atom_ao_lda()` and `fill_atom_aow_gga()` helper functions.
- `pyscf/OpenCL/xc_grid.py`: Updated launch to 2D global `(n_iTiles, n_jTiles * WGS_V)` with local `(1, WGS_V)`, `WGS_V=256`.

## Results

### Correctness

All tests pass:
- water: `vxc max_abs_err=8.497e-07` vs CPU
- benzene: `vxc max_abs_err=4.295e-06` vs CPU
- pentacene: `vxc max_abs_err=5.673e-06` vs CPU
- PTCDA: `vxc max_abs_err=7.130e-6` vs CPU

### Performance

| System | vmat old | vmat new | Speedup | Total old | Total new | CPU |
|--------|----------|----------|---------|-----------|-----------|-----|
| water | 5.17s | 0.027s | 192x | 5.94s | 0.18s | 0.03s |
| benzene | 5.17s | 0.089s | 58x | 5.92s | 0.58s | 0.49s |
| pentacene | ~40s | 0.57s | ~70x | 40.6s | 2.98s | 4.53s |
| PTCDA | — | 0.86s | — | — | 4.33s | 8.84s |

GPU is now **faster than CPU** for benzene, pentacene, and PTCDA.

### Current bottleneck

`rho` kernel is now the dominant cost:

| System | rho | vmat | Total |
|--------|-----|------|-------|
| benzene | 0.140s | 0.089s | 0.58s |
| pentacene | 1.182s | 0.570s | 2.98s |
| PTCDA | 1.896s | 0.859s | 4.33s |

## Future Optimizations

1. **Increase NPTILE** to 32 or 64 to reduce grid tile iterations (currently `ngrids/16`). Local memory: `2*32*60*4 = 15 KB` (NPTILE=32) or `2*64*60*4 = 30 KB` (NPTILE=64).

2. **Optimize rho kernel**: Now the bottleneck. Similar local AO cache approach could help, though rho already uses local `wfRj` and `dm_blk`.

3. **Benchmark NATILE variants**: `NATILE=2` reduces `VBLK_SIZE` to `900` (QPT=4), less register pressure but more workgroups.

4. **Precompute and cache buffers**: `buf_vmat`, `buf_wv`, `buf_rho` are allocated and zeroed every call. Could be preallocated.
