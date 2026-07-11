---
type: BenchmarkReport
title: GPU XC Stage Benchmarks (Benzene)
description: Per-cycle veff XC timing for OpenCL paths — wall+finish vs OpenCL event profiling; hybrid OTF ρ + radial vmat; split-K tile sweep
tags: [opencl, dft, xc, gpu, benchmark, profiling]
timestamp: 2026-07-08
last_updated: 2026-07-08-machine-info
---

Benzene (`ccpvdz`, grid level 3, `nao=114`, `ngrids=143560`, OMP=1). One timed **veff XC call** per path (warm + 1 profiled); setup is one-time pre-SCF. Hardware: see **Test machine** below.

**Driver:** `expamples_prokop/profile_xc_stages_benzene.py`  
**Profiles:** `pyscf/OpenCL/gpu_profiles.py`  
**Cookbook:** `/home/prokop/git/pyscf/doc/opencl_gpu_paths_cookbook.md`

**Scope note:** this document is an isolated, single-thread CPU-XC benchmark
for benzene. Do not use its 10–50x XC-only ratios as full-SCF speedups. The
same-input four-thread pentacene/PTCDA cycle decomposition is in
`doc/acceptance_2026-07-11.md` (`profile_gpu_amdahl_strict.py`).

---

## Test machine

Recorded 2026-07-08 on host `GTX3090` (Linux `6.8.0-51-generic`, Ubuntu x86_64). Benchmarks use **`OMP_NUM_THREADS=1`** unless noted (CPU libxc row is single-threaded PySCF `NumInt`).

### CPU (host — libxc reference path)

| Item | Value |
|------|-------|
| Model | **AMD Ryzen 7 5800X** (Zen 3, 1× socket) |
| Cores / threads | 8 cores / 16 threads |
| ISA | x86_64 — AVX2, FMA, BMI2, AES |
| L1d / L1i / L2 / L3 | 256 KiB / 256 KiB / 4 MiB / **32 MiB** |
| RAM | **32 GiB** |
| Governor | `schedutil` (boost enabled, max ~4.85 GHz) |

CPU libxc timings in this report run on this host with `lib.num_threads(1)` — not the OpenCL device.

### OpenCL — GPU device used for all GPU paths

`pyscf/OpenCL/__init__.py:init_device` auto-selects **NVIDIA** when present.

| Item | Value |
|------|-------|
| Platform | NVIDIA CUDA — **OpenCL 3.0** (CUDA 12.4.131) |
| Device | **NVIDIA GeForce RTX 3090** |
| Driver | **550.120** |
| OpenCL C | 1.2 |
| Compute units | **82** |
| Max clock | 1800 MHz |
| Global memory | **24 GB** GDDR6X (25.4 GB reported) |
| `__local` / WG | 48 KiB max; max WG size **1024** (dims 1024×1024×64) |
| FP32 / FP64 | full rate / supported (production kernels use **f32**) |

Queue created with `PROFILING_ENABLE` for `clGetEventProfilingInfo` (see Profiling methodology).

### OpenCL — other platform (not used in this report)

| Item | Value |
|------|-------|
| Platform | **PoCL** 5.0+debian (LLVM 16, `haswell` target) |
| Device | `cpu-haswell-AMD Ryzen 7 5800X 8-Core Processor` |
| Role | Available for CPU OpenCL experiments; **not** the device behind numbers here |

### Software stack (relevant)

| Component | Notes |
|-----------|-------|
| Python OpenCL | PyOpenCL (distro packages) |
| PySCF | repo checkout via `PYTHONPATH=/home/prokop/git/pyscf` |
| C extensions | pip-installed `.so` fallback (`pyscf/lib/misc.py:load_library`) |
| CPU threads in GPU runs | `OMP_NUM_THREADS=1` in drivers |

Re-query OpenCL devices:

```bash
PYTHONPATH=/home/prokop/git/pyscf python3 -c "
import pyopencl as cl
for p in cl.get_platforms():
    print(p.name, p.version)
    for d in p.get_devices():
        print(' ', d.name, cl.device_type.to_string(d.type), d.driver_version)
"
```


## Profiling methodology

Earlier wall-clock-only timings under-reported GPU work (async queue, no drain). Current instrumentation uses two complementary measures:

| Key suffix | Mechanism | What it measures |
|------------|-----------|------------------|
| `gpu_*` (wall) | `queue.finish()` before/after + `perf_counter` | Total time until GPU queue is drained for that stage |
| `gpu_*_cl` | `clGetEventProfilingInfo` on kernel completion event | Device execution time of the enqueued kernel(s) |

Implementation: `pyscf/OpenCL/gpu_timing.py` (`profile_kernel`, `profile_call`, `event_elapsed_s`). Queue is created with `PROFILING_ENABLE` in `pyscf/OpenCL/__init__.py:init_device`. Stage sums land in `plan.last_timing` when `profile=True`; `_finalize_gpu_timing` in `xc_grid.py` aggregates `gpu_total` and `gpu_total_cl`.

**Validation (this run):** wall and CL agree within ~0.1–0.4 ms per stage on all paths — profiling is trustworthy for optimization decisions.

**Pitfall:** matmul chains (dm→cart c2s) use `profile_call` — wall is accurate, `*_cl` equals wall (no per-GEMM events). Single-kernel stages (ρ, vmat, PBE) get true CL times.

Re-run:

```bash
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 python3 -u expamples_prokop/profile_xc_stages_benzene.py
```

---

## Per-cycle stage timing (ms)

Wall columns from staged `plan.last_timing`; **gpu CL** = sum of `*_cl` kernel events.

| Method | h2d dm | dm→cart | ρ (wall) | ρ (CL) | PBE+xc | vmat (wall) | vmat (CL) | vmat D2H | **gpu CL** | **wall** | \|vxc\| err |
|--------|--------|---------|----------|--------|--------|-------------|-----------|----------|------------|----------|-------------|
| **CPU libxc** | — | — | — | — | — | — | — | — | — | **448** | ref |
| **OTF cubic** | 0.0 | 0.1 | 5.3 | 5.3 | 0.3 | 22.6 | 22.6 | 0.6 | 28.3 | **29.0** | 3.15e-05 |
| **OTF quintic** | 0.0 | 0.1 | 4.3 | 4.3 | 0.3 | 22.4 | 22.4 | 0.9 | 27.1 | **28.1** | 3.19e-05 |
| **Radial precomp** | 0.2 | 0.0 | 5.5 | 5.4 | 0.3 | 15.8 | 15.8 | 0.5 | 21.5 | **22.2** | 3.15e-05 |
| **OTF ρ + rad vmat** | 0.0 | 0.1 | 5.0 | 5.0 | 0.3 | 15.5 | 15.5 | 0.4 | 20.9 | **21.4** | 3.15e-05 |
| **OTF ρ + split-K** | 0.0 | 0.1 | 4.3 | 4.2 | 0.3 | 7.1 | 6.9 | 0.2 | 11.8 | **12.6** | 3.16e-05 |

Profiles: `production_otf`, `production_otf_quintic`, `production_radial`, `production_otf_radial_vmat`, `production_otf_radial_vmat_splitk`.

---

## One-time setup (ms)

| Method | setup | Notes |
|--------|-------|-------|
| OTF cubic | 404 | Hermite tables + kernel compile |
| OTF quintic | **193** | half radial nodes (`memory_equivalent_du`) |
| Radial precomp | 400 | `build_radial_on_grid_tiled` + radial ρ/vmat buffers |
| **OTF ρ + rad vmat** | 397 | same radial build as precomp; OTF ρ tables |

---

## Takeaways

1. **Split-K hybrid is fastest per cycle** — **~12.6 ms** (`production_otf_radial_vmat_splitk`): OTF ρ + split-K radial vmat. ~1.7× faster than non-split hybrid (~21 ms).
2. **Hybrid path (non-split)** — **21.4 ms** (`production_otf_radial_vmat`): OTF Hermite ρ + radial-gather vmat. Same \|vxc\| as full radial precomp (3.15e-05).
3. **vmat dominates OTF** (~22.6 ms of ~29 ms); radial gather cuts vmat by ~7 ms; split-K cuts it further (~16 → ~7 ms).
4. **ρ is cheap on all GPU paths** (~4–5 ms); becoming ~40% of split-K total — next optimization target.
5. **Quintic ≈ cubic** per cycle (28.1 vs 29.0 ms); quintic wins on **setup** (193 vs 404 ms), not bandwidth (memory-equivalent table size).
6. **Host overhead negligible** (~0.4–1.1 ms): GPU c2s, GPU nelec/exc reduction, minimal D2H.
7. **PBE on GPU** ~0.3 ms — not a bottleneck.
8. **CPU libxc** ~50× slower than best GPU path for this XC step alone.

---

## Hybrid path (implemented)

**Profile:** `production_otf_radial_vmat` — `setup_onthefly(..., vmat_mode='radial_precomp')`.

| Stage | Kernel | Work |
|-------|--------|------|
| ρ | `rho_gga_tiled` | OTF Hermite radial eval (same accuracy as `production_otf`) |
| PBE | `pbe_xc_f32` + `compute_wv_gga_f32` | on-device, no ρ/wv PCIe |
| vmat | `vmat_gga_radial_precomp_pair` | gather precomputed `R,dR[ir,g]` — no Hermite in hot loop |

`R,dR` built once at setup via `OpenCLAOHermiteEvaluator.build_radial_on_grid_gpu()` (~included in setup ms). Grid coords fixed → no per-cycle rebuild.

**vs full radial precomp:** hybrid saves nothing on vmat (same kernel) but keeps OTF ρ instead of `rho_gga_radial_precomp_pair` — here ρ cost is similar (~5 ms); hybrid is marginally faster on total wall (21.4 vs 22.2 ms).

**Implementation pitfall (fixed):** radial metadata buffers (`buf_atom_coords_h`, `buf_radial_l_h`, …) must be retained in `XCGridPlan.otf` — kernel args alone do not keep Python references; GC caused `INVALID_MEM_OBJECT`.

---

## Why quintic is not faster (despite “half the nodes”)

Memory-equivalent design: `du_quintic ≈ 2 × du_cubic` → same table bytes, more arithmetic per eval. See `/home/prokop/git/pyscf/doc/quintic_hermite_spline.md`.

| | Cubic | Quintic |
|--|-------|---------|
| `nrad` | ~450 | ~226 |
| bytes/node | 8 | 16 |
| **table size** | ~0.27 MB | ~0.27 MB |
| per-cycle ρ | ~5.3 ms | ~4.3 ms |
| per-cycle vmat | ~22.6 ms | ~22.4 ms |

Quintic wins on **accuracy per byte** and **setup time**, not on per-SCF bandwidth under memory-equivalent encoding.

---

## Bottleneck diagnosis

### ρ (`rho_gga_tiled`) — not the limiter (~5 ms)

Grid-outer tiled WG; Hermite eval into `__local`. `rad_node` (~0.27 MB) fits L2.

### vmat OTF (`vmat_gga_tiled`) — dominant (~22 ms)

Duplicate Hermite work vs ρ; grid loop inside atom-pair WG; `fill_atom_aow_gga` + dot over `NPTILE`.

### vmat radial (`vmat_gga_radial_precomp_pair`) — fast (~15.5 ms)

Gather `rad_val[ir*ngrids+g]` + shell unfold — no spline arithmetic. That's the ~7 ms gap vs OTF vmat.

---

## Optimization roadmap (updated)

### Done

| # | Change | Result |
|---|--------|--------|
| **1** | Event profiling (`gpu_timing.py` + `profile_kernel`) | wall ≈ CL; trustworthy stage breakdown |
| **2** | Hybrid OTF ρ + radial vmat | **21.4 ms** — best per-cycle path |
| **3** | Quintic spline OTF ρ (`spline_order='quintic'`) | parity OK; setup 2× faster, cycle ≈ cubic |

### Next (OTF vmat still ~22 ms if hybrid not used)

| # | Change | Hypothesis |
|---|--------|------------|
| **4** | Hoist `hermite_map_point` per (g, atom) | less duplicate spline work in `fill_atom_*` |
| **5** | Compile-time cubic vs quintic kernels (`-DSPLINE_ORDER`) | remove runtime branch |
| **6** | Vectorize vmat inner `ip` dot (`float4`/`float8`) | memory-bound accumulation |
| **7** | Upper-triangle atom tiles for GGA | skip redundant atom-pair WGs |

### Do not pursue first

- Coalesced χ — high ρ gather cost on benzene
- Fusing PBE into ρ — PBE is 0.3 ms
- Quintic for speed under memory-equivalent `du` — contradicts design

---

## Tests and parity

| Test | What |
|------|------|
| `expamples_prokop/profile_xc_stages_benzene.py` | stage timing table (this report) |
| `expamples_prokop/test_opencl_xc_onthefly.py` | OTF end-to-end parity + speedup |
| `expamples_prokop/test_quintic_rho_otf.py` | quintic vs cubic ρ on formic dimer |
| `expamples_prokop/sweep_splitk_tiles.py` | 1-neighborhood tile/WGS/splits sweep for split-K |

---

## Report: split-K vmat and tile-parameter sweep (2026-07-08)

**System:** benzene, `ccpvdz`, grid level 3, `nao=114`, `ngrids=143560`, OMP=1 — hardware per **Test machine** above.  
**Drivers:** `expamples_prokop/profile_xc_stages_benzene.py`, `expamples_prokop/sweep_splitk_tiles.py`  
**Profile:** `production_otf_radial_vmat_splitk` in `pyscf/OpenCL/gpu_profiles.py`

### Summary

Split-K radial vmat (`vmat_gga_radial_precomp_pair_splitk` + `reduce_split_vmat`) cuts vmat CL from ~16 ms to ~7 ms vs the non-split hybrid on benzene. Total XC per `veff` call drops from ~21 ms to ~12 ms (gpu CL ~12 ms) with \|vxc\| ≈ 3.16e-05. Tile parameters were tuned with a **1-neighborhood coordinate descent** on a power-of-2 lattice (not brute-force Cartesian product). Locked defaults: `NPTILE=64`, `NATILE=2`, `WGS_VMAT=128` (split-K profile only), `vmat_grid_splits=64`.

---

### 1. Full path comparison (ms per veff XC call)

From `profile_xc_stages_benzene.py` (warm + one profiled call; run-to-run ±~1 ms).

| Method | setup | outer | gpu CL | ρ_cl | vmat_cl | xc | \|vxc\| |
|--------|------:|------:|-------:|-----:|--------:|---:|--------:|
| CPU libxc | 0 | 638 | 0 | — | — | — | ref |
| OTF cubic | 443 | 28 | 27 | 4.8 | 21.3 | 0.7 | 3.15e-05 |
| OTF quintic | 208 | 49 | 42 | 4.9 | 31.2 | 2.8 | 3.19e-05 |
| Radial precomp | 422 | 23 | 22 | 5.3 | 15.9 | 0.4 | 3.15e-05 |
| OTF ρ + rad vmat | 425 | 22 | 21 | 4.2 | 15.9 | 0.3 | 3.15e-05 |
| **OTF ρ + split-K** | 599 | **13** | **12** | 4.2 | **6.9** | 0.6 | 3.16e-05 |

Split-K stage breakdown (representative): `gpu_vmat_split` ≈ 6.9 ms CL, `gpu_vmat_reduce` ≈ 0.01 ms CL.

---

### 2. Split-K implementation

| Stage | Kernel | Notes |
|-------|--------|-------|
| ρ | `rho_gga_tiled` | OTF Hermite (same as hybrid) |
| PBE | `pbe_xc_f32` + `compute_wv_gga_f32` | on-device |
| vmat | `vmat_gga_radial_precomp_pair_splitk` | 3D launch; each WG owns one grid chunk → partial vmat |
| reduce | `reduce_split_vmat` | sum `partial_vmat[nsplit,ncart,ncart]` → `vmat` |

| Knob | Where | Locked value | Notes |
|------|-------|-------------|-------|
| `NPTILE`, `NATILE` | `TileConfig` → `kernels.cl` | 64, 2 | compile-time |
| `WGS_VMAT` | `TileConfig` | **128** | via `_ensure_splitk_tile_config()` in `apply_gpu_profile`; **not** global default (WGS=128 globally regresses OTF tiled vmat ~2×) |
| `vmat_grid_splits` | `setup_kw` | **64** | runtime; grid-parallel factor K |

---

### 3. Sweep methodology: 1-neighborhood coordinate descent

Tile knobs `{NPTILE, NATILE, WGS_VMAT, vmat_grid_splits}` sit on a **power-of-2 lattice** and are **resource-coupled** (`WGS_VMAT ≥ NPTILE×NATILE`; larger tiles consume more `__local` / lanes). Optima often lie on **diagonals** (trade one axis for another), not on a full Cartesian grid.

**Probe set from center** `c = (NPTILE, NATILE, WGS, splits)`:

| Move type | Rule |
|-----------|------|
| **Axis** | Change **exactly one** parameter by **one** step (×2 or ÷2) |
| **Diagonal** | Change **two** parameters by one step each — typically opposite on the resource budget: `NPTILE×2,WGS÷2`; `NATILE×2,WGS÷2`; `NPTILE×2,NATILE÷2`; `splits×2,WGS÷2` |

**Rules:** at most one power-of-2 step per axis per probe; skip invalid configs; require parity OK (`|vxc| < 1e-4`); pick best neighbor; recenter and repeat until no gain.

**Multi-hop:** `WGS=64` or `32` from `(64,2,128,*)` is invalid as a single-axis move — requires coupled descent (e.g. `NATILE→1` then `WGS÷2`).

```bash
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 python3 -u \
  expamples_prokop/sweep_splitk_tiles.py --neighbor --seed 64,2,128,32 \
  --out debug/sweep_splitk_tiles/benzene_neighbor.csv
```

Legacy `--quick` (brute-force subset) kept for first exploration only.

---

### 4. Pass 1 — neighborhood from seed `(64, 2, 128, 32)`

11 axis + diagonal probes. CSV: `debug/sweep_splitk_tiles/benzene_neighbor.csv`. Metric: `gpu_total_cl_ms` (OpenCL event sum).

| NPTILE | NATILE | WGS | splits | outer (ms) | gpu CL (ms) | vmat CL (ms) | ρ_cl (ms) | Note |
|-------:|-------:|----:|-------:|-----------:|------------:|-------------:|----------:|------|
| 64 | 2 | 128 | **64** | 11.2 | **10.9** | **6.3** | 4.2 | **best gpu CL in pass 1** |
| 64 | 2 | 128 | 16 | 11.6 | 11.3 | 6.7 | 4.2 | fewer splits |
| 128 | 1 | 128 | 32 | 11.4 | 11.1 | 6.3 | 4.4 | diagonal NPTILE×2 NATILE÷2 |
| 64 | 1 | 256 | 32 | 12.0 | 11.8 | 6.9 | 4.5 | diagonal NATILE÷2 WGS×2 |
| 64 | 2 | 128 | 32 | 12.3 | 13.7 | 7.8 | 5.3 | **seed (center)** |
| 64 | 1 | 128 | 32 | 12.8 | 12.9 | 7.2 | 5.4 | NATILE=1 |
| 64 | 2 | 256 | 32 | 13.6 | 13.3 | 8.3 | 4.7 | WGS too large |
| 32 | 2 | 128 | 32 | 14.5 | 14.1 | 8.8 | 4.7 | NPTILE too small |
| 32 | 4 | 128 | 32 | 15.0 | 14.8 | 7.9 | 6.5 | NATILE=4 diagonal |
| 32 | 2 | 256 | 32 | 17.6 | 17.0 | 11.6 | 5.1 | diagonal NPTILE÷2 WGS×2 |

**Pass 1 conclusion:** move `splits` 32→64; keep `(64,2,128)` tile shape; recenter to `(64,2,128,64)`.

---

### 5. Pass 2 — neighborhood from seed `(64, 2, 128, 64)`

| NPTILE | NATILE | WGS | splits | outer (ms) | gpu CL (ms) | vmat CL (ms) | Note |
|-------:|-------:|----:|-------:|-----------:|------------:|-------------:|------|
| 64 | 2 | 128 | **128** | 13.2 | **11.9** | **6.3** | best gpu CL in pass 2 |
| 128 | 1 | 128 | 64 | 12.4 | 12.3 | 6.2 | best vmat CL; diagonal branch |
| 64 | 2 | 128 | 32 | 12.5 | 12.5 | 6.9 | splits÷2 |
| 64 | 2 | 128 | 64 | 13.8 | 14.2 | 8.6 | center (run variance) |
| 64 | 2 | 256 | 64 | 13.1 | 13.5 | 8.0 | WGS×2 |
| 32 | 4 | 128 | 64 | 15.9 | 14.8 | 7.6 | NATILE=4 diagonal |
| 32 | 2 | 128 | 64 | 15.8 | 27.6 | 8.9 | NPTILE÷2 (outlier run) |

**Pass 2 conclusion:** `splits=128` competitive with `64` on gpu CL (~11.9 vs ~10.9–14 ms depending on run); `(128,1,128)` branch viable. Profile locked at `splits=64` as stable default; `128` worth re-check under load.

---

### 6. Extended probes — `NATILE=4`, `WGS ∈ {64, 32}`

Configs not reachable in one hop from `(64,2,128,*)` without coupled moves; probed explicitly.

| NPTILE | NATILE | WGS | splits | gpu CL (ms) | vmat CL (ms) | Verdict |
|-------:|-------:|----:|-------:|------------:|-------------:|---------|
| 64 | 4 | 256 | 64 | 12.6 | 7.4 | NATILE=4 — not better |
| 64 | 1 | 64 | 64 | 13.2 | 8.0 | WGS=64 — not better |
| 32 | 2 | 64 | 64 | 12.8 | 7.3 | WGS=64 — not better |
| 64 | 1 | 128 | 64 | 12.0 | 6.8 | reference NATILE=1 axis |
| 32 | 1 | 64 | 32 | 13.0 | 7.5 | WGS=64, NATILE=1 |

**Conclusion:** on benzene, `NATILE=4` and smaller `WGS` do not beat `(64,2,128)`. `WGS=128` remains optimal for split-K pair vmat; `WGS=256` consistently worse (idle lanes during NPTILE fill).

---

### 7. Conclusions and next steps

| Finding | Action |
|---------|--------|
| Split-K ~2.5× vmat speedup vs hybrid | `production_otf_radial_vmat_splitk` is best per-cycle GPU XC path |
| `(64,2,128)` optimal tile shape on benzene | keep `TileConfig` defaults |
| `WGS_VMAT=128` split-K only | `_ensure_splitk_tile_config()`; global default stays 256 |
| `vmat_grid_splits=64` | locked in profile (`32` and `128` within noise) |
| `NATILE=4`, `WGS∈{64,32}` ruled out on benzene | no change |
| ρ ~4–5 ms, now ~40% of split-K total | next target: radial-precomp ρ kernel |
| Benzene-only tuning | re-run `--neighbor` on H₂O / formic dimer before hard-locking |

**Re-run full stage table:**

```bash
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 python3 -u expamples_prokop/profile_xc_stages_benzene.py
```

**Continue tile descent from new center:**

```bash
PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 python3 -u \
  expamples_prokop/sweep_splitk_tiles.py --neighbor --seed 64,2,128,64
```
