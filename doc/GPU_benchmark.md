---
type: BenchmarkReport
title: GPU XC Stage Benchmarks (Benzene)
description: Per-cycle veff XC timing for OpenCL paths — wall+finish vs OpenCL event profiling; hybrid OTF ρ + radial vmat
tags: [opencl, dft, xc, gpu, benchmark, profiling]
timestamp: 2026-07-08
---

Benzene (`ccpvdz`, grid level 3, `nao=114`, `ngrids=143560`, OMP=1, RTX 3090). One timed **veff XC call** per path (warm + 1 profiled); setup is one-time pre-SCF.

**Driver:** `expamples_prokop/profile_xc_stages_benzene.py`  
**Profiles:** `pyscf/OpenCL/gpu_profiles.py`  
**Cookbook:** `/home/prokop/git/pyscf/doc/opencl_gpu_paths_cookbook.md`

---

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

Profiles: `production_otf`, `production_otf_quintic`, `production_radial`, `production_otf_radial_vmat`.

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

1. **Hybrid path is fastest per cycle** — **21.4 ms** (`production_otf_radial_vmat`): OTF Hermite ρ + radial-gather vmat. Same \|vxc\| as full radial precomp (3.15e-05).
2. **vmat dominates OTF** (~22.6 ms of ~29 ms); radial gather cuts vmat by ~7 ms.
3. **ρ is cheap on all GPU paths** (~4–5 ms); not the optimization bottleneck.
4. **Quintic ≈ cubic** per cycle (28.1 vs 29.0 ms); quintic wins on **setup** (193 vs 404 ms), not bandwidth (memory-equivalent table size).
5. **Host overhead negligible** (~0.4–1.1 ms): GPU c2s, GPU nelec/exc reduction, minimal D2H.
6. **PBE on GPU** ~0.3 ms — not a bottleneck.
7. **CPU libxc** ~21× slower than best GPU path for this XC step alone.

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
