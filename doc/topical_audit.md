---
type: TopicalAudit
title: OpenCL DFT XC Grid and DF J/K
tags: [opencl, dft, xc, gpu]
---

## Summary

GPU offload of the RKS exchange–correlation grid integral (ρ projection → PBE vxc → vmat) and density-fitting Coulomb J is implemented in `pyscf/OpenCL/`, integrated into `pyscf/dft/rks.py` via `mf.backend` and `mf.setup_gpu()`. Profile setup now hoists static grid/DF/GPU-plan work before `mf.kernel()`; only density-dependent XC/J/K work remains per SCF cycle. **Best per-cycle XC path (benzene cc-pVDZ, RTX 3090):** `production_otf_radial_vmat_splitk` — OTF Hermite ρ + split-K radial-gather vmat (~12 ms gpu CL vs ~21 ms non-split hybrid vs ~29 ms full OTF). Non-split hybrid `production_otf_radial_vmat` remains valid. Default general path: `production_otf` (no radial setup). Stage timing: `gpu_timing.py` (wall+`queue.finish()` and `clGetEventProfilingInfo`). PBE on GPU verified vs libxc; max |vxc| ~3e-5 on benzene (f32 ρ).

## Implementations

| Location | Status | Notes |
|----------|--------|-------|
| `pyscf/OpenCL/xc_grid.py` — `XCGridPlan`, setup/run API | active | ρ/PBE/vmat; `vmat_mode='radial_precomp'`; `vmat_grid_splits` split-K; `plan.last_timing` |
| `pyscf/OpenCL/gpu_timing.py` — kernel profiling helpers | active | `profile_kernel` (wall+CL events), `profile_call`; requires `PROFILING_ENABLE` queue |
| `pyscf/OpenCL/kernels.cl` — tiled ρ/vmat, pair kernels | active | OTF tiled/pair, quintic Hermite, radial precomp, **split-K vmat + reduce**, PBE, reductions |
| `pyscf/OpenCL/pbe.cl` — GPU PBE vxc | active | f32 default; f64 path with D2H for high precision; unpolarized PBE only |
| `pyscf/OpenCL/hermite_spline.py` + `radial_hermite.py` | active | Host radial table build; mapped u-grid, cubic/quintic |
| `pyscf/OpenCL/ao_hermite.py` — GPU Hermite AO setup | active | Optional pre-SCF AO materialization (`ao_proj='hermite_gpu'`) |
| `pyscf/OpenCL/grid_screen.py` — atom tile screening | active | Rcut from Hermite tails; sparse pair atom lists |
| `pyscf/OpenCL/df_jk.py` — RI J/K on GPU | active | Separate from XC; `mf.with_df.backend=2` |
| `pyscf/OpenCL/gpu_profiles.py` | active | named profiles plus static `prepare_df_for_scf`; cookbook in `doc/opencl_gpu_paths_cookbook.md` |
| `expamples_prokop/profile_xc_stages_benzene.py` | active | Per-stage wall vs CL timing; benzene benchmark driver for `doc/GPU_benchmark.md` |
| `expamples_prokop/profile_gpu_amdahl_strict.py` | active | Same-input, additive CPU/GPU full-cycle profile; strict reference for large-molecule Amdahl claims |
| `expamples_prokop/sweep_splitk_tiles.py` | active | `--neighbor` tile/WGS/splits sweep for split-K profile |
| `pyscf/dft/rks.py` — `backend`, `setup_gpu`, `get_veff` | active | Entry point for SCF; backend 3 = CPU/GPU compare |
| CPU reference — `pyscf/dft/numint.py` | active | libxc + CPU `eval_ao`; parity baseline |

## Parity Status

| Pair | Tolerance / result | Test reference |
|------|-------------------|----------------|
| GPU ρ (OTF / precomp) vs CPU `NumInt` | max rel ~1e-5 (f32) | `test_opencl_xc_rho_precomp.py`, `test_opencl_xc_e2e_mols.py` |
| GPU PBE wv vs CPU libxc | max abs ~1e-4–1e-3 on ρ components | `test_opencl_xc_full_gpu_parity.py` |
| GPU vmat vs CPU (given same wv) | max rel ~1e-5 | `test_opencl_xc_vmat_precomp.py` |
| Full path ρ→PBE→vmat vs CPU vxc | max ~3e-5 (benzene cc-pVDZ, f32) | `test_opencl_xc_onthefly.py`, `test_opencl_xc_full_gpu_parity.py` |
| Hybrid OTF ρ + radial vmat vs CPU vxc | max ~3.15e-5 | `profile_xc_stages_benzene.py`, `production_otf_radial_vmat` |
| Split-K OTF ρ + radial vmat vs CPU vxc | max ~3.16e-5 | `profile_xc_stages_benzene.py`, `production_otf_radial_vmat_splitk` |
| Quintic OTF ρ vs cubic OTF | shell-dependent; memory-equivalent du | `test_quintic_rho_otf.py` |
| SCF energy convergence | matches CPU at conv_tol 1e-8 | `profile_gpu_scf.py`, `test_opencl_xc_scf.py` |
| Hermite AO vs exact GTO | shell-dependent; see quintic report | `test_opencl_hermite_ao.py`, `hermite_radial_study.py` |

## Open Issues

- XC limited to **LDA + GGA PBE** on GPU eval path; other functionals fall back to CPU libxc (`xc_eval='cpu'`)
- **Meta-GGA / hybrid / range-separated** not ported
- Large molecules (nao≈300+): precomp χ can exceed GPU memory; OTF path required
- Tile defaults tuned on **benzene** — re-validate on other molecules before hard-locking (`sweep_splitk_tiles.py --neighbor`)
- `WGS_VMAT` optimal value is **profile-specific** (128 for split-K; 256 default for OTF tiled)
- `MAX_ITILE` / `MAX_AO_ATOM` compile-time caps — molecules with many atoms per tile need tile reconfig or kernel extension
- K matrix on GPU via DF exists but PBE RKS production profile uses J only
- `generate_pbe_cl.py` must be re-run when updating libxc PBE source

---

## CPU smallDFT — grid-parallel XC (nao ≲ 200)

### Summary

CPU fast path for RKS grid XC on small molecules: OpenMP ρ and vmat in `libsmalldft.so`, libcint F-contiguous AO layout, drop-in `pyscf.smallDFT.nr_rks`. Complements the OpenCL GPU path for systems where GPU offload is unavailable or AO is cached on CPU. Python `ThreadPoolExecutor` grid tiles are legacy fallback only.

### Implementations

| Location | Status | Notes |
|----------|--------|-------|
| `pyscf/smallDFT/nr_rks.py` — drop-in `nr_rks` | active | C dispatch when `libsmalldft` built; `GridWorkspace` AO cache |
| `pyscf/lib/smalldft/small_grid.c` — `SMALL_rho_*`, `SMALL_vmat_*` | active | TILE=512, strided BLAS, private vmat + hermi fix |
| `pyscf/smallDFT/rho.py`, `vmat.py` | active | `use_c=True` → ctypes; Python threads deprecated |
| `pyscf/smallDFT/workspace.py` — `GridWorkspace` | active | prealloc ρ/vmat plus raw AO buffer filled directly by libcint |
| `pyscf/smallDFT/patch.py` — `enable()` | experimental | monkey-patch `NumInt.nr_rks` |
| `pyscf/dft/numint.py` — reference CPU | active | parity baseline; OMP in libcint/libdft |
| OpenCL `pyscf/OpenCL/xc_grid.py` | active | GPU analogue; see OpenCL topical section above |

### Parity Status

| Pair | Tolerance / result | Test reference |
|------|-------------------|----------------|
| C ρ_gga vs Python `rho_gga` | ~1e-12 | `expamples_prokop/test_small_dft.py --rho` |
| C vmat_gga vs Python | ~1e-14 | same |
| `smallDFT.nr_rks` vs `numint.nr_rks` (PBE) | vmat ~1e-14 | `expamples_prokop/test_small_dft.py` |
| benzene scaling ρ @8 CPU | 5.94× (sub-task) | `doc/CPU_benchmark.md` |

### Open Issues

- **RI-J / DF J** becomes the largest converged-cycle cost after cached XC is accelerated
- **Tiled rho → libxc → vmat** is not implemented — current GGA path makes two χ passes per `get_veff`
- **vmat scales 3.9×** on 8 CPU (memory-bound GEMM vs ρ 5.9×)
- **Python thread path** — do not extend; C-only policy
- **MGGA / UKS / sparse screening** — not ported; LDA+GGA RKS only

Doc: `/home/prokop/git/pyscf/doc/smallDFT_cpu_path.md` · Benchmarks: `/home/prokop/git/pyscf/doc/CPU_benchmark.md`

---

## Dimer scan XC path benchmarks

### Summary

Inter-fragment distance scans (E_bind vs separation) are the acceptance test for GPU XC paths on non-covalent interactions: each geometry runs full PBE/DF SCF with dm warm-start, comparing CPU libxc to all six OpenCL profiles on the same rigid trajectory. Documented in `/home/prokop/git/pyscf/doc/dimer_scan_benchmarks.md`.

### Implementations

| Location | Status | Notes |
|----------|--------|-------|
| `expamples_prokop/profile_dimer_scan.py` | active | General driver; `--n0` = first atom of fragment 2 |
| `expamples_prokop/dimer_scan_frames.py` | active | Rigid shift from relaxed XYZ + distance grid |
| `expamples_prokop/xc_path_modes.py` | active | SSOT path keys → `gpu_profiles.py` presets |
| `expamples_prokop/plot_scan_ez.py` | active | Primary E(z) plot — all paths on one figure |
| `expamples_prokop/plot_h2o_dimer_scan_energy.py` | active | 4-panel diagnostic + text analysis |
| `expamples_prokop/profile_xc_paths_single.py` | active | Single-point ΔE bar chart only (not scan) |
| `debug/profile_h2o_dimer_scan/` | active | H₂O results (39 pt DFTB grid, Jul 2026) |
| `debug/profile_formic_dimer_scan/` | active | Formic results (rigid shift, n0=5, Jul 2026) |

### Parity Status

| Pair | Tolerance / result | Test reference |
|------|-------------------|----------------|
| gpu_otf / gpu_coalesced / gpu_radial vs CPU E_bind(z) | RMS ≤ 0.005 kcal/mol (H₂O, formic) | `debug/profile_*_dimer_scan/scan_scf_profile.csv` |
| gpu_gto vs CPU E_bind(z) | H₂O outlier ~0.09 kcal/mol RMS; formic OK ~0.002 | same CSVs |
| gpu_full vs CPU E_bind(z) | ~0.015–0.03 kcal/mol RMS (relaxed SCF tol 1e-6) | same CSVs |
| gpu_coalesced multi-frame scan | no MEM_OBJECT_ALLOCATION_FAILURE | requires `release_gpu_between_frames` in scan driver |

### Open Issues

- **gpu_gto** H₂O scan deviation (~0.1 kcal/mol) — investigate; formic within spec
- DFTB reference overlay is not a parity target (qualitative shape only)
