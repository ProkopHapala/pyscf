---
type: TopicalAudit
title: OpenCL DFT XC Grid and DF J/K
tags: [opencl, dft, xc, gpu]
---

## Summary

GPU offload of the RKS exchange–correlation grid integral (ρ projection → PBE vxc → vmat) and density-fitting Coulomb J is implemented in `pyscf/OpenCL/`, integrated into `pyscf/dft/rks.py` via `mf.backend` and `mf.setup_gpu()`. Production path is **Hermite on-the-fly** (`production_otf`): no full χ storage, pair-gather ρ/vmat kernels with radial Hermite eval in registers/local memory. Precomputed GTO paths (row-major, coalesced, radial-precomp) exist for parity/debug and large-nao cases. PBE on GPU is verified vs libxc; full SCF converges on benzene with ~3e-6 max |vxc| error (f32 ρ).

## Implementations

| Location | Status | Notes |
|----------|--------|-------|
| `pyscf/OpenCL/xc_grid.py` — `XCGridPlan`, setup/run API | active | Orchestrates ρ, PBE, vmat; LDA+GGA; stage timing |
| `pyscf/OpenCL/kernels.cl` — tiled ρ/vmat, pair kernels | active | 55 kernels; OTF pair, precomp pair/coalesced/radial, legacy block paths |
| `pyscf/OpenCL/pbe.cl` — GPU PBE vxc | active | f32 default; f64 path with D2H for high precision; unpolarized PBE only |
| `pyscf/OpenCL/hermite_spline.py` + `radial_hermite.py` | active | Host radial table build; mapped u-grid, cubic/quintic |
| `pyscf/OpenCL/ao_hermite.py` — GPU Hermite AO setup | active | Optional pre-SCF AO materialization (`ao_proj='hermite_gpu'`) |
| `pyscf/OpenCL/grid_screen.py` — atom tile screening | active | Rcut from Hermite tails; sparse pair atom lists |
| `pyscf/OpenCL/df_jk.py` — RI J/K on GPU | active | Separate from XC; `mf.with_df.backend=2` |
| `pyscf/OpenCL/gpu_profiles.py` | active | Named presets; cookbook in `doc/opencl_gpu_paths_cookbook.md` |
| `pyscf/dft/rks.py` — `backend`, `setup_gpu`, `get_veff` | active | Entry point for SCF; backend 3 = CPU/GPU compare |
| CPU reference — `pyscf/dft/numint.py` | active | libxc + CPU `eval_ao`; parity baseline |

## Parity Status

| Pair | Tolerance / result | Test reference |
|------|-------------------|----------------|
| GPU ρ (OTF / precomp) vs CPU `NumInt` | max rel ~1e-5 (f32) | `test_opencl_xc_rho_precomp.py`, `test_opencl_xc_e2e_mols.py` |
| GPU PBE wv vs CPU libxc | max abs ~1e-4–1e-3 on ρ components | `test_opencl_xc_full_gpu_parity.py` |
| GPU vmat vs CPU (given same wv) | max rel ~1e-5 | `test_opencl_xc_vmat_precomp.py` |
| Full path ρ→PBE→vmat vs CPU vxc | max ~3e-6 (benzene cc-pVDZ) | `test_opencl_xc_full_gpu_parity.py`, Part 6 in optimization report |
| SCF energy convergence | matches CPU at conv_tol 1e-8 | `profile_gpu_scf.py`, `test_opencl_xc_scf.py` |
| Hermite AO vs exact GTO | shell-dependent; see quintic report | `test_opencl_hermite_ao.py`, `hermite_radial_study.py` |

## Open Issues

- XC limited to **LDA + GGA PBE** on GPU eval path; other functionals fall back to CPU libxc (`xc_eval='cpu'`)
- **Meta-GGA / hybrid / range-separated** not ported
- Large molecules (nao≈300+): precomp χ can exceed GPU memory; OTF path required
- `MAX_ITILE` / `MAX_AO_ATOM` compile-time caps — molecules with many atoms per tile need tile reconfig or kernel extension
- K matrix on GPU via DF exists but PBE RKS production profile uses J only
- `generate_pbe_cl.py` must be re-run when updating libxc PBE source

---

## Dimer scan XC path benchmarks

### Summary

Inter-fragment distance scans (E_bind vs separation) are the acceptance test for GPU XC paths on non-covalent interactions: each geometry runs full PBE/DF SCF with dm warm-start, comparing CPU libxc to all six OpenCL profiles on the same rigid trajectory. Documented in `/home/prokophapala/git/pyscf/doc/dimer_scan_benchmarks.md`.

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
