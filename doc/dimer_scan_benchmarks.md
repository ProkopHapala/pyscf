---
type: ConceptDoc
title: Dimer Scan XC Path Benchmarks
description: Inter-fragment distance scans comparing CPU libxc vs all GPU OpenCL XC execution paths
tags: [benchmark, dimer, xc, gpu, parity, opencl]
timestamp: 2026-07-08
---

## Summary

Rigid **inter-fragment distance scans** on dimers validate that every GPU OpenCL XC path reproduces the **CPU PBE/DF binding curve** E(z), not just single-point energies. A scan exercises changing density, grid screening, and SCF warm-start — failure modes that bar charts at one geometry miss. The workhorse is `expamples_prokop/profile_dimer_scan.py`; path labels and kernel mapping live in `xc_path_modes.py` (SSOT).

**Primary deliverable:** `energy_profile_ez.png` — CPU + all GPU paths on one E_bind(z) plot (kcal/mol) plus ΔE_bind vs CPU panel. Secondary: 4-panel diagnostic via `plot_h2o_dimer_scan_energy.py`, single-point bar chart via `profile_xc_paths_single.py`.

## Fragment split (`--n0`)

| Concept | Definition |
|---------|------------|
| **n0** | 0-based index of the **first atom in fragment 2** (mobile monomer) |
| Fragment 1 (fixed) | atoms `0 … n0−1` |
| Fragment 2 (mobile) | atoms `n0 … natom−1` — translated rigidly along the scan axis |
| Scan coordinate **z** | Distance between anchor atoms (auto: closest cross-fragment pair; prefers O···O when both sides have oxygen) |

Examples: H₂O dimer `n0=3`; formic dimer (10 atoms) `n0=5` → anchor O(3)···O(7).

## Geometry sources

| Mode | Flags | Use when |
|------|-------|----------|
| Pre-built trajectory | `--scan-xyz` + optional `--distances-file` | DFTB/MM scan already exists (H₂O DFTB grid) |
| Rigid shift from one XYZ | `--geom` + `--distances-file` + **`--n0`** | Only relaxed dimer geometry available (formic) |

Distance grid: one distance per line in `distances.dat` (Å). Default H₂O grid: 39 points, 0.1 Å near r_eq, coarser at long range (`CompChemUtils/tmp/H2O_dimer_scan_dftb/`).

## XC paths compared

Defined in `expamples_prokop/xc_path_modes.py`. See also `/home/prokophapala/git/pyscf/doc/opencl_gpu_paths_cookbook.md`.

| Mode key | Label | ρ / vmat | J | SCF tol |
|----------|-------|----------|---|---------|
| `cpu` | CPU libxc | CPU eval_ao + libxc | CPU RI-J | 1e-8 |
| `gpu_otf` | Hermite OTF | pair-gather, in-kernel Hermite | CPU RI-J | 1e-8 |
| `gpu_coalesced` | Precomp coalesced | gather χ[iAO,iG] | CPU RI-J | 1e-8 |
| `gpu_radial` | Radial precomp | R,dR on grid; χ only for vmat | CPU RI-J | 1e-8 |
| `gpu_gto` | Exact GTO χ | CPU eval_ao → upload χ | CPU RI-J | 1e-8 |
| `gpu_full` | OTF fast (GPU J) | **same kernels as gpu_otf** | GPU RI-J f32 | **1e-6** |

`gpu_full` vs `gpu_otf`: identical ρ/vmat integration; differences are GPU Coulomb J and relaxed SCF tolerance.

## Verified parity (PBE/6-31g, grid level 3, DF, Jul 2026)

Reference: CPU at each geometry; binding E_bind(z) = [E_tot(z) − E_tot(z_ref=20 Å)] × 627.5095 kcal/mol/Ha.

### H₂O dimer (39 points, DFTB xyz grid, n0=3)

| Path | ΔE_bind RMS vs CPU | max \|ΔE_bind\| | Notes |
|------|-------------------|-----------------|-------|
| gpu_otf | 0.0047 kcal/mol | 0.0055 | production path |
| gpu_coalesced | 0.0044 | 0.0053 | χ memory ~28 MB/frame; requires `release_gpu_between_frames` |
| gpu_radial | 0.0045 | 0.0054 | fastest GPU path on this system |
| gpu_gto | 0.0892 | 0.1019 | outlier on H₂O — investigate |
| gpu_full | 0.0152 | 0.0295 | SCF tol + GPU J |

CPU r_min = 2.804 Å, E_bind_min = −8.24 kcal/mol.

### Formic dimer (39 points, rigid shift from `data/xyz/formic_dimer.xyz`, n0=5)

| Path | ΔE_bind RMS vs CPU | max \|ΔE_bind\| |
|------|-------------------|-----------------|
| gpu_otf | 0.0009 kcal/mol | 0.0017 |
| gpu_coalesced | 0.0018 | 0.0034 |
| gpu_radial | 0.0009 | 0.0020 |
| gpu_gto | 0.0016 | 0.0061 |
| gpu_full | 0.0304 | 0.0596 |

CPU r_min = 2.704 Å, E_bind_min = −21.66 kcal/mol.

DFTB reference overlay is qualitative only (different method/basis); do not use for GPU parity gates.

**Detailed report:** [`/home/prokophapala/git/pyscf/doc/opencl-xc-reports/2026-07-dimer-scan-xc-paths.md`](/home/prokophapala/git/pyscf/doc/opencl-xc-reports/2026-07-dimer-scan-xc-paths.md)

## File map

| File | Role |
|------|------|
| `expamples_prokop/profile_dimer_scan.py` | General scan driver: timers, dm warm-start, CSV, auto-plots |
| `expamples_prokop/dimer_scan_frames.py` | Rigid shift frame builder from relaxed XYZ + distance grid |
| `expamples_prokop/xc_path_modes.py` | SSOT for path keys, labels, kernel descriptions |
| `expamples_prokop/plot_scan_ez.py` | Essential E(z) two-panel plot from CSV |
| `expamples_prokop/plot_h2o_dimer_scan_energy.py` | 4-panel diagnostic + `energy_analysis.txt` |
| `expamples_prokop/profile_h2o_dimer_scan.py` | Thin wrapper: H₂O defaults (n0=3, DFTB paths) |
| `expamples_prokop/profile_formic_dimer_scan.py` | Thin wrapper: formic (n0=5, `--geom`) |
| `expamples_prokop/profile_xc_paths_single.py` | Single-geometry ΔE bar chart (not a scan) |

## Outputs (under `debug/`)

| Run | Directory | Key artifacts |
|-----|-----------|---------------|
| H₂O | `debug/profile_h2o_dimer_scan/` | `scan_scf_profile.csv`, **`energy_profile_ez.png`**, `energy_profile.png`, `energy_analysis.txt` |
| Formic | `debug/profile_formic_dimer_scan/` | same layout |

CSV columns include per-frame `r_A`, `mode`, `E_Ha`, SCF/setup timing, GPU ρ/PBE/vmat stage ms.

## Tutorial

```bash
# Any dimer — set n0 to first atom index of fragment 2
PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 python3 -u \
  expamples_prokop/profile_dimer_scan.py \
  --geom data/xyz/formic_dimer.xyz --n0 5 \
  --distances-file /path/to/distances.dat \
  --mode cpu gpu_otf gpu_coalesced gpu_radial gpu_gto gpu_full

# H₂O with pre-built DFTB trajectory
PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 python3 -u \
  expamples_prokop/profile_h2o_dimer_scan.py \
  --mode cpu gpu_otf gpu_coalesced gpu_radial gpu_gto gpu_full

# Replot E(z) from existing CSV
python3 expamples_prokop/plot_scan_ez.py \
  --csv debug/profile_formic_dimer_scan/scan_scf_profile.csv \
  --title "formic dimer" --z-label "O···O distance (Å)"
```

Wrappers pass through extra CLI flags (`--max-frames`, `--no-plot-diag`, etc.).

## Notes and pitfalls

- **dm warm-start** between geometries is default; use `--cold-each-frame` to disable. Warm-start reduces cycles but can mask discontinuities — compare cold run if binding curve looks noisy.
- **GPU memory:** `gpu_coalesced` uploads full χ each setup; `_xc_plan_cache` must be cleared between frames (`release_gpu_between_frames` in scan driver) or scan fails with `MEM_OBJECT_ALLOCATION_FAILURE` mid-run.
- **Binding reference distance:** plots use z_ref = 20 Å (last grid point). ylim on E(z): `vmin = E_min(CPU)×1.2`, `vmax = −2×vmin`.
- **Single-point bar charts** (`profile_xc_paths_single.py`) do not substitute for E(z); always run `profile_dimer_scan.py` for path comparison on scans.
- **Anchor override:** `--anchor-fixed` / `--anchor-mobile` when auto pair is wrong; `--no-prefer-o` for non-O contact pairs.

## Related

- `/home/prokophapala/git/pyscf/doc/topical_audit.md` — OpenCL XC implementation map
- `/home/prokophapala/git/pyscf/doc/opencl_gpu_paths_cookbook.md` — profile knobs
- `/home/prokophapala/git/pyscf/expamples_prokop/README.md` — script index
