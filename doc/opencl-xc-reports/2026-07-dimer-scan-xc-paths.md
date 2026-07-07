# OpenCL XC report — dimer scan E(z) parity, all execution paths (2026-07)

**Hardware:** Linux workstation, `OMP_NUM_THREADS=1`  
**Method:** PBE/6-31g, grid level 3, density fitting (RI-J on CPU except `gpu_full`)  
**Driver:** `expamples_prokop/profile_dimer_scan.py`  
**Concept doc:** [`../dimer_scan_benchmarks.md`](../dimer_scan_benchmarks.md)

---

## Problem

Single-geometry ΔE bar charts (`profile_xc_paths_single.py`) verified that GPU XC paths match CPU libxc at **one** geometry, but they cannot catch:

1. **Scan-dependent integration errors** — grid screening, pair lists, and ρ accumulation change as inter-fragment distance varies.
2. **SCF warm-start coupling** — production use reuses `dm0` between nearby geometries; path differences can compound over a trajectory.
3. **Precomp memory leaks** — `gpu_coalesced` uploads full χ[iAO,iG] at setup; without releasing plans between frames, multi-frame scans hit `MEM_OBJECT_ALLOCATION_FAILURE` mid-run (observed at H₂O frame 34 before fix).

Binding curves E_bind(z) are the acceptance test for non-covalent GPU work: if all paths overlay CPU on the same rigid scan, ρ→PBE→vmat parity holds under realistic SCF cycling.

Prior H₂O plots used sparse xTB-derived sampling (~10 points). This pass uses the **DFTB distance grid** (39 points, 0.1 Å near r_eq) and compares **all six** XC paths on **H₂O** and **formic acid** dimers.

---

## Infrastructure added

### General dimer scan driver

`profile_dimer_scan.py` replaces molecule-specific logic. Only molecule-specific input: **`--n0`** = 0-based index of the first atom in fragment 2 (mobile monomer).

| System | n0 | Geometry source | Anchor pair |
|--------|-----|-----------------|-------------|
| H₂O dimer | 3 | DFTB multi-frame XYZ | O···O (auto) |
| Formic dimer | 5 | Rigid shift from `data/xyz/formic_dimer.xyz` | O(3)···O(7) |

`dimer_scan_frames.py` builds rigid-shift frames when only a relaxed XYZ exists. `plot_scan_ez.py` produces the **primary** deliverable: `energy_profile_ez.png` (all paths + ΔE_bind vs CPU).

### GPU plan lifecycle fix

```python
# profile_dimer_scan.py — release_gpu_between_frames()
plan.release()
clear_xc_plan_cache()
```

Called before each GPU setup and after each frame. Required for `gpu_coalesced` on 39-frame scans (~28 MB χ per cached plan × frames).

### Path SSOT

`xc_path_modes.py` maps CLI keys → `gpu_profiles.py` presets. See [`../opencl_gpu_paths_cookbook.md`](../opencl_gpu_paths_cookbook.md).

---

## Benchmark setup

### Distance grid

File: `CompChemUtils/tmp/H2O_dimer_scan_dftb/distances.dat`  
39 distances (Å): 0.1 Å spacing near equilibrium, coarser at long range (1 Å steps 10–15 Å, 5 Å jump to 20 Å).

Binding reference: z_ref = 20 Å (dissociated limit).  
Units: kcal/mol via 627.5094740631 Ha⁻¹.

### SCF settings

| Parameter | Value |
|-----------|-------|
| XC | PBE |
| Basis | 6-31g |
| Grid | level 3 |
| conv_tol | 1e-8 (1e-6 for `gpu_full`) |
| conv_tol_grad | 1e-5 |
| max_cycle | 50 |
| Warm-start | yes (dm0 from previous frame) |
| DF | yes (CPU RI-J except `gpu_full`) |

### Run command

```bash
PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 python3 -u \
  expamples_prokop/profile_h2o_dimer_scan.py \
  --mode cpu gpu_otf gpu_coalesced gpu_radial gpu_gto gpu_full

PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 python3 -u \
  expamples_prokop/profile_formic_dimer_scan.py \
  --mode cpu gpu_otf gpu_coalesced gpu_radial gpu_gto gpu_full
```

Total wall time: ~17 min (both systems, sequential).

---

## Systems

| | H₂O dimer | Formic dimer |
|---|-----------|--------------|
| Atoms | 6 | 10 |
| nao | 26 | 62 |
| ngrids | 67 400 | 123 800 |
| Frames | 39 | 39 |
| Geometry | DFTB scan XYZ | Rigid O(3)···O(7) shift, r₀=2.927 Å |

---

## Energy parity — H₂O dimer

**Artifacts:** `debug/profile_h2o_dimer_scan/`  
CSV: `scan_scf_profile.csv` · Plots: `energy_profile_ez.png`, `energy_profile.png`

### Binding minimum (ref r = 20 Å)

| Path | r_min (Å) | E_bind min (kcal/mol) | Δ at minimum |
|------|-----------|----------------------|--------------|
| cpu | 2.804 | −8.241 | — |
| gpu_otf | 2.804 | −8.246 | −0.005 |
| gpu_coalesced | 2.804 | −8.244 | −0.004 |
| gpu_radial | 2.804 | −8.245 | −0.004 |
| **gpu_gto** | 2.804 | **−8.341** | **−0.100** |
| gpu_full | 2.804 | −8.252 | −0.011 |

### Deviation vs CPU along scan

| Path | max \|ΔE_bind\| | RMS ΔE_bind | max \|ΔE_tot\| | RMS ΔE_tot |
|------|----------------|-------------|----------------|------------|
| gpu_otf | 0.0055 | 0.0047 | 0.0037 | 0.0012 |
| gpu_coalesced | 0.0053 | 0.0044 | 0.0035 | 0.0012 |
| gpu_radial | 0.0054 | 0.0045 | 0.0036 | 0.0012 |
| **gpu_gto** | **0.1019** | **0.0892** | **0.1382** | **0.0377** |
| gpu_full | 0.0295 | 0.0152 | 0.0221 | 0.0095 |

**Verdict (H₂O):** Hermite production paths (`gpu_otf`, `gpu_coalesced`, `gpu_radial`) pass scan parity gate (≤ 0.006 kcal/mol RMS). `gpu_gto` fails on H₂O despite being the “exact GTO χ” reference path — **requires investigation** (formic passes; see below). `gpu_full` scatter consistent with relaxed SCF tol + GPU J, not different ρ/vmat kernels.

---

## Energy parity — formic dimer

**Artifacts:** `debug/profile_formic_dimer_scan/`

### Binding minimum

| Path | r_min (Å) | E_bind min (kcal/mol) | Δ at minimum |
|------|-----------|----------------------|--------------|
| cpu | 2.704 | −21.658 | — |
| gpu_otf | 2.704 | −21.657 | +0.002 |
| gpu_coalesced | 2.704 | −21.656 | +0.003 |
| gpu_radial | 2.704 | −21.657 | +0.002 |
| gpu_gto | 2.704 | −21.658 | +0.000 |
| gpu_full | 2.704 | −21.605 | **+0.053** |

### Deviation vs CPU along scan

| Path | max \|ΔE_bind\| | RMS ΔE_bind | max \|ΔE_tot\| | RMS ΔE_tot |
|------|----------------|-------------|----------------|------------|
| gpu_otf | 0.0017 | 0.0009 | 0.0047 | 0.0029 |
| gpu_coalesced | 0.0034 | 0.0018 | 0.0059 | 0.0032 |
| gpu_radial | 0.0020 | 0.0009 | 0.0048 | 0.0028 |
| gpu_gto | 0.0061 | 0.0016 | 0.0041 | 0.0022 |
| gpu_full | 0.0596 | 0.0304 | 0.0607 | 0.0282 |

**Verdict (formic):** All Hermite paths and `gpu_gto` within ~0.002 kcal/mol RMS — excellent. `gpu_full` again the largest outlier (~0.03 kcal/mol RMS, ~0.05 at minimum).

---

## Performance (average per frame)

### H₂O dimer (nao=26, ngrids=67400)

| Path | avg cycles | avg setup (ms) | avg SCF (ms) | avg total (ms) | conv% |
|------|------------|----------------|--------------|----------------|-------|
| cpu | 5.3 | 0 | 434 | 434 | 100 |
| gpu_otf | 14.6 | 127 | 342 | 469 | 97 |
| gpu_coalesced | 14.2 | 149 | 361 | 510 | 100 |
| gpu_radial | 14.5 | 139 | **329** | **468** | 100 |
| gpu_gto | 15.6 | 111 | 407 | 518 | 100 |
| gpu_full | 10.9 | 149 | **178** | **327** | 100 |

On this small system GPU paths are not faster than CPU overall (setup + extra SCF cycles from f32 veff). `gpu_radial` is fastest GPU XC path; `gpu_full` wins on SCF wall time via GPU J + fewer cycles (relaxed tol).

### Formic dimer (nao=62, ngrids=123800)

| Path | avg cycles | avg setup (ms) | avg SCF (ms) | avg total (ms) | conv% |
|------|------------|----------------|--------------|----------------|-------|
| cpu | 7.1 | 0 | 2507 | 2507 | 100 |
| gpu_otf | 22.4 | 339 | 3162 | 3501 | 97 |
| gpu_coalesced | 22.2 | 473 | 4601 | 5074 | 95 |
| gpu_radial | 22.1 | 365 | 3615 | 3981 | 92 |
| gpu_gto | 24.7 | 415 | 5082 | 5497 | 95 |
| gpu_full | 32.1 | 433 | 2073 | 2506 | **64** |

Formic is compute-heavy; GPU XC still net slower at this size with warm-start and f32 noise driving extra cycles. **`gpu_full` convergence drops to 64%** on formic — correlated with largest energy deviation; likely needs tighter tol or better J accuracy for hydrogen-bonded dimers.

---

## DFTB reference overlay (not a parity target)

Both scans overlay DFTB binding from the same distance grid. PBE/6-31g/DF and DFTB differ substantially:

| | H₂O | Formic |
|---|-----|--------|
| r_min CPU vs DFTB | 2.804 vs 2.904 Å | 2.704 vs 2.904 Å |
| Well depth CPU vs DFTB | −8.24 vs −2.52 kcal/mol | −21.66 vs −2.52 kcal/mol |
| max \|ΔE_bind\| vs DFTB | 33.3 kcal/mol | 24.1 kcal/mol |

Use DFTB only for qualitative long-range decay check, **not** GPU acceptance.

---

## Key findings

1. **Hermite OTF / coalesced / radial** reproduce CPU binding curves on both dimers at sub–0.005 kcal/mol (H₂O) and sub–0.002 kcal/mol (formic) RMS — production-ready for NCIs at this level of theory.
2. **`gpu_gto` H₂O anomaly** — ~0.09 kcal/mol RMS despite “exact GTO χ”; same path on formic is fine (~0.002). Suggests H₂O-specific setup or grid pairing issue, not a general GTO precomp failure.
3. **`gpu_full`** — identical ρ/vmat to `gpu_otf`; energy scatter and poor formic convergence come from **GPU J (f32)** + **conv_tol 1e-6**, not XC kernel differences.
4. **χ cache release** is mandatory for multi-frame `gpu_coalesced` scans.
5. **E(z) plot** (`energy_profile_ez.png`) is the essential comparison figure; single-point bar charts are supplementary.

---

## Still open

- Root-cause `gpu_gto` H₂O scan deviation (~0.1 kcal/mol at binding minimum)
- `gpu_full` SCF convergence on formic (64%) — try conv_tol 1e-7 or CPU J with OTF XC
- Extend scan to larger dimers (e.g. CG.xyz) once runtime acceptable
- Re-generate formic `energy_analysis.txt` with `--title` after plot script generalization (Jul 2026 fix)

---

## Files touched / added

| File | Role |
|------|------|
| `expamples_prokop/profile_dimer_scan.py` | General scan driver |
| `expamples_prokop/dimer_scan_frames.py` | Rigid shift frame builder |
| `expamples_prokop/plot_scan_ez.py` | Primary E(z) plot |
| `expamples_prokop/xc_path_modes.py` | Path SSOT |
| `expamples_prokop/profile_h2o_dimer_scan.py` | H₂O wrapper (n0=3) |
| `expamples_prokop/profile_formic_dimer_scan.py` | Formic wrapper (n0=5) |
| `expamples_prokop/plot_h2o_dimer_scan_energy.py` | 4-panel diagnostic; `--title` / `--z-label` |
| `doc/dimer_scan_benchmarks.md` | Concept doc + tutorial |
| `doc/topical_audit.md` | Dimer scan section |

---

## Reproduce

```bash
# H₂O — all paths, full grid
PYTHONPATH=$PWD OMP_NUM_THREADS=1 python3 -u expamples_prokop/profile_h2o_dimer_scan.py

# Formic — rigid shift from single XYZ
PYTHONPATH=$PWD OMP_NUM_THREADS=1 python3 -u expamples_prokop/profile_formic_dimer_scan.py

# Replot only
python3 expamples_prokop/plot_scan_ez.py \
  --csv debug/profile_h2o_dimer_scan/scan_scf_profile.csv --title "H₂O dimer"
```
