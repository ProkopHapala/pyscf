# expamples_prokop

Custom benchmarks and parity checks for the OpenCL XC/DF GPU path — not upstream `examples/`. Run from repo root: `PYTHONPATH=/home/prokop/git/pyscf python3 expamples_prokop/<script>.py`. Dimer scan workflow: `/home/prokop/git/pyscf/doc/dimer_scan_benchmarks.md`.

## XC pipeline — end-to-end and parity

- **xc_path_modes.py** — SSOT mapping from scan mode keys (`cpu`, `gpu_otf`, …) to GPU profiles, kernel names, plot labels; shared by all profiling scripts
- **test_opencl_xc_full_gpu_parity.py** — full-GPU ρ→PBE→vmat audit vs CPU libxc on benzene
- **test_opencl_xc_e2e_mols.py** — multi-molecule E2E bench (benzene, pentacene, PTCDA); stage timing
- **test_opencl_xc_scf.py** — all execution paths with per-stage timing in one SCF
- **test_opencl_xc_onthefly.py** — Hermite OTF harness; primary tile/path tuning entry point
- **test_opencl_xc_onthefly_scaling.py** — OTF scaling vs grid size / molecule size
- **test_opencl_xc_rho_precomp.py** — ρ projection only, precomp GTO path vs CPU
- **test_opencl_xc_vmat_precomp.py** — vmat only (given ρ/wv), precomp path vs CPU
- **test_opencl_xc_cpu_threads.py** — CPU libxc thread scaling baseline

## Hermite AO — setup, accuracy, radial study

- **hermite_radial_study.py** — cubic/quintic Hermite accuracy sweeps (β, grid type); plots to `debug/`
- **test_opencl_xc_ao_hermite_setup.py** — GPU Hermite AO setup smoke test
- **test_opencl_xc_hermite_ao.py** — Hermite AO values on grid vs reference
- **test_opencl_hermite_ao.py** — `OpenCLAOHermiteEvaluator` vs exact Cartesian GTO
- **test_radial_hermite_ao.py** — radial Hermite table reconstruction vs Python exact eval
- **plot_hermite_cubic_quintic.py** — wrapper for hermite_radial_study plot commands
- **plot_carbon_radial_map_b.py** — radial map-β visualization for carbon shells
- **plot_ao_deriv_detail.py** — AO derivative error heatmaps / 1D scans
- **plot_ao_deriv_debug.py** — worst-case AO derivative debugging plots

## Dimer scan — E(z) CPU vs GPU path comparison

- **profile_dimer_scan.py** — general inter-fragment distance scan; **`--n0`** = first atom of fragment 2; dm warm-start; CSV + `energy_profile_ez.png`
- **dimer_scan_frames.py** — rigid shift frame builder: relaxed XYZ + distance grid → PySCF atom strings
- **plot_scan_ez.py** — essential two-panel E_bind(z) plot (all XC paths + Δ vs CPU)
- **plot_h2o_dimer_scan_energy.py** — 4-panel diagnostic (binding deviation, optional DFTB ref)
- **profile_h2o_dimer_scan.py** — H₂O wrapper (`n0=3`, DFTB scan xyz + distance grid)
- **profile_formic_dimer_scan.py** — formic wrapper (`n0=5`, rigid shift from `data/xyz/formic_dimer.xyz`)
- **profile_xc_paths_single.py** — single-geometry ΔE bar chart + timing (not a scan; use profile_dimer_scan for E(z))

## SCF profiling and tuning

- **profile_xc_stages_benzene.py** — per-stage XC timing table (wall + CL events); all paths including split-K; feeds `doc/GPU_benchmark.md`
- **profile_amdahl_budget.py** — setup vs cycle vs full-job budget; `--df-storage incore|outcore|auto` (default incore); see `doc/df_storage_and_benchmark_hygiene.md`
- **profile_scf_cycle.py** — full SCF cycle breakdown (J + XC per iteration)
- **profile_gpu_scf.py** — full SCF convergence profile, per-cycle vxc accuracy
- **profile_gpu_amdahl_strict.py** — non-overlapping same-input CPU/GPU SCF-cycle Amdahl profile; validates manual `veff` against `mf.get_veff`
- **plot_gpu_scf_amdahl.py** — plot prepared end-to-end and repeated-cycle SCF timing data from the acceptance run
- **profile_dft.py** — CPU DFT bottleneck profiler (cProfile + PySCF timers)
- **sweep_splitk_tiles.py** — **recommended** tile sweep for split-K: `--neighbor` 1-step lattice walk from `--seed`
- **sweep_opencl_tiles.py** — legacy brute-force `TileConfig` grid (NPTILE, NATILE, WGS_VMAT, …)
- **test_opencl.py** — early integration smoke tests (xc_grid, df_jk, `mf.backend=2`)
