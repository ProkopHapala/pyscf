# pyscf/OpenCL

OpenCL GPU backend for DFT grid XC (ρ, PBE vxc, vmat) and density-fitting J/K. Integrated via `mf.backend=2` and `mf.setup_gpu()` in `pyscf/dft/rks.py`.

- **xc_grid.py** — `XCGridPlan` orchestrator: OTF and precomputed ρ/vmat paths, GPU PBE, stage timing (`last_timing`), hybrid `vmat_mode`
- **gpu_timing.py** — accurate GPU profiling: `queue.finish()` wall times + `clGetEventProfilingInfo` kernel events
- **gpu_profiles.py** — named production presets plus `prepare_df_for_scf`: with profile setup enabled, static grids/DF tensors/GPU DF buffers are prepared before `mf.kernel()`
- **kernels.cl** — tiled/pair ρ and vmat kernels, radial precomp gather, **split-K vmat + reduce**, quintic Hermite, PBE wv, reductions
- **pbe.cl** — generated PBE vxc (from libxc via `generate_pbe_cl.py`)
- **ao_hermite.py** — GPU Hermite AO evaluator; `build_radial_on_grid_gpu` for hybrid/radial vmat
- **hermite_spline.py** / **radial_hermite.py** — host radial table build; cubic/quintic, memory-equivalent `du`
- **tile_config.py** — compile-time `NPTILE`, `NATILE`, `WGS_VMAT`; env overrides; **split-K profile may recompile with different WGS** (`_ensure_splitk_tile_config`)
- **df_jk.py** — RI density-fitting J/K on GPU (separate from XC)
- **grid_screen.py** — atom-tile screening from Hermite radial tails
- **buffers.py** — shared OpenCL buffer helpers
- **__init__.py** — device init, `PROFILING_ENABLE` queue, kernel program cache

Cookbook: `/home/prokop/git/pyscf/doc/opencl_gpu_paths_cookbook.md`  
Benchmarks: `/home/prokop/git/pyscf/doc/GPU_benchmark.md` · Full-cycle Amdahl: `/home/prokop/git/pyscf/doc/acceptance_2026-07-11.md` · Experience: `/home/prokop/git/pyscf/doc/GPU_optimixation_experience.md`

**Profiling:** queue must use `PROFILING_ENABLE`; stage times in `plan.last_timing` when `profile=True`. See `gpu_timing.py`.

**Tile tuning:** `expamples_prokop/sweep_splitk_tiles.py --neighbor` (1-neighborhood coordinate descent); legacy `sweep_opencl_tiles.py` for brute-force exploration.
