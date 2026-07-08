# pyscf/OpenCL

OpenCL GPU backend for DFT grid XC (ρ, PBE vxc, vmat) and density-fitting J/K. Integrated via `mf.backend=2` and `mf.setup_gpu()` in `pyscf/dft/rks.py`.

- **xc_grid.py** — `XCGridPlan` orchestrator: OTF and precomputed ρ/vmat paths, GPU PBE, stage timing (`last_timing`), hybrid `vmat_mode`
- **gpu_timing.py** — accurate GPU profiling: `queue.finish()` wall times + `clGetEventProfilingInfo` kernel events
- **gpu_profiles.py** — named production presets (`production_otf`, `production_otf_radial_vmat`, …); apply via `apply_gpu_profile(mf, name)`
- **kernels.cl** — tiled/pair ρ and vmat kernels, radial precomp gather, quintic Hermite, PBE wv, reductions
- **pbe.cl** — generated PBE vxc (from libxc via `generate_pbe_cl.py`)
- **ao_hermite.py** — GPU Hermite AO evaluator; `build_radial_on_grid_gpu` for hybrid/radial vmat
- **hermite_spline.py** / **radial_hermite.py** — host radial table build; cubic/quintic, memory-equivalent `du`
- **tile_config.py** — compile-time `NPTILE`, `NATILE`, `WGS_VMAT`; env overrides
- **df_jk.py** — RI density-fitting J/K on GPU (separate from XC)
- **grid_screen.py** — atom-tile screening from Hermite radial tails
- **buffers.py** — shared OpenCL buffer helpers
- **__init__.py** — device init, `PROFILING_ENABLE` queue, kernel program cache

Cookbook: `/home/prokop/git/pyscf/doc/opencl_gpu_paths_cookbook.md` · Benchmarks: `/home/prokop/git/pyscf/doc/GPU_benchmark.md`
