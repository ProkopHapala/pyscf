# doc

Notes, architecture reports, and agent guidance for PySCF OpenCL DFT GPU work. **Start here:** `/home/prokophapala/git/pyscf/doc/opencl_gpu_paths_cookbook.md` (path knobs + profiles) and `/home/prokophapala/git/pyscf/doc/opencl_xc_architecture.md` (memory layout). Cross-topic status map: `/home/prokophapala/git/pyscf/doc/topical_audit.md`. Dimer scan benchmarks (E(z) CPU vs GPU paths): `/home/prokophapala/git/pyscf/doc/dimer_scan_benchmarks.md`.

## OpenCL XC / DFT — architecture and execution

- **opencl_gpu_paths_cookbook.md** — SCF integration map, valid backend combinations, named profiles (`gpu_profiles.py`), setup vs per-cycle work
- **opencl_xc_architecture.md** — central quantities, array layouts (DM, χ, ρ, wv, vmat), memory budgets, path comparison (OTF vs precomp)
- **OpenCL_XC_execution.md** — one SCF cycle: three distinct ops (eval_ao, ρ projection, vmat), what runs once vs every `get_veff`
- **opencl-xc-developer-guide.md** — living developer guide: hot path checklist, hoisting rules, debugging workflow
- **opencl-kernel-cookbook.md** — kernel design rules (gather vs scatter, rho vs vmat geometry, local memory, atomics)
- **opencl_rho_precomp_layout.md** — ρ projection kernel memory layout and coalesced gather design (precomp GTO path)
- **rho_vmat_vxc_GPU_optimization.report.md** — master optimization report (Parts 1–10): vmat tiling, pair kernels, parity audits, benzene/pentacene/PTCDA benchmarks, SCF profiling
- **quintic_hermite_spline.md** — cubic vs quintic Hermite radial spline study, accuracy vs memory trade-offs
- **dft_profiling_results.md** — CPU DFT baseline timings (`profile_dft.py`), grid-level and thread sweeps
- **dimer_scan_benchmarks.md** — inter-fragment distance scans: `--n0` fragment split, all XC paths, E(z) parity results (H₂O, formic)

## OpenCL XC — chronological reports

- **opencl-xc-reports/** — one file per optimization pass (not rewritten); index in `opencl-xc-reports/README.md`
- **opencl-xc-reports/2026-07-dimer-scan-xc-paths.md** — dimer E(z) scan parity: H₂O + formic, all GPU XC paths vs CPU
- **opencl-xc-reports/2026-03-rho-itile-host-setup.md** — ρ `iTile` inner loop, GPU-final ρ[g], pre-SCF setup timing

## Research notes (not OpenCL-specific)

- **ToOpenCL.chat.md** — early GPU port analysis: what to offload (grid XC vs DF J/K), backend library map (libcint/libxc)
- **speedup_dft_small.chat.md** — small-molecule DFT bottleneck discussion, low-rank SCF motivation
- **low_rank_perturbation.md** — multigrid / coarse-subspace ideas for electronic structure (research notes)

## Agent skills and protocols (`AGENTS/`)

- **AGENTS/skills/** — reusable agent workflows: `doc-audit`, `port-to-opencl`, `gpu-debug`, `gpu-optimize`, `numerical-parity`, `running-tests`, `python-perf`, …
- **AGENTS/protocols/** — domain and general protocols (parity checking, performance optimization, quantum mechanics context)
- **AGENTS/workflows/** — pre/post inventory checklists for agent tasks
- **AGENTS/agentic_debugging_principles.md** — cross-cutting debugging principles for scientific code
