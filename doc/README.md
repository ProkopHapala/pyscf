# doc

Notes, architecture reports, and agent guidance for PySCF OpenCL DFT GPU work and CPU smallDFT. **Start here:** `/home/prokop/git/pyscf/doc/opencl_gpu_paths_cookbook.md` (GPU path knobs) and `/home/prokop/git/pyscf/doc/smallDFT_cpu_path.md` (CPU grid-parallel XC). Cross-topic status: `/home/prokop/git/pyscf/doc/topical_audit.md`.

## CPU smallDFT — grid-parallel XC

- **smallDFT_cpu_path.md** — implementation guide: layout, C kernels, API, parity, design decisions
- **CPU_benchmark.md** — benzene PBE scaling tables (original vs C/OpenMP), bottleneck waterfall, **one-SCF-cycle Amdahl profile**
- **CPU_optimixation_experience.md** — strategies, caveats, disproved assumptions, generalization
- **CPU_small_DFT.chat.md** — original analysis/plan notes (pre-implementation); **test machine** specs (Ryzen 5800X, 32 GiB RAM — same host as `GPU_benchmark.md`)

## OpenCL XC / DFT — architecture and execution

- **opencl_gpu_paths_cookbook.md** — SCF integration map, valid backend combinations, named profiles (`gpu_profiles.py`), and static pre-SCF XC/DF preparation
- **opencl_xc_architecture.md** — central quantities, array layouts (DM, χ, ρ, wv, vmat), memory budgets, path comparison (OTF vs precomp)
- **OpenCL_XC_execution.md** — one SCF cycle: three distinct ops (eval_ao, ρ projection, vmat), what runs once vs every `get_veff`
- **opencl-xc-developer-guide.md** — living developer guide: hot path checklist, hoisting rules, debugging workflow
- **opencl-kernel-cookbook.md** — kernel design rules (gather vs scatter, rho vs vmat geometry, local memory, atomics)
- **opencl_rho_precomp_layout.md** — ρ projection kernel memory layout and coalesced gather design (precomp GTO path)
- **rho_vmat_vxc_GPU_optimization.report.md** — master optimization report (Parts 1–10): vmat tiling, pair kernels, parity audits, benzene/pentacene/PTCDA benchmarks, SCF profiling
- **quintic_hermite_spline.md** — cubic vs quintic Hermite radial spline study, accuracy vs memory trade-offs
- **GPU_benchmark.md** — benzene isolated-XC stage timings (wall vs CL events); hybrid OTF ρ + radial vmat; split-K; tile sweep report
- **GPU_optimixation_experience.md** — lessons learned: strategies, caveats, disproved assumptions, generalization from GPU XC optimization
- **dft_profiling_results.md** — CPU DFT baseline timings (`profile_dft.py`), grid-level and thread sweeps
- **dimer_scan_benchmarks.md** — inter-fragment distance scans: `--n0` fragment split, all XC paths, E(z) parity results (H₂O, formic)
- **acceptance_2026-07-11.md** — CPU/GPU rigor and performance acceptance report; RTX 3090 large-molecule strict Amdahl profile and static-preparation results

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
