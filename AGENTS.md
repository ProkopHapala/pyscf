We develop rigorous scientific software where debuggability, physical consistency, numerical correctness and stability, as well as performance and simplicity are paramount. Follow these principles:

## Core Principles

- **KISS** (Keep It Simple), Simplest solution that works. One-liner > ten-liner.
- **AHA** (Avoid Hasty Abstractions), avoid boilerplate
- **YAGNI** : **Surgical Edits** — Touch only what's needed. No unrelated cleanup. Comment out, don't delete. Ask if ambiguous.
- **DRY** : Inventory existing code before writing new. Generalize rather than duplicate. See `reusable-architecture/SKILL.md`.
- **SoC** (Separation of Concerns), separate module for Compute, plotting, Backend, CLI, GUI. Thin test scripts call general workhorse function from shared modules.
- **SSOT** : Authoritative single source of truth must be defined to avoid ambiguity and confusion
- **TDD** : Define verification before coding. Parity checks vs reference/analytical/physical invariants. Run tests after every change. See `numerical-parity/SKILL.md`, `forcefield-validation/SKILL.md`.
- **Testing** : Tests in `tests/`, reference data in `tests/ref_data/` (git-tracked), debug output in `debug/<script_name>/` (gitignored). See `running-tests/SKILL.md`.
- **Fail Fast** : No silent fallbacks (try-catch). Crashes with stack traces > masked bugs. Look for root cause, not symptoms. See `visual-debugging/SKILL.md`, `gpu-debugging/SKILL.md`.
- **Performance** : preallocate, minimize python orchestration; to C/C++/OpenCL kernels. Data-oriented-desing: Flat arrays, cache-aware, usel local memory in OpenCL. See `port-to-opencl/SKILL.md`, `python-performance/SKILL.md`.
- Compact code, unlimited line lengh (function call must be one line).  Short names for math symbols (`E_tot`, `T_ij`).
- **Never pipe long job output** (grep/tail/head/pipe) — user sees empty screen and kills it. Always show full stdout.

## Running repo version (not pip-installed)

- Pip-installed pyscf at `~/.local/lib/python3.10/site-packages/pyscf/` (binary `.so` libs)
- Run repo code with: `PYTHONPATH=/home/prokop/git/pyscf python3 script.py`
- C extensions (`.so`) load from pip install via fallback in `pyscf/lib/misc.py:load_library`; Python code from repo
- Control threads: `OMP_NUM_THREADS=1` (or 4). Or in code: `lib.num_threads(n)`
- Verify which pyscf: `python3 -c "import pyscf; print(pyscf.__file__)"`

## Our examples and docs

- Custom examples/tests go in `expamples_prokop/` (not `examples/` which is upstream)
- Notes and profiling results go in `doc/`
- **GPU path cookbook:** `doc/opencl_gpu_paths_cookbook.md` · profiles `pyscf/OpenCL/gpu_profiles.py`
- **GPU benchmarks / lessons:** `doc/GPU_benchmark.md` · `doc/GPU_optimixation_experience.md`
- **Best per-cycle GPU XC (small mol, benzene-tuned):** `production_otf_radial_vmat_splitk` — OTF ρ + split-K radial vmat (~12 ms gpu CL vs ~21 ms hybrid); stage driver `expamples_prokop/profile_xc_stages_benzene.py`
- **CPU smallDFT path:** `doc/smallDFT_cpu_path.md` · experience `doc/CPU_optimixation_experience.md` · benchmarks `doc/CPU_benchmark.md`
- Profiling scripts: `expamples_prokop/profile_dft.py` (CPU baseline), `expamples_prokop/profile_scf_cycle.py` (one SCF cycle, init vs cycle), `expamples_prokop/profile_xc_stages_benzene.py` (XC stage wall+CL), `expamples_prokop/sweep_splitk_tiles.py` (`--neighbor` tile tune), `expamples_prokop/test_small_dft.py` (parity + ρ/vmat scaling)
- **Dimer scan benchmarks:** `doc/dimer_scan_benchmarks.md` · driver `expamples_prokop/profile_dimer_scan.py` (`--n0` = first atom of fragment 2)

## CPU smallDFT (grid-parallel XC)

- Entry: `pyscf.smallDFT.nr_rks` (drop-in for `numint.nr_rks`) or `pyscf.smallDFT.patch.enable()`
- C kernels: `pyscf/lib/smalldft/small_grid.c` — build with `pyscf/lib/smalldft/build.sh`
- AO cache: `GridWorkspace.eval_ao()` once per geometry; reuse χ across SCF cycles
- Threading: `lib.num_threads(N)` for grid OpenMP + libcint; **`OPENBLAS_NUM_THREADS=1`** (avoid nested OMP + threaded BLAS)
- Layout: keep libcint F-contiguous `(ngrids, nao)`; inner C loops stride-1 over grid `g`
- Profiling: use real `mf.kernel()` in `profile_scf_cycle.py` (not `--manual`); patch **`RHF.get_jk`** for J timing (RKS inherits RHF, not `SCF.get_jk` alone)

## DFT execution path (for profiling/debugging)

- Entry: `mf.kernel()` → `pyscf/scf/hf.py:kernel()` (SCF loop at line 170)
- DFT-specific: `pyscf/dft/rks.py:get_veff()` (line 37) → calls `NumInt.nr_rks()` + `get_j()`
- XC integration: `pyscf/dft/numint.py:nr_rks()` (line 1074) → `block_loop()` (line 2887) → `eval_ao()` → `eval_xc_eff()`; **smallDFT** replaces ρ/vmat with OpenMP grid tiles when `libsmalldft` is built
- Grid build: `pyscf/dft/gen_grid.py:Grids.build()` (line 647)
- Built-in timing: `logger.timer(rec, msg, cpu0, wall0)` in `pyscf/lib/logger.py:167`. Enable with `mol.verbose = 5`
- GPU XC stage timing: `plan.nr_rks_hermite_onthefly(dm, profile=True)` → `plan.last_timing` (`gpu_timing.py`; requires `PROFILING_ENABLE` queue)
- cProfile: `python3 -m cProfile -s tottime -o /tmp/prof.out script.py`

## Repo navigation

- SCF core: `pyscf/scf/hf.py` (2563 lines, `kernel`, `get_jk`, `get_veff`, `SCF` class)
- DFT: `pyscf/dft/rks.py` (RKS class, `get_veff`), `pyscf/dft/numint.py` (XC integration), `pyscf/dft/gen_grid.py` (grids)
- Examples: `examples/scf/`, `examples/dft/` (upstream, don't modify)
- Tests: `pyscf/scf/test/`, `pyscf/dft/test/`


