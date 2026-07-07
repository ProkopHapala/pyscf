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
- Run repo code with: `PYTHONPATH=/home/prokophapala/git/pyscf python3 script.py`
- C extensions (`.so`) load from pip install via fallback in `pyscf/lib/misc.py:load_library`; Python code from repo
- Control threads: `OMP_NUM_THREADS=1` (or 4). Or in code: `lib.num_threads(n)`
- Verify which pyscf: `python3 -c "import pyscf; print(pyscf.__file__)"`

## Our examples and docs

- Custom examples/tests go in `expamples_prokop/` (not `examples/` which is upstream)
- Notes and profiling results go in `doc/`
- **GPU path cookbook:** `doc/opencl_gpu_paths_cookbook.md` · profiles `pyscf/OpenCL/gpu_profiles.py`
- Profiling script: `expamples_prokop/profile_dft.py`
- **Dimer scan benchmarks:** `doc/dimer_scan_benchmarks.md` · driver `expamples_prokop/profile_dimer_scan.py` (`--n0` = first atom of fragment 2)

## DFT execution path (for profiling/debugging)

- Entry: `mf.kernel()` → `pyscf/scf/hf.py:kernel()` (SCF loop at line 170)
- DFT-specific: `pyscf/dft/rks.py:get_veff()` (line 37) → calls `NumInt.nr_rks()` + `get_j()`
- XC integration: `pyscf/dft/numint.py:nr_rks()` (line 1074) → `block_loop()` (line 2887) → `eval_ao()` → `eval_xc_eff()`
- Grid build: `pyscf/dft/gen_grid.py:Grids.build()` (line 647)
- Built-in timing: `logger.timer(rec, msg, cpu0, wall0)` in `pyscf/lib/logger.py:167`. Enable with `mol.verbose = 5`
- cProfile: `python3 -m cProfile -s tottime -o /tmp/prof.out script.py`

## Repo navigation

- SCF core: `pyscf/scf/hf.py` (2563 lines, `kernel`, `get_jk`, `get_veff`, `SCF` class)
- DFT: `pyscf/dft/rks.py` (RKS class, `get_veff`), `pyscf/dft/numint.py` (XC integration), `pyscf/dft/gen_grid.py` (grids)
- Examples: `examples/scf/`, `examples/dft/` (upstream, don't modify)
- Tests: `pyscf/scf/test/`, `pyscf/dft/test/`


