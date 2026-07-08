# smallDFT

CPU grid-parallel RKS XC for small molecules (`nao ≲ 200`, `ngrids ~ 30k–150k`): OpenMP ρ/vmat in `libsmalldft`, libcint AO layout, drop-in `nr_rks`. See `/home/prokop/git/pyscf/doc/smallDFT_cpu_path.md`.

## When to use

- Small molecules with many grid points; **multiple SCF cycles per geometry** (AO cache pays).
- LDA + GGA (PBE tested). Not MGGA / UKS yet.
- Set `OPENBLAS_NUM_THREADS=1`; grid parallelism via `lib.num_threads(N)`.

## Quickstart

```bash
# Build C kernels (required for production path)
pyscf/lib/smalldft/build.sh

# Parity + ρ/vmat scaling
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokop/git/pyscf \
  python3 expamples_prokop/test_small_dft.py

# Full SCF cycle: init vs per-iteration (Amdahl)
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokop/git/pyscf \
  python3 expamples_prokop/profile_scf_cycle.py --mol benzene --path ref smallDFT_ws --threads 8
```

## Modules

- **nr_rks.py** — drop-in replacement for `numint.nr_rks`; dispatches to C when `libsmalldft` loaded
- **rho.py** / **vmat.py** — ρ and vmat drivers; `use_c=True` → ctypes → OpenMP kernels
- **_ctypes.py** — `libsmalldft` load + `SMALL_*` bindings
- **workspace.py** — `GridWorkspace`: preallocated ρ/vmat buffers; `eval_ao()` sets χ from `eval_ao_native`
- **layout.py** — keep libcint F-contiguous `(ngrids, nao)`; `eval_ao_native` entry point
- **patch.py** — `enable()` / `disable()` monkey-patch on `NumInt.nr_rks`
- **profile.py** — `profile_xc_bottleneck`, `profile_compare`, timing breakdowns
- **parallel.py** — legacy Python `ThreadPoolExecutor` tiles (fallback only; do not extend)

## Related docs

- Implementation: `doc/smallDFT_cpu_path.md`
- Benchmarks: `doc/CPU_benchmark.md`
- Lessons learned: `doc/CPU_optimixation_experience.md`
