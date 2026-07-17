# libsmalldft

OpenMP grid-tile kernels for LDA/GGA ρ and vmat on PySCF F-contiguous χ. Built as `libsmalldft.so` beside other PySCF libs. See `/home/prokop/git/pyscf/doc/smallDFT_cpu_path.md`.

## Build

```bash
# Quick standalone gcc build (no full PySCF cmake)
pyscf/lib/smalldft/build.sh

# Optional: tune grid tile size at compile time (default TILE=512)
SMALLDFT_TILE=1024 pyscf/lib/smalldft/build.sh

# Optional: local SIMD tuning (not portable)
SMALLDFT_NATIVE=1 pyscf/lib/smalldft/build.sh
```

Rebuild after any change to `small_grid.c`. Verify load:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokop/git/pyscf \
  python3 -c "from pyscf.smallDFT import has_c_lib; print(has_c_lib())"
```

## Design

- **Parallel axis:** grid tiles `[g0, g1)` via OpenMP; `TILE=512` default (benchmark on target machine — 1024/2048 can be slower).
- **Layout:** χ F-contiguous `(ngrids, nao)`; inner loops **stride-1 over `g`**, AO index outer.
- **ρ:** disjoint writes per thread; GGA uses one `DM@χ` GEMM per tile + stride-1 accumulation.
- **vmat:** F-order tile buffers; `dgemm("T","N")`; private `V_t` per thread + `omp critical` reduce; GGA hermi via temp buffer.
- **BLAS:** caller must set `OPENBLAS_NUM_THREADS=1`; grid OMP via `lib.num_threads(N)`.

## Files

- **small_grid.c** — `SMALL_rho_*`, `SMALL_vmat_*` (full-χ cache path)
- **stream_grid.c** — `SMALL_stream_*` (block χ; no full-grid materialization)
- **small_grid.h** — C API declarations
- **CMakeLists.txt** — cmake target linked to `np_helper` + BLAS + OpenMP
- **build.sh** — quick standalone gcc build without full PySCF cmake

Python switch: `prepare_smalldft_for_scf(..., ao_mode='cache'|'stream')`.

## Related docs

- `doc/CPU_benchmark.md` — scaling tables and TILE sweep
- `doc/CPU_optimixation_experience.md` — strategies and false premises
