---
name: cpu-perf
description: Optimizing CPU compiled code (C/C++/Fortran) — cache locality, SIMD, data layout, OpenMP, profiling, roofline reasoning
trigger:
  glob:
    - "**/*.cpp"
    - "**/*.h"
    - "**/*.c"
    - "**/*.f"
    - "**/*.f90"
    - "**/*.rs"
    - "**/lib/**/*.c"
---

## Core Principle

Modern CPUs are out-of-order machines with deep caches and wide SIMD. **Identify the bottleneck, apply one structural fix, measure again.** Do not cargo-cult micro-optimizations from fixed hardware numbers.

## Optimization Workflow (MUST)

1. Preserve a correct reference implementation.
2. Measure end-to-end; use `perf stat` for cycles, instructions, IPC, branch-misses, cache-misses.
3. Isolate the hot loop or function.
4. Classify: memory bandwidth, cache/TLB misses, dependency chains, branch mispredicts, insufficient parallelism, allocation.
5. Estimate arithmetic intensity (useful FLOPs / required bytes) — roofline reasoning.
6. Form one hypothesis; change one structural issue.
7. Validate numerics; benchmark representative sizes.
8. Inspect compiler vectorization reports (`-fopt-info-vec` GCC, `-Rpass=loop-vectorize` Clang).
9. Keep the change only if improvement is meaningful.

## Memory Hierarchy (illustrative — query target hardware)

Typical x86_64 (sizes vary by CPU):

| Level | Typical size | Typical latency | Notes |
|-------|-------------|-----------------|-------|
| L1 | 32 KB/core | ~1–4 cycles | split I/D |
| L2 | 256 KB–1 MB/core | ~4–12 cycles | |
| L3 | 8–64 MB shared | ~12–40 cycles | |
| DRAM | GBs | ~100–300 ns | bandwidth often limits before latency |

- **64-byte cache lines** are common on x86 but not universal — treat as an example.
- **Spatial locality**: traverse contiguous memory sequentially.
- **Temporal locality**: reuse data while still in cache.
- **Cache blocking/tiling**: for O(N²)/O(N³) kernels, tile so working sets fit in L1/L2.

## Rules by Strength

### MUST

- Profile before optimizing; keep a reference result.
- No heap allocation (`malloc`/`new`) in hot paths — preallocate once.
- Avoid false sharing: pad per-thread accumulators to cache-line boundary when threads write adjacent scalars.
- Verify aliasing before `#pragma omp simd` — it asserts no loop-carried dependencies prevent SIMD.
- Reset per-iteration reduction state in persistent OpenMP regions.

### PREFER

- Flat contiguous arrays over pointer-chasing (`float*` + index arithmetic beats `vector<vector<float>>`).
- `__restrict__` / `restrict` pointers when inputs do not alias outputs.
- Hoist loop-invariant computations out of inner loops.
- Fuse loops with the same iteration space when combined body still vectorizes.
- First-touch NUMA policy: initialize data on the thread that will use it.
- OpenMP: one `#pragma omp parallel` region around outer iterative loop, `#pragma omp for` inside — avoid fork/join per iteration.

### CONSIDER / BENCHMARK

- SoA vs AoS vs AoSoA — choose by access pattern (see Data Layout).
- Branchless arithmetic vs branches — modern compilers use cmov/masks; measure both.
- Manual SIMD intrinsics — only when auto-vectorization fails despite good layout.
- `alloca()` / VLAs — stack overflow risk, inhibits optimization; prefer fixed small arrays or thread-local scratch buffers.
- Software prefetch — only after profiling shows cache misses dominate.
- Loop interchange — when it improves stride-1 access in the inner loop.

## Data Layout

Choose layout by how the hot loop accesses data:

| Pattern | PREFER |
|---------|--------|
| Neighboring threads read same field from neighboring objects | **SoA** — `float *x, *y, *z` |
| One thread consumes nearly all fields of one object | **AoS** or packed struct |
| SIMD blocks of fixed width | **AoSoA** — `float x[BLOCK], y[BLOCK], z[BLOCK]` |
| Large records with few hot fields | **Hot/cold split** — separate arrays |

```c
// Hot every timestep
float *pos_x, *pos_y, *pos_z;
float *force_x, *force_y, *force_z;

// Cold / rarely accessed
float *parameters;
int   *topology;
```

- Align large array starts to SIMD width (16/32 bytes) or cache line when it prevents false sharing — do not pad every object to 64 bytes unless measured benefit.
- Keep hot traversal contiguous; avoid frequent cache-line crossings for small records.

## SIMD / Vectorization

- **PREFER auto-vectorization** (`-O3`, `-march=native`) over manual intrinsics — focus on layout and loop shape.
- Vectorization-friendly: unit stride, no function calls in inner loop (or `always_inline`), no aliasing (`restrict`), predictable branches.
- `#pragma omp simd` — correctness assertion, not a harmless hint; verify dependencies first.
- Manual SSE/AVX intrinsics as last resort after checking `-fopt-info-vec` / `-Rpass=loop-vectorize`.

```c
void add(float *restrict a, float *restrict b, float *restrict c, int n)
{
    #pragma omp simd
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}
```

## Loop Optimization

- **Hoist** invariants out of loops.
- **Fuse** loops with same trip count when register pressure stays manageable.
- **Split** loops when combined body is too large (register pressure) or branches diverge badly.
- **Loop interchange** — put stride-1 dimension innermost.
- **Scalar cleanup** — vectorized interior + simple scalar boundary loop for non-multiple sizes.

### Branches in inner loops

> First make branches predictable and vectorizable. Replace with masks only when measured branch cost exceeds extra work.

```c
// Measure before assuming branchless is faster
for (int i = 0; i < n; i++) {
    if (mask[i])
        c[i] = a[i] + b[i];
    else
        c[i] = a[i] - b[i];
}
```

## Cache-Blocked Matrix Multiply (guarded template)

```c
#define BS 32
for (int ii = 0; ii < M; ii += BS)
  for (int jj = 0; jj < N; jj += BS)
    for (int kk = 0; kk < K; kk += BS)
      for (int i = ii; i < ii + BS && i < M; i++)
        for (int j = jj; j < jj + BS && j < N; j++) {
          float acc = C[i * N + j];
          for (int k = kk; k < kk + BS && k < K; k++)
            acc += A[i * K + k] * B[k * N + j];
          C[i * N + j] = acc;
        }
```

Choose `BS` so relevant A, B, C tile blocks fit in cache — not just A. Real microkernels are more complex; this template is correctness-safe, not peak-performance.

## Memory Allocation

- **Preallocate** buffers once; reuse across iterations.
- **Stack**: fixed-size `float buf[N]` for small N known at compile time.
- **Avoid** `std::vector::resize` in loops — `reserve(n)` once.
- **Arena/pool** allocators for many same-sized objects.
- **`alloca()` / VLAs**: not "effectively free" — can overflow stack, inhibit optimization, create large per-thread consumption.

## Precomputation & Sub-expression Reuse

```c
// Bad: exp called twice
double morse(double x, double D, double a, double r0) {
    double dx = x - r0;
    return D * (exp(-2*a*dx) - 2*exp(-a*dx));
}

// Good: compute e once, reuse for value and derivative
inline void morse_eval(double D, double a, double r0, double x, double *E, double *F) {
    double dx = x - r0;
    double e  = exp(-a * dx);
    *E = D * (e*e - 2*e);
    *F = 2*D*a * (e*e - e);
}
```

- Precompute lookup tables for repeated expensive evaluations.
- Recompute cheap values instead of loading from memory when arithmetic is cheaper than a cache miss.

## Avoid Transcendental Functions

`sin`, `cos`, `exp`, `log` cost tens to hundreds of cycles. Alternatives:

- Polynomial / spline approximations under an error budget.
- Complex multiplication for repeated rotation (one `cos`/`sin`, then multiply).
- Algebraic representations (quaternions, rotation matrices) over angles.

When unavoidable: compute outside loop, or use fast approximations with documented accuracy.

## Multi-threading (OpenMP)

### Avoid repeated fork/join

```c
// BAD: fork-join every iteration (~µs latency each)
for (int itr = 0; itr < niter; itr++) {
    #pragma omp parallel for reduction(+:E)
    for (int i = 0; i < n; i++) E += evalForce(i);
}

// GOOD: persistent parallel region, reset state per iteration
#pragma omp parallel
{
    for (int itr = 0; itr < niter; itr++) {
        #pragma omp single
        E = 0.0;

        #pragma omp for reduction(+:E)
        for (int i = 0; i < n; i++)
            E += evalSingleAtom(i);

        #pragma omp for
        for (int i = 0; i < n; i++)
            moveSingleAtom(i);
    }
}
```

### Assembly pattern (avoid contended writes)

When multiple interactions contribute to one atom, use per-interaction buffers + gather assembly instead of contended atomics — benchmark against atomics when contention is low.

```c
// Phase 1: each thread evaluates its atoms (no write races)
#pragma omp for reduction(+:E)
for (int ia = 0; ia < natoms; ia++)
    E += evalSingleAtom(ia);

// Phase 2: assemble neighbor contributions
#pragma omp for
for (int ia = 0; ia < natoms; ia++)
    assemble_atom(ia);
```

### Scheduling and affinity

- `schedule(static)` for uniform work; `schedule(dynamic)` for irregular cost.
- Pin threads to cores when NUMA or cache affinity matters (`OMP_PROC_BIND`, `numactl`).
- Avoid oversubscription: don't nest OpenMP inside threaded BLAS — set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` consistently.

## Profiling Tools

```bash
perf stat -e cycles,instructions,cache-misses,branch-misses ./program
perf record -g ./program && perf report
```

Useful metrics: IPC, L1/L2/L3 miss rate, TLB misses, effective memory bandwidth, vectorization ratio from compiler reports.

## Bottleneck Decision Tree

| Symptom | Likely cause | Try |
|---------|-------------|-----|
| High cache-miss rate, low IPC | Memory bandwidth / poor locality | SoA, blocking, loop interchange, hot/cold split |
| Low IPC, few cache misses | Dependency chains, scalar code | Unroll cautiously, CSE, algebraic simplification |
| High branch-miss rate | Unpredictable branches | Sort by branch outcome, split kernels, separate rare cases |
| Good single-thread, poor scaling | False sharing, NUMA, oversubscription | Pad accumulators, first-touch, thread pinning |
| OpenMP overhead dominates | Too many fork/join | Persistent parallel region |

## Repo-specific: PySCF CPU smallDFT (grid XC arc)

Validated on benzene PBE 6-31g (`nao=66`, `ngrids≈144k`). Full numbers: `doc/CPU_benchmark.md`; distilled lessons: `doc/CPU_optimixation_experience.md`.

### Problem shape

Grid XC is **memory-bandwidth bound**: stream χ `(ngrids, nao)` F-contiguous; reuse DM in cache; parallel axis is **grid index `g`**, not atoms or Python threads.

### Profiling (MUST for this repo)

Three levels — run all after each change:

1. **Parity + sub-task scaling:** `expamples_prokop/test_small_dft.py` (`--rho`, C vmat)
2. **XC waterfall:** `pyscf.smallDFT.profile_xc_bottleneck`
3. **Full SCF cycle (Amdahl):** `expamples_prokop/profile_scf_cycle.py` — real `mf.kernel()`, init vs cycle columns; not `--manual`

Patch **`RHF.get_jk`** for J timing (RKS inherits RHF). Set `OPENBLAS_NUM_THREADS=1`; authoritative OMP via `lib.num_threads(N)`.

If `perf_event_paranoid` blocks hardware counters, use wall time + GCC `-fopt-info-vec` + cProfile (`--profile`).

### Structural fixes that worked

1. **Grid OpenMP tiles** — private ρ writes; private `V_t` + reduction for vmat (no hot-loop atomics).
2. **Keep libcint F-layout** — BLAS `lda=ngrids` on tiles; do not transpose χ at the Python boundary.
3. **Stride-1 inner loop over `g`** — AO outer, grid inner; not `ddot` per point with stride `ngrids`.
4. **F-order tile buffers** — `dgemm("T","N")` for vmat, not `("T","T")` on C-order tiles.
5. **AO cache** — `GridWorkspace.eval_ao()` once per geometry (~4× cycle win vs ref).

### False premises (benchmark-disproved)

- Python `ThreadPoolExecutor` over grid tiles scales (caps ~2×; use C/OpenMP).
- Bigger `TILE` always helps (512 optimal for benzene; 2048 slower @8 CPU).
- Coulomb J is next bottleneck for benzene 6-31g (incremental J ~2 ms; XC ~26 ms/cycle).
- `eval_ao` scales with more CPUs (flat ~55–66 ms; libcint serial on grid axis).
- Hand-rolled `get_veff` timing equals real kernel (misses `RHF.get_jk`, wrong `energy_tot`).

### Open issues

Fuse ρ+PBE+vmat in one χ pass; grid-parallel `eval_gto`; preallocated C scratch (no malloc in hot path); auto-attach `GridWorkspace` in `patch.enable()`.

## Related Skills

- skill:`python-perf` — Python harness overhead, NumPy temporaries, when to escape to compiled code
- skill:`gpu-optimize` — GPU-specific optimization (tiling, local memory, atomics); same physics, different parallel axis
- skill:`numerical-parity` — validate optimized path against reference
