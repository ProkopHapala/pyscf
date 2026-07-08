---
type: ExperienceReport
title: CPU smallDFT optimization — lessons learned
description: Strategies, caveats, disproved assumptions, and generalization from grid-parallel XC on small molecules
tags: [cpu, optimization, smallDFT, openmp, amdahl, profiling]
timestamp: 2026-07-08
---

## Scope

This document captures what we learned optimizing **CPU grid XC** for small molecules (`nao ≲ 200`, `ngrids ~ 30k–150k`) in PySCF: grid-parallel ρ and vmat in `libsmalldft`, drop-in `smallDFT.nr_rks`, and full-SCF profiling.

**Not covered here:** GPU OpenCL path (see `doc/GPU_benchmark.md`), general DFT theory, or large-molecule / MGGA / UKS extensions.

**Numbers:** benzene PBE 6-31g grid 3, `nao=66`, `ngrids=143560`, `OPENBLAS_NUM_THREADS=1`, OMP via `lib.num_threads(N)`. Full tables: [/home/prokop/git/pyscf/doc/CPU_benchmark.md](/home/prokop/git/pyscf/doc/CPU_benchmark.md).

---

## Problem shape

Grid XC on small molecules is a **memory-bandwidth loop over grid points** with occasional dense linear algebra:

```
eval_ao(χ)  →  ρ(χ, DM)  →  libxc(ρ)  →  vmat(χ, wv)
```

PySCF’s reference path parallelizes **atoms/blocks** inside libcint and libdft. For small `nao`, that leaves most cores idle while each thread walks a long serial grid loop. The winning CPU strategy is:

1. **Parallel axis = grid index** (embarrassingly parallel over `g`)
2. **Keep libcint AO layout** (F-contiguous `(ngrids, nao)`)
3. **Escape hot loops to C/OpenMP**; Python is orchestration only
4. **Profile the real call graph**, not hand-rolled substitutes

---

## What worked (strategies)

### 1. Grid as the parallel axis

Each OpenMP thread owns a disjoint grid tile `[g0, g1)`. Writes to `ρ[g]` are disjoint; vmat uses **private `V_t` buffers** + reduction (no atomics in the hot loop).

**Result:** ρ scales **4.2×** (1→8 CPU) vs original PySCF **1.98×** on the same sub-task.

### 2. Keep native χ layout; BLAS with `lda=ngrids`

ctypes passes NumPy pointers directly. BLAS `dgemm` on χ tiles uses `lda=ngrids` on F-order slices — **no pack buffer** into row-major `(nao, ngrids)`.

**Rule:** adapt the math to the layout you already have; don’t transpose “because GEMM wants it” unless profiling proves the copy is cheaper than strided BLAS.

### 3. C/OpenMP for ρ and vmat; abandon Python thread pool on the hot path

`ThreadPoolExecutor` over grid tiles capped at ~2× and added GIL/orchestration overhead. Production policy: **C-only** when `libsmalldft` is built.

### 4. Pin BLAS to one thread; parallelize on grid

Set `OPENBLAS_NUM_THREADS=1` (or `MKL_NUM_THREADS=1`) and use explicit grid OpenMP. Nested OMP + threaded BLAS oversubscribes and destroys scaling.

### 5. Cache AO once per geometry (`GridWorkspace`)

`ws.eval_ao()` once, then reuse χ across SCF cycles. This removes libcint from the per-cycle path.

**Result:** ref `nr_rks` @8 CPU plateaus at ~120 ms/cycle (includes `eval_ao`); smallDFT_ws drops to **~26 ms/cycle** for XC only.

### 6. Stride-1 inner loops over grid (Jul 2026)

Early C kernels used `ddot` per grid point with stride `ngrids` over AO — cache-hostile. Reordering to **AO outer, grid inner** with `#pragma omp simd` on the inner loop:

| kernel @8 CPU | before | after |
|---------------|-------:|------:|
| `rho_gga` | ~20 ms | **~10 ms** |
| `vmat_gga` | ~16 ms | **~11 ms** |
| XC subtotal | ~47 ms | **~24 ms** |

F-order tile buffers for `aow` / `chi_w` and `dgemm("T","N")` instead of `("T","T")` helped vmat further.

### 7. Profile real `mf.kernel()`, not a fake decomposition

Hand-rolled timers that call `nr_rks` + `get_j` separately **missed** `RHF.get_jk`, `rks.get_veff`, and density-fitting paths. The proper profiler monkey-patches the actual PySCF call graph (`expamples_prokop/profile_scf_cycle.py`).

### 8. Measure optimized sub-tasks separately from full SCF

Micro-benchmarks (ρ only, vmat only, AO cached) isolate kernel scaling. Full SCF cycle profiling exposes **Amdahl limits** (J, diag, init guess, pre-SCF).

### 9. One structural change, then re-benchmark

Following cpu-perf workflow: parity check → one hypothesis → measure. Examples: GGA ρ 4× `DM@χ` bug, vmat Fortran/C transpose on reduce, hermi `V += V.T` via temp buffer, `GridWorkspace` F-contiguity bug.

---

## Caveats and pitfalls

### Layout and correctness

| pitfall | symptom | fix |
|---------|---------|-----|
| Preallocate `(4, ngrids, nao)` for χ | `chi[0]` not F-contiguous → 3× slower Python path | always use `eval_ao_native` output |
| vmat BLAS layout mismatch | wrong vmat / parity fail | transpose on reduce; GGA hermi via temp buffer |
| GGA ρ with 4× `DM@χ` | 4× wrong ρ₀ | one GEMM + factor 2 (hermi=1) |
| `energy_tot` without `ecoul`/`exc` on tagged vhf | silent re-call of full `get_veff` (~130 ms fake “energy”) | return properly tagged `vhf` from `get_veff` |

### Parallelism

| pitfall | symptom | fix |
|---------|---------|-----|
| OMP + OpenBLAS both at N threads | scaling flat or negative | `OPENBLAS_NUM_THREADS=1`; grid OMP only |
| OpenMP `shared(two)` wrong in GGA ρ | wrong ρ₀ / segfault | correct `private`/`shared` scoping |
| Patch `SCF.get_jk` only | timer shows no J in RKS | patch **`RHF.get_jk`** (RKS inherits from RHF) |
| Summing sub-timer mins that each run full `get_veff` | “energy” = 50% of cycle | time full calls once; use `_min_veff` pattern |

### Profiling

| pitfall | symptom | fix |
|---------|---------|-----|
| Sub-task mins from separate invocations | J looks like 120 ms when it’s 2 ms | one invocation, record all components |
| `--manual` profile mode | no `get_jk` in table | use default `mf.kernel()` mode |
| Only micro-benchmarks | miss DF J, init cost, diag | run `profile_scf_cycle.py` |

### Memory

| pitfall | symptom | fix |
|---------|---------|-----|
| `malloc`/`free` per tile per call | overhead on small systems; breaks first-touch | preallocated scratch (TODO) |
| `omp critical` vmat reduce | serializes at end of each kernel | acceptable for `nao≈66`; consider tree reduce for large `nao` |
| Larger `TILE` always better | 2048 slower than 512 @8 CPU | benchmark; default `TILE=512` |

---

## Rules of thumb

### When to use this CPU path

| criterion | guidance |
|-----------|----------|
| `nao` | ≲ 200 (policy default); benefits fade below ~30 |
| `ngrids` | ≳ 30k (below that, setup dominates) |
| XC | LDA + GGA RKS; PBE tested |
| SCF | multiple cycles per geometry (AO cache pays) |
| threads | 4–8 grid OMP + BLAS pinned to 1 |

### Parallelism

- **Parallel axis:** grid index `g`, not atom, not Python thread pool.
- **ρ:** embarrassingly parallel; disjoint writes.
- **vmat:** private `V_t` per thread + reduction; no hot-loop atomics.
- **Scaling expectation:** ρ ~4–6× on 8 CPU; vmat ~3–4× (heavier GEMM, memory-bound).

### Memory layout

- **χ:** F-contiguous `(ngrids, nao)` or `(4, ngrids, nao)` per component — libcint native.
- **ρ, wv:** C-contiguous `(4, ngrids)` — libxc convention.
- **DM, vmat:** C-contiguous `(nao, nao)`.
- **Inner loop:** stride-1 over `g` inside each tile; AO index outer.

### Profiling workflow

1. **Parity first** — `expamples_prokop/test_small_dft.py`
2. **Sub-task scaling** — `test_small_dft.py --rho`, `profile_xc_bottleneck()`
3. **Full SCF cycle** — `profile_scf_cycle.py` (real `mf.kernel`, init vs cycle columns)
4. **One change at a time** — re-run all three layers

### Amortization

| work | frequency | benzene @8 CPU (typical) |
|------|-----------|-------------------------:|
| `Grids.build` | once / geometry | ~100 ms |
| `eval_ao` | once / geometry | ~66 ms |
| `nr_rks` (XC) | every SCF cycle | ~26 ms |
| `get_jk` incremental | every SCF cycle | ~2 ms |
| `eig` | every SCF cycle | <1 ms |

**End-to-end @ N cycles:** pre_scf + init + N×cycle. At N=10: ref ~1530 ms → smallDFT_ws ~560 ms (~2.7×). At N=20: ~3.3×.

---

## False premises disproved by benchmarks

| premise | reality | evidence |
|---------|---------|----------|
| “More Python threads will scale ρ/vmat” | caps ~2×; GIL + BLAS contention | benzene Python nw=8 ≈ nw=1 on full `nr_rks` |
| “OpenMP inside original PySCF ρ is enough” | 1.98× @8 CPU on ρ sub-task | vs 4.2× smallDFT C |
| “Optimizing ρ alone speeds up SCF” | XC still 98% of cycle; then vmat, then AO | Amdahl: ref cycle 120 ms, ρ sub-task 20 ms |
| “vmat is negligible after ρ is fast” | vmat was 41% of XC; now still 46% | bottleneck waterfall |
| “eval_ao will scale with more CPUs” | flat ~55 ms | 0.94× scale 1→8 |
| “Coulomb J is always the next bottleneck” | incremental J ~2 ms for benzene 6-31g direct J | `scf.get_jk` cycle column |
| “Density fitting doesn’t matter for small molecules” | `df.get_jk` ~30 ms/cycle (~20% of veff) | `--df` profile |
| “Diagonalization limits small-molecule SCF” | <1 ms eig vs 26 ms XC | `profile_scf_cycle` |
| “Transpose χ to (nao, ngrids) for faster GEMM” | strided BLAS on F-tiles wins | no pack buffer needed |
| “Bigger TILE is always better” | TILE=2048 worse than 512 @8 CPU | tile sweep |
| “ddot per grid point is fine in C” | stride-1 reorder: ρ 20→10 ms | Jul 2026 kernel rewrite |
| “Hand-rolled get_veff timing equals real kernel” | missed J path, wrong energy_tot cost | proper profiler |
| “Linear scaling on ρ implies linear SCF” | full `nr_rks` limited by serial eval_ao | ref plateaus at 120 ms/cycle |
| “H2O is a good scaling benchmark” | cycle <3 ms; grid build dominates | not representative |

---

## Amdahl lessons (benzene 6-31g, 8 CPU)

### Before optimization (ref)

```
get_veff  122 ms
  nr_rks  120 ms  (98%)   ← eval_ao + block_loop + ρ + libxc + vmat
  get_jk    2 ms  ( 2%)   ← incremental Δdm only
  eig      <1 ms
```

### After smallDFT_ws + stride-1 C

```
get_veff   28 ms
  nr_rks   26 ms  (93%)   ← still XC-limited
  get_jk    2 ms  ( 7%)
```

**Floor if XC → 0:** ~3 ms (J_incr + diag). Not reachable until `nr_rks` is fully optimized.

**What hides inside `nr_rks` after AO cache:**

| step | time @8 CPU |
|------|------------:|
| ρ_gga C | 10 ms |
| vmat_gga C | 11 ms |
| libxc | 3 ms |

**Next bottleneck after XC kernels are fast:** fuse ρ+vmat (one χ pass), then **grid-parallel eval_ao** (~58 ms once/geometry).

---

## Generalization: CPU optimization of similar problems

This pattern applies to any **grid-driven integral** where:

- `ngrids ≫ nao` (or at least `ngrids ~ 10³–10⁵`)
- the hot loop is over grid points with gather/scatter on a moderate-dimensional basis
- a legacy library owns the “natural” layout (here: libcint F-order χ)

### Template

```
1. Identify parallel axis     →  grid index (not the library’s internal axis)
2. Keep authoritative layout  →  don’t transpose at the Python boundary
3. Tile for cache            →  TILE ~ 512, working set fits L2/L3
4. Inner loop stride-1       →  over the axis you parallelize
5. BLAS for dense blocks     →  DM@χ, χᵀ@aow; one thread per BLAS call
6. Private buffers + reduce  →  for accumulations (vmat), not atomics
7. Hoist invariant work      →  AO per geometry, grid per job, DM constant per cycle
8. Escape to C/OpenMP        →  when Python/thread overhead > 10% of kernel
9. Profile three levels      →  kernel / XC integral / full physics step (SCF cycle)
10. Re-profile after each win →  Amdahl shifts the bottleneck
```

### Analogous problems in electronic structure

| problem | parallel axis | accumulate | hoist |
|---------|---------------|------------|-------|
| grid XC ρ | grid `g` | disjoint `ρ[g]` | χ per geometry |
| grid XC vmat | grid `g` | private `V_t` + sum | χ, wv per cycle |
| grid property integrals | grid `g` | private buffer | AO, operator |
| real-space operators on grids | grid `g` | often disjoint | grid weights |

### When this template does *not* apply

- **`nao` large** — `nao²` vmat reduce and memory dominate; different tiling
- **MGGA / meta-GGA** — extra ρ derivatives and terms
- **UKS / spin** — doubled χ and ρ channels
- **Molecular dynamics** — geometry changes every step; AO cache invalidates unless fused with update
- **Tiny systems** — setup and Python overhead dominate; optimize elsewhere

### CPU vs GPU analogy (same physics, different axis)

| aspect | CPU smallDFT | GPU OpenCL (see GPU notes) |
|--------|--------------|----------------------------|
| parallel axis | grid OpenMP | grid tiles (ρ) vs atom tiles (vmat tiled) |
| layout | keep libcint F-order | Hermite/precomp tradeoffs |
| bottleneck shift | eval_ao → fuse ρ+vmat | vmat geometry → radial precomp |
| lesson | same: **match parallelism to the loop that owns the work** | same |

---

## Implementation checklist (for the next optimizer)

- [ ] Parity: `test_small_dft.py` (PBE, LDA, C ρ, C vmat, full `nr_rks`)
- [ ] Sub-task: ρ, vmat, ρ+libxc+vmat @ 1/2/4/8 threads
- [ ] XC waterfall: `profile_xc_bottleneck()`
- [ ] SCF cycle: `profile_scf_cycle.py` ref vs `smallDFT_ws`, init vs cycle
- [ ] Optional: `--df`, `--profile` (cProfile)
- [ ] Check `OPENBLAS_NUM_THREADS=1`
- [ ] Document numbers in `CPU_benchmark.md`

---

## Open issues (priority)

| P | item | why |
|---|------|-----|
| 1 | Fuse ρ + PBE + vmat in one χ tile pass | removes second χ read; needs C PBE or libxc-per-tile |
| 2 | Grid-parallel `eval_gto` | ~58 ms flat; dominant once XC is fast |
| 3 | Preallocated C scratch (no malloc in hot path) | first-touch, less overhead |
| 4 | `patch.enable()` attaches `GridWorkspace` on `mf` | production ergonomics |
| 5 | vmat tree reduction vs `omp critical` | matters when `nao` grows |

---

## Related docs

- [/home/prokop/git/pyscf/doc/smallDFT_cpu_path.md](/home/prokop/git/pyscf/doc/smallDFT_cpu_path.md) — implementation guide
- [/home/prokop/git/pyscf/doc/CPU_benchmark.md](/home/prokop/git/pyscf/doc/CPU_benchmark.md) — numbers and reproduce commands
- [/home/prokop/git/pyscf/doc/CPU_small_DFT.chat.md](/home/prokop/git/pyscf/doc/CPU_small_DFT.chat.md) — original analysis
- [/home/prokop/git/pyscf/.cursor/skills/cpu-perf/SKILL.md](/home/prokop/git/pyscf/.cursor/skills/cpu-perf/SKILL.md) — CPU optimization workflow
- `expamples_prokop/profile_scf_cycle.py` — proper SCF-cycle profiler
- `expamples_prokop/test_small_dft.py` — parity and sub-task scaling
