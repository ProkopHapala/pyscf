# USER

yes, I want you to make detailed plan how we can write optimized paralle path for small molecule focusing on grid opertions (projectiong denity on grid, projection AO on grid, assembling VMAT form grid), if we hould do it using python array slicing or in C/C++ with OpenMP, but plese think carefully about data oriented desing since this will be perhaps memory-bound operation, and we must avoid thread synchornizaton (atomic write shared variables) 

do not program anythin yet, just analyse the problem and write notes

---

# Cursore Composer 2.5

# Plan: Optimized CPU Parallel Path for Small-Molecule Grid Operations

Analysis notes only — no implementation. Goal: a CPU path competitive with (or complementary to) your OpenCL work for **nao ≲ 200**, **ngrids ≈ 30k–150k**, focusing on AO projection, ρ on grid, and vmat assembly.

---

## 1. Problem statement

Per SCF cycle, GGA XC needs:

| Step | Math | Current CPU hot spot |
|------|------|----------------------|
| **AO on grid** | χ_μ(g), ∂χ_μ/∂x,y,z | `eval_gto` → `GTOeval_loop` in `grid_ao_drv.c` |
| **ρ on grid** | ρ(g) = Σ_μν DM_μν χ_μ(g)χ_ν(g); GGA adds ∇ρ | `eval_rho1` → `_dot_ao_dm_sparse` + `_contract_rho_sparse` |
| **vmat** | V_μν = Σ_g w_g · v_xc(g) · χ_μ(g) · (weighted χ)_ν(g) | `_scale_ao_sparse` + `_dot_ao_ao_sparse` |
| **XC** (out of scope here but coupled) | ρ → vrho, vsigma via libxc | `LIBXC_eval_xc` — already OpenMP over grid |

Your OpenCL kernels (`rho_*_pair`, `vmat_*_pair`) already solve the **parallel decomposition** problem correctly: tile over grid points, reuse DM in fast memory, avoid scatter. The CPU path should mirror that logic, not the current PySCF `block_loop` + tiny GEMM approach.

---

## 2. Why the existing CPU path is the wrong shape

Three structural issues (from prior analysis):

1. **Serial Python orchestration** — `block_loop` runs 1–3 iterations for small molecules; no cross-block parallelism.
2. **`nao < SWITCH_SIZE (800)` gate** — disables sparse C kernels; falls back to `lib.dot`/`lib.ddot` on matrices too small for threaded BLAS.
3. **Wrong memory layout for grid-parallel access** — PySCF stores χ as **F-contiguous `[iG, iAO]`** (stride `nao` along grid). Grid-parallel threads want **contiguous `iG`** at fixed `iAO` → your OpenCL doc calls this the "stride-n gather" bug.

Existing sparse C code (`nr_numint_sparse.c`) is designed for **large nao** (shell-box decomposition, `BOXSIZE1_N` etc.). It is not tuned for tiny nao + huge ngrids.

---

## 3. Roofline / memory-bound analysis

For precomputed AO (GGA, 4 components):

```
Memory (read per SCF):  χ ≈ 4 × nao × ngrids × 8 bytes
                        DM ≈ nao² × 8 bytes  (fits in L2/L3)
                        ρ, wv ≈ O(ngrids)

Flops (ρ):              ≈ 2 × nao² × ngrids  (symmetric DM)
Flops (vmat):           ≈ 2 × nao² × ngrids  (same structure)
```

Example H2O cc-pVDZ: nao=24, ngrids=33k

| Quantity | Value |
|----------|-------|
| χ read (GGA) | 4 × 24 × 33k × 8 ≈ **25 MB** |
| DM | 24² × 8 ≈ **4.6 KB** |
| ρ flops | 2 × 576 × 33k ≈ **38 MFLOP** |
| Arithmetic intensity | ~1.5 FLOP/byte → **memory-bound** |

Example benzene: nao=114, ngrids=144k

| Quantity | Value |
|----------|-------|
| χ read (GGA) | 4 × 114 × 144k × 8 ≈ **525 MB** |
| DM | 114² × 8 ≈ **100 KB** |
| ρ flops | 2 × 13k × 144k ≈ **3.7 GFLOP** |
| Arithmetic intensity | ~7 FLOP/byte → still **memory-bound** on typical CPUs (~10–20 FLOP/byte peak) |

**Conclusion:** This is a **bandwidth + cache reuse** problem, not a raw FLOP problem. Design must:

- Stream χ with **unit stride in `iG`**
- **Reuse DM** maximally while touching each χ slice once
- **Fuse** ρ and (where possible) vmat to avoid multiple full χ passes
- Avoid Python/NumPy temporaries that double memory traffic

---

## 4. Python slicing vs C/C++ OpenMP

| Approach | Verdict |
|----------|---------|
| **Python `block_loop` + NumPy slices** | Too slow: GIL released per call but each step allocates/strides; 4–5 separate passes over χ; no fusion; BLAS on tiny nao doesn't thread well. Good for prototyping parity only. |
| **Numba/Cython** | Possible for ρ/vmat if precomputed χ; still awkward for `eval_gto` (libcint integration). Not PySCF convention. |
| **Extend existing `libdft` C + OpenMP** | **Recommended.** Matches PySCF architecture, reuses `non0tab`/screening, callable from `numint.py` like `VXCdot_ao_ao_sparse`. |
| **Separate small-mol module** | Also fine: `lib/dft/small_grid.c` with clean API, opt-in from `numint.py` when `nao < N_SMALL`. |

**Rule:** Python owns setup, grid build, libxc call, and result packaging. **All per-grid hot loops in C.**

---

## 5. Data-oriented design (SSOT layouts)

Adopt the same layouts your OpenCL precomp doc recommends — CPU and GPU should share indexing logic.

### 5.1 Primary arrays (Structure of Arrays, grid-major)

```
coords[g, 3]          // unchanged from Grids
weights[g]            // unchanged

χ0[iAO, iG]           // AO value;     C-contiguous in iG  (transpose of PySCF default)
χ1[iAO, iG]           // dφ/dx
χ2[iAO, iG]           // dφ/dy  
χ3[iAO, iG]           // dφ/dz

ρ[4, iG]              // or ρ0[iG], ρx[iG], ... separate (better for LDA)

wv[4, iG]             // weighted XC potential factors after libxc

DM[iAO, jAO]          // symmetric; store full or upper-tri in registers

V[iAO, jAO]           // vmat output; symmetric for LDA, full for GGA then hermi
```

**Index:** `χc[iAO * ngrids + iG]` — threads with consecutive `iG` read consecutive addresses.

### 5.2 Atom-blocked variant (optional, mirrors OpenCL `NATILE`)

For molecules with ≤ ~20 atoms:

```
χ[c, ia, a, iG]   where a ∈ [0, nao_atom[ia]), pad to MAX_AO_ATOM
DM_blk[ia, ja, a, b]  // pre-extracted per atom pair, setup once per SCF
```

Benefits:
- Inner loops are dense `a × b` without `ao_loc` indirection
- Natural parallel task = `(gTile, ia, ja)` atom pair
- Screening: skip pair if `pair_mask[ia,ja]` or atom not in grid shell

### 5.3 Tile constants (start values, tune per CPU)

| Symbol | Suggested | Role |
|--------|-----------|------|
| `NPTILE` | 32 or 64 | Grid points per tile (fits L1: 32 × nao × 8 × 4 comp) |
| `NATILE` | 1–2 atoms | Atom batch in inner loop |
| `NAO_TILE` | 16–32 | AO block when not atom-grouped |

For H2O (nao=24): entire AO fits one tile → `NPTILE=64` gives 64×24×4×8 ≈ 49 KB χ tile in L1.

### 5.4 What not to store

- Avoid materializing `aow = χ × wv` as full `[nao, ngrids]` for GGA if fusing vmat — that's an extra nao × ngrids write+read.
- Avoid `[4, ngrids, nao]` NumPy layout from current precomp staging.

---

## 6. Parallel decomposition without atomics

**Principle:** every parallel task must own **disjoint output memory** for the duration of the parallel region. Merge with **deterministic reduction** after the parallel loop (or use disjoint writes that cover the output once).

### 6.1 ρ on grid — parallel over `gTile` (recommended primary)

```
Task t owns grid indices [t*NPTILE, (t+1)*NPTILE)
Writes: ρ[c, g] for g in that range only
Reads:  DM (shared, read-only), χ[c, :, g] (private streaming)
```

- **No atomics.** Each `g` written by exactly one thread.
- Inner loop: for each `g` in tile, accumulate `Σ_μν DM_μν χ_μ(g) χ_ν(g)` with μ,ν loops in registers.
- Symmetric DM: iterate `μ ≤ ν`, factor 2 off-diagonal.

**Alternative (not preferred for small nao):** parallel over AO shells, private `ρ_t[ngrids]` buffer per thread → `NPomp_dsum_reduce` at end (existing pattern in `VXCdcontract_rho_sparse` when `nao*2 >= ngrids`). Works but multiplies ρ buffer by n_threads and needs reduction.

### 6.2 vmat assembly — parallel over `(iAO_block, jAO_block)` or fused with ρ

**Option A — disjoint AO blocks (mirrors existing `VXCdot_ao_ao_sparse`):**

```
Task (ib, jb) owns V[i0:i1, j0:j1]
Accumulates over all g:  V_μν += Σ_g wv(g) χ_μ(g) χ_ν(g)   [GGA: more terms]
```

- Each matrix block written by one thread → **no atomics**
- After parallel loop: symmetrize if hermi (`V += V.T` for GGA, or only compute upper triangle)
- Existing code already does this with `outbuf` per task; problem is task granularity is shell-box sized for large systems, not tuned for nao=24

**Option B — parallel over `gTile` with private `V_t[nao, nao]`:**

```
Per thread t:  V_t = 0
For g in tile:  V_t[μ,ν] += wv(g) χ_μ(g) χ_ν(g)
After parallel: V = Σ_t V_t   (nao² reduction, nao small → cheap)
```

- **No atomics** during accumulation
- Memory: `n_threads × nao² × 8` — for nao=114, 8 threads → ~800 KB, acceptable
- Advantage: same gTile traversal as ρ → **fuse ρ + vmat in one χ pass**

**Recommendation:** Option B (gTile + private vmat) for **nao < 150** because it enables fusion and matches OpenCL `rho_gga_pair` structure. Option A for larger nao if memory for per-thread vmat becomes costly.

### 6.3 AO on grid — two sub-strategies

**Path 1 — Precomputed χ (per geometry, like GPU `precomputed`):**

- Setup once: evaluate χ on full grid (can use existing `eval_gto`), transpose to `[iAO, iG]`
- Per SCF: only ρ + vmat (χ reused) — **this is the highest ROI for repeated SCF**
- AO eval parallelization only matters at setup / geometry change

**Path 2 — On-the-fly Hermite (like GPU `onthefly`):**

- Parallel over `(gTile, ia, ja)` atom pair — same task graph as `rho_lda_tiled`
- Each task: evaluate radial × angular for atoms ia, ja at grid tile, contract with DM block immediately
- **No χ buffer at all** — trades compute for memory
- For CPU: worth it only if χ memory exceeds L3 and hurts; for H2O χ is 25 MB → precompute is fine

**AO eval parallelization (when needed):**

```
Parallel over gTile:
  Each thread evaluates all AOs at coords[g0:g1]  → writes χ[:, g0:g1]
  Disjoint output columns → no atomics
```

This is better than current `GTOeval_loop` (parallel over shell×grid_block) for small molecules because it keeps **one grid tile hot in cache** while sweeping shells, rather than one shell hot while jumping grid blocks.

However, rewriting `eval_gto` is a large project (libcint). **Pragmatic phase 1:** use existing `eval_gto` at setup, transpose once. **Phase 2:** Hermite CPU port from your `radial_hermite.py` / `ao_hermite.py` logic.

---

## 7. Operation-by-operation design

### 7.1 AO projection (setup or per-geometry)

```
Input:  coords[g], mol, basis
Output: χ[c, iAO, iG]  for c=0..3 (GGA)

Steps:
  1. eval_gto (existing) → buf_F[c, iG, iAO]   # PySCF layout
  2. transpose per component (one-time, can OpenMP):
       χ[c, iAO, iG] = buf_F[c, iG, iAO].T      # C-contiguous in iG
  3. optional: atom-block reorder χ[c, ia, a, iG]
  4. optional: build pair_mask[ia,ja], atom_ao_offsets
```

**Screening:** keep `non0tab[g_blk, shell]` from grid build; at ρ/vmat time skip atoms whose shells are zero on that grid tile. Don't skip the transpose — it's cheap vs SCF.

### 7.2 ρ projection (per SCF)

**Kernel: `small_rho_gga_gtile`**

```
#pragma omp parallel for schedule(static)  // gTile loop, no dynamic overhead
for (gTile = 0; gTile < ngrids; gTile += NPTILE):
    g0 = gTile; g1 = min(g0+NPTILE, ngrids)
    // stack arrays: chi0_tile[NAO_TILE][NPTILE] loaded with coalesced reads
    
    for g in g0..g1:
        ρ[g] = 0; ρx[g] = 0; ...
        for ia in atoms:
          if !atom_active_on_tile(ia, gTile): continue
          for ja in ia..natoms:   // symmetric
            if !pair_mask[ia,ja]: continue
            load DM_blk[ia,ja] into registers (small)
            for a,b in atom AOs:
              accumulate ρ, ρx, ρy, ρz from χ values at g
```

**Micro-optimizations:**
- `schedule(static)` — equal work per gTile
- `#pragma omp simd` on inner `g` or `b` loop where compiler supports it
- `fma` for ρ accumulation
- For H2O: unroll entire nao=24 double loop (specialized `small_rho_gga_nao24` dispatch)

### 7.3 vmat assembly (per SCF, after libxc)

**Fused kernel: `small_vmat_gga_gtile_fused`** (if ρ already computed, still can fuse χ read with vmat):

```
#pragma omp parallel for schedule(static)
for gTile ...:
    V_priv[nao, nao] = 0   // thread-local, stack or malloc once per thread
    
    for g in tile:
        w0 = weights[g] * vrho[g]
        ws = weights[g] * vsigma[g] * ...   // GGA chain rule factors
        for μ,ν:
            V_priv[μ,ν] += w0*χ0[μ,g]*χ0[ν,g] + ws * (∂χ terms)...
    
    // reduction: only upper triangle if hermi
    merge V_priv into global V  // see below
```

**Merge without atomics:**

```c
// After parallel region — serial reduction over threads (nao² small)
for (t = 0; t < nthreads; t++)
    for (ij = 0; ij < nao*nao; ij++)
        V[ij] += V_thread[t][ij];
```

Or: give each thread a **disjoint slice of gTile** and accumulate into **one shared V** using **private buffers + single parallel reduction** (`NPomp_dsum_reduce` pattern already in PySCF).

**Do not** use `omp atomic` on `V[μ,ν]` inside the `g` loop.

### 7.4 libxc (keep as-is for now)

`LIBXC_eval_xc` already parallelizes over grid points with thread-local buffers. After ρ:

```
Input:  ρ[4, ngrids]  (C-contiguous in g)
Output: vrho[g], vsigma[g], exc[g]
```

Then compute `wv[g]` (weighted combination for vmat) — can fuse into vmat kernel to avoid storing `wv` if desired.

---

## 8. Avoiding synchronization pitfalls

| Anti-pattern | Why bad | Fix |
|--------------|---------|-----|
| `omp critical` on `V[μ,ν]` per grid point | Serializes hot loop | Private `V_t`, reduce after |
| `omp atomic` on ρ or V | Memory-bound + contention | Disjoint `gTile` ownership |
| Nested `omp parallel` (eval_ao + BLAS + libxc + vmat) | Oversubscription | **One parallel region** per phase; use `lib.num_threads(n)` globally; consider `OMP_MAX_ACTIVE_LEVELS=1` |
| Dynamic scheduling on uniform gTile | Overhead > benefit for small mol | `schedule(static)` |
| Multiple χ passes (ρ, then scale_ao, then dot) | 3× memory traffic | Fuse ρ+vmat or ρ+wv+vmat |
| Per-block `malloc` in parallel region | Allocator contention | Thread-local buffers allocated once in `#pragma omp parallel` before `for` |

Existing `VXCdot_ao_ao` (dense) uses `omp critical` for reduction — **do not copy this pattern** for the small-mol path.

---

## 9. Integration with PySCF (when implementing)

Suggested hook (later):

```
numint.py:block_loop / nr_rks
  if nao <= N_SMALL and xctype in ('LDA','GGA'):
      if plan.has_precomputed_chi:
          libdft.SMALL_nr_rks_precomp(...)
      else:
          libdft.SMALL_nr_rks_hermite(...)
  else:
      existing path
```

Or separate `backend=4` / `xc_path='small_cpu'` mirroring your GPU hook in `rks.py:get_veff()`.

**API sketch:**

```c
// setup (once per geometry)
void SMALL_setup_chi(double *chi[4], int nao, int ngrids, ...);

// per SCF
void SMALL_rho_gga(const double *dm, const double *chi[4],
                   double *rho, int nao, int ngrids, int nthreads);

void SMALL_vmat_gga(const double *chi[4], const double *wv,
                    double *vmat, int nao, int ngrids, int hermi);

// fused
void SMALL_rho_vmat_gga(...);  // one χ pass if libxc done in between — or two-phase
```

Keep `float64` for CPU parity with reference; optional `float32` dispatch later.

---

## 10. Phased implementation plan

### Phase 0 — Baseline & gates (1–2 days)
- [ ] Parity tests: H2O, benzene — ρ, vmat vs `ni.nr_rks` (backend=1)
- [ ] Micro-benchmark: time per component vs threads (1,2,4,8)
- [ ] Document crossover: `nao` where current sparse path wins

### Phase 1 — Layout + precomputed ρ/vmat (highest ROI)
- [ ] Transpose χ to `[iAO, iG]` after `eval_ao` (OpenMP parallel transpose)
- [ ] C kernel: `SMALL_rho_gga_gtile` — gTile parallel, no atomics
- [ ] C kernel: `SMALL_vmat_gga_gtile` — private V_t + reduction
- [ ] Wire thin Python wrapper; call libxc between ρ and vmat
- [ ] Target: beat dense `lib.ddot` path for nao < 200

### Phase 2 — Fusion & specialization
- [ ] Fused ρ+vmat where libxc allows (or fused wv application)
- [ ] Atom-blocked χ layout + pair screening
- [ ] Static dispatch: `nao ≤ 24`, `≤ 64`, `≤ 128` unrolled kernels
- [ ] SIMD intrinsics on inner `μ,ν` for fixed small nao

### Phase 3 — AO evaluation (optional, bigger scope)
- [ ] gTile-parallel AO eval wrapper around libcint (or)
- [ ] CPU Hermite on-the-fly port from OpenCL logic (`rho_*_pair` structure)
- [ ] Decide precomp vs OTF per molecule size / memory budget

### Phase 4 — SCF integration
- [ ] Hook in `get_veff()` or `numint.nr_rks` with auto-dispatch
- [ ] `setup_small_cpu()` mirroring `setup_gpu()` — cache χ across SCF cycles
- [ ] Profile full SCF: compare 1 vs N threads, vs OpenCL backend

---

## 11. Verification strategy

| Check | Tolerance |
|-------|-----------|
| ρ vs `ni.nr_rks` | max abs ~1e-10, rel ~1e-8 |
| vmat vs reference | max abs ~1e-9 (GGA symmetrization) |
| nelec, exc | match libxc path |
| Thread safety | 1-thread vs N-thread bitwise identical (reduction order fixed) |

Run with `backend=3` (CPU new + CPU old) during development, same pattern as GPU parity.

---

## 12. Key design decisions (summary)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | **C + OpenMP** | PySCF convention; no GIL; fusion; SIMD |
| Primary parallel axis | **`gTile` (grid tiles)** | Disjoint ρ[g]; enables χ streaming with stride-1 |
| vmat accumulation | **Private V_t per thread + reduction** | nao² small; avoids atomics; enables fusion |
| Memory layout | **`χ[iAO, iG]` C-contiguous in iG** | Memory-bound; matches OpenCL A1 layout |
| AO strategy (phase 1) | **Precompute + transpose at setup** | Avoids libcint rewrite; χ reused every SCF |
| AO strategy (phase 3) | **Hermite OTF atom-pair** | Port OpenCL kernel structure to CPU |
| Python role | **Setup + libxc + dispatch only** | No hot-loop slicing |
| Atomics | **Never in hot loop** | Reduction after parallel region only |

---

## 13. Relation to OpenCL work

Your OpenCL `rho_gga_pair` / `vmat_gga_pair` kernels are the **algorithmic spec** for the CPU port:

| OpenCL concept | CPU equivalent |
|----------------|----------------|
| `gTile`, `NPTILE` | `gTile` with `NPTILE=32..64` |
| `__local dm_blk` | L1/stack `DM_blk[nao_i][nao_j]` |
| `__local wfRj[NPTILE][NATILE][MAX_SHELL]` | stack arrays per thread |
| Coalesced `χ[iAO, iG]` | same layout, prefetch/stream |
| One workgroup per gTile | one OpenMP task per gTile |
| `psum` tree reduction in local mem | thread-private ρ, no cross-thread reduction needed |

The CPU path won't match GPU throughput but should achieve **near-linear scaling to 4–8 cores** on small molecules — the thing current PySCF cannot do.

---

I can turn this into a `doc/small_mol_cpu_grid_plan.md` in the repo, or go deeper on one piece next (e.g. exact fused kernel pseudocode for GGA vmat, or the atom-blocked index formulas matching your OpenCL headers).

---

Yes. There are still good CPU optimization opportunities, and the biggest one is not “more threads” yet. It is fixing memory layout inside the C kernels so each thread has stride-1 work.

I followed `cpu-perf` workflow as far as possible: `perf stat` is blocked on this machine (`perf_event_paranoid=4`), so I used source inspection, current timings, and GCC vectorization diagnostics.

## Highest-Value Opportunities

### 1. Rewrite tile loops to use stride-1 over grid points

Current C kernels keep PySCF’s F-contiguous AO layout, which is good for BLAS tile GEMM. But some scalar post-processing loops walk AO index `mu` for a fixed grid point `g`, which means huge-stride memory access:

```34:49:pyscf/lib/smalldft/small_grid.c
static void _rho_tile_lda(double *rho, const double *chi0, const double *dm,
                          double *c0, int tile, int nao, int ig0, int ngrids)
{
        // ...
        for (t = 0; t < tile; t++) {
                rho[ig0 + t] = ddot_(&nao, chi0 + ig0 + t, &inc_chi,
                                     c0 + t, &inc_c0);
        }
}
```

This is cache-unfriendly: `inc_chi = ngrids`, so each AO component jumps by ~143k doubles for benzene. GCC also reported these loops as not vectorized.

Better structure:

```c
// pseudo
rho0[tile] = 0
rho1[tile] = 0
rho2[tile] = 0
rho3[tile] = 0

for mu in nao:
    chi0_mu = chi0 + mu*ngrids + ig0
    c0_mu   = c0   + mu*tile
    chi1_mu = chi1 + mu*ngrids + ig0
    ...
    #pragma omp simd
    for t in tile:
        rho0[t] += chi0_mu[t] * c0_mu[t]
        rho1[t] += 2 * chi1_mu[t] * c0_mu[t]
        rho2[t] += 2 * chi2_mu[t] * c0_mu[t]
        rho3[t] += 2 * chi3_mu[t] * c0_mu[t]
```

This makes the inner loop stride-1 and SIMD-friendly. It should improve `rho_gga`, currently ~20 ms @8 CPU.

### 2. Store `aow` / `chi_w` as F-order tile buffers

`vmat_gga` currently builds `aow[t*nao + mu]`, then calls BLAS with `dgemm("T","T")`:

```235:249:pyscf/lib/smalldft/small_grid.c
for (t = 0; t < tile; t++) {
        int g = ig0 + t;
        for (mu = 0; mu < nao; mu++) {
                double val = 0.;
                for (c = 0; c < 4; c++) {
                        val += wv[(size_t)c * ngrids + g]
                             * chi[(size_t)c * ao_size + g
                                   + (size_t)mu * ngrids];
                }
                aow[(size_t)t * nao + mu] = val;
        }
}

dgemm_("T", "T", &nao, &nao, &tile, &one,
       chi + ig0, &ngrids, aow, &nao, &one, v_priv, &nao);
```

This again reads `chi` with poor stride in the inner loop. Better: store `aow` as F-order `(tile, nao)` using `aow[t + mu*tile]`, loop `mu` outside and `t` inside, then call:

```c
dgemm_("T", "N", &nao, &nao, &tile, &one,
       chi + ig0, &ngrids, aow, &tile, &one, v_priv, &nao);
```

This should improve vmat formation and make the non-BLAS part vectorize cleanly. It also removes the awkward Fortran/C transpose interpretation.

### 3. Fuse C rho + libxc/PBE + vmat per tile

Today the path is:

```text
C rho_gga → Python libxc full-grid call → C vmat_gga
```

That forces two full AO passes plus full `rho`/`wv` materialization. True fusion requires moving XC evaluation into the tile loop:

```text
for tile:
    c0 = chi0 @ DM
    rho(tile)
    PBE vxc(tile)
    aow(tile)
    V_thread += chi0.T @ aow
```

This is likely the largest remaining structural win after stride-1 fixes. But it needs either:

- call libxc C API safely per tile, or
- implement the unpolarized PBE formula in C with parity tests.

For production PBE, this is probably worth it. It would cut memory traffic and Python orchestration.

### 4. Make `GridWorkspace` the default patched path

Current numbers show this clearly:

| path @8 CPU | cycle `nr_rks` |
|-------------|---------------:|
| ref | ~120 ms |
| smallDFT_ws | ~42 ms |

The workspace avoids AO inside `nr_rks`. But this only helps when users explicitly build/use `GridWorkspace`. `patch.enable()` should attach and reuse workspace on `mf` automatically per geometry/grid.

This is not a low-level CPU trick, but it is a big real-path win.

### 5. Remove heap allocation from C hot calls

`small_grid.c` allocates per-thread buffers inside each call: `malloc/free` for `c0`, `aow`, `chi_w`, `v_priv`, and hermi temp. This is not the main cost for benzene, but it violates the CPU-perf rule and will matter over many SCF cycles.

Better options:

- pass scratch buffers from `GridWorkspace`
- or introduce a C workspace object with per-thread scratch:
  - `scratch_tile[nthreads][tile*nao]`
  - `scratch_vmat[nthreads][nao*nao]`
  - aligned to 64 bytes
  - first-touch initialized inside parallel region

### 6. Build with optional `-march=native`

GCC diagnostics showed only 16-byte vectorization in key loops. The standalone build uses:

```bash
gcc -shared -fPIC -O3 -fopenmp -std=c99
```

For local performance builds, add optional:

```bash
-march=native -mtune=native
```

Not portable as default, but useful behind e.g. `SMALLDFT_NATIVE=1`.

## Priority Order

1. **Rewrite rho/vmat tile loops for stride-1 grid inner loops.** Lowest risk, likely immediate speedup.
2. **Change `aow`/`chi_w` to F-order tile buffers and use `dgemm("T","N")`.**
3. **Auto-attach `GridWorkspace` in `patch.enable()`**, so real SCF always gets AO caching.
4. **Preallocate C scratch buffers** to remove malloc/free and support first-touch.
5. **Fused PBE tile kernel** (`rho → PBE → vmat`) for the next big jump.
6. **Grid-parallel AO / eval_gto** if we need better first-geometry or moving-geometry performance.

The next concrete implementation I’d do is the stride-1 rewrite of `SMALL_rho_gga` and `SMALL_vmat_gga`, then run parity + `profile_xc_bottleneck()` before touching fusion.

---

## Test machine (benchmark host)

Recorded **2026-07-08** on host `GTX3090`. All CPU smallDFT timings in this thread and in `doc/CPU_benchmark.md` were measured on this system unless noted.

### Host OS

| Item | Value |
|------|-------|
| Hostname | `GTX3090` |
| Kernel | Linux `6.8.0-51-generic` (`#52-Ubuntu SMP PREEMPT_DYNAMIC`) |
| Arch | x86_64, little-endian |
| `perf` | **`perf_event_paranoid=4`** — hardware counters blocked; profiling used wall time, GCC `-fopt-info-vec`, cProfile |

### CPU — AMD Ryzen 7 5800X (Zen 3, Vermeer)

| Item | Value |
|------|-------|
| Model | **AMD Ryzen 7 5800X 8-Core Processor** |
| Family / model / stepping | 25 / 33 / 0 |
| Sockets | 1 |
| Cores / threads | **8 cores / 16 threads** (SMT on) |
| NUMA | 1 node (CPUs 0–15) |
| Base / boost | min **2.2 GHz**; max **~4.85 GHz** (boost enabled) |
| Governor | `schedutil` (sampled ~4.2–4.6 GHz under load) |
| ISA (relevant) | AVX2, FMA, BMI1/2, AES, SHA, CLZERO |

**Cache hierarchy** (`lscpu`):

| Level | Size | Notes |
|-------|------|-------|
| L1d | **256 KiB** | 8× 32 KiB (per core) |
| L1i | **256 KiB** | 8× 32 KiB (per core) |
| L2 | **4 MiB** | 8× 512 KiB (per core) |
| L3 | **32 MiB** | shared, 1 instance |

**Relevance for smallDFT:** benzene GGA χ is ~525 MB (`4 × nao × ngrids × 8`); does not fit L3 — kernels are **memory-bandwidth bound**. DM (`nao²`) and tile buffers (`TILE×nao`, `TILE=512`) fit in L2/L3 and should be reused across the inner grid loop.

### Memory

| Item | Value |
|------|-------|
| RAM | **32 GiB** (`MemTotal` 31.3 GiB) |
| Swap | none |
| Typical free under load | ~18 GiB available (buff/cache heavy) |

Working sets for benzene SCF (χ + DM + vmat + scratch) are well below RAM; no paging concern.

### Storage

| Item | Value |
|------|-------|
| Primary | **WD SN850 2 TB** NVMe (`WDS200T1X0E-00AFY0`, `nvme0n1`) |

Not a bottleneck for in-memory grid XC; relevant only for build I/O and large ref data.

### Threading defaults in CPU benchmarks

| Knob | Value |
|------|-------|
| Grid / libcint OMP | `lib.num_threads(N)` — authoritative for `libsmalldft` + libcint |
| BLAS | **`OPENBLAS_NUM_THREADS=1`** (avoid nested OMP + threaded GEMM) |
| Typical sweeps | N = 1, 2, 4, 8 (physical cores); 16 threads rarely faster for grid kernels |

### GPU on same host (context only)

OpenCL benchmarks on this machine use **NVIDIA GeForce RTX 3090** (24 GB, 82 CUs). See `doc/GPU_benchmark.md` § Test machine for full GPU stack. CPU smallDFT work does not use the GPU.

### Reproduce machine query

```bash
lscpu
free -h
grep -E 'model name|cpu MHz' /proc/cpuinfo | head -2
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq   # kHz
uname -a
```

Full `lscpu` snapshot (2026-07-08):

```
Architecture:            x86_64
CPU(s):                  16
Model name:              AMD Ryzen 7 5800X 8-Core Processor
Thread(s) per core:      2
Core(s) per socket:      8
Socket(s):               1
CPU max MHz:             4850.19
CPU min MHz:             2200.00
L1d cache:               256 KiB (8 instances)
L1i cache:               256 KiB (8 instances)
L2 cache:                4 MiB (8 instances)
L3 cache:                32 MiB (1 instance)
NUMA node(s):            1
Flags:                   ... avx2 fma bmi2 ...
```